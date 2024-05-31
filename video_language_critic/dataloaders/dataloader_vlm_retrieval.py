from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from torch.utils.data import Dataset
import numpy as np
import pickle
from .rawvideo_util import RawVideoExtractor


class VLM_DataLoader(Dataset):
    """VLM dataset loader."""

    def __init__(
        self,
        subset,
        data_path,
        features_path,
        tokenizer,
        max_words=30,
        feature_framerate=1,
        max_frames=100,
        image_resolution=224,
        frame_order=0,
        slice_framepos=0,
        frame_indices_to_use=None,
        efficient_subsample=False,
        is_test=False,
        video_max_len=-1,
        use_failures_as_negatives_only=False,
        augment_images=False,
        load_data=True,
    ):
        self.video_max_len = video_max_len
        self.is_test = is_test
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames_for_sampler = max_frames
        self.max_frames = len(frame_indices_to_use) if frame_indices_to_use is not None else max_frames
        self.frame_indices_to_use = frame_indices_to_use
        self.tokenizer = tokenizer
        self.negative_captions = (
            (
                "Perform failed grasp",
                "Do nothing",
                # "No caption",
            )
            if use_failures_as_negatives_only
            else ()
        )
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly,
        # 3: sample from uniform intervals.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2, 3]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
        caption_file = os.path.join(self.data_path, "raw-captions.pkl")

        video_ids = []
        if load_data:
            with open(video_id_path_dict[self.subset], "r") as fp:
                video_ids = [itm.strip() for itm in fp.readlines()]

            with open(caption_file, "rb") as f:
                captions = pickle.load(f)

        video_dict = {}
        if load_data:
            for root, dub_dir, video_files in os.walk(self.features_path):
                for video_file in video_files:
                    video_id_ = ".".join(video_file.split(".")[:-1])
                    if video_id_ not in video_ids:
                        continue
                    file_path_ = os.path.join(root, video_file)
                    video_dict[video_id_] = file_path_
        self.video_dict = video_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in video_ids:
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))
        if video_max_len > 0:
            self.update_sentences_dict()

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = False  # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence count: {}".format(self.subset, self.sentence_num))
            print("For {}, video count: {}".format(self.subset, self.video_num))

        print("Video count: {}".format(len(self.video_dict)))
        print("Total pairs: {}".format(len(self.sentences_dict)))

        self.augment_images = not is_test and augment_images
        self.sample_len = len(self.sentences_dict)
        framerate = 0 if self.slice_framepos == 3 else feature_framerate
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=framerate,
            size=image_resolution,
            random_augment=self.augment_images,
            efficient_subsample=efficient_subsample,
        )
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

    def update_sentences_dict(self):
        print(
            f"Slicing every video into max length of {self.video_max_len} frames. This may take a while."
        )
        sentences_dict_new = {}
        for idx, info in self.sentences_dict.items():
            (video_id, caption) = info
            cap = cv2.VideoCapture(self.video_dict[video_id])
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            slice_amount = int(round(frameCount / self.video_max_len))
            for idx in range(slice_amount):
                sentences_dict_new[len(sentences_dict_new)] = (
                    {
                        "video_id": video_id,
                        "start_time": int(
                            np.floor(idx * self.video_max_len / fps)
                        ),  # producing overlaps with floor and ceil
                        "end_time": int(np.ceil((idx + 1) * self.video_max_len / fps)),
                    },
                    caption,
                )
        self.sentences_dict = sentences_dict_new

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.compat.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.compat.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.compat.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)

            # tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            # sentence = self.tokenizer.tokens_to_sentence(tokens)

            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids, video_input=None):
        video_mask = np.zeros(
            (len(choice_video_ids), self.max_frames), dtype=np.compat.long
        )
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros(
            (
                len(choice_video_ids),
                self.max_frames,
                1,
                3,
                self.rawVideoExtractor.size,
                self.rawVideoExtractor.size,
            ),
            dtype=float,
        )

        for i, video_id in enumerate(choice_video_ids):
            start_time = None
            end_time = None

            if video_input is None:
                if isinstance(video_id, str):
                    video_path = self.video_dict[video_id]
                else:
                    video_path = self.video_dict[video_id["video_id"]]
                    start_time = video_id["start_time"]
                    end_time = video_id["end_time"]
                video_slice = self.rawVideoExtractor.sample_video_data(
                    video_path,
                    self.max_frames_for_sampler,
                    self.slice_framepos,
                    self.frame_order,
                    start_time=start_time,
                    end_time=end_time,
                    frame_indices_to_use=self.frame_indices_to_use,
                )
            else:
                # video_path = video_input

                video_slice = self.rawVideoExtractor.process_video_data(
                    video_input,
                    self.max_frames_for_sampler,
                    2,  # self.slice_framepos,
                    0,  # self.frame_order,
                    start_time=start_time,
                    end_time=end_time,
                    frame_indices_to_use=self.frame_indices_to_use,
                )

            slice_len = video_slice.shape[0]
            max_video_length[i] = (
                max_video_length[i] if max_video_length[i] > slice_len else slice_len
            )
            if slice_len < 1:
                pass
            else:
                video[i][:slice_len, ...] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def get_raw_data(self, idx):
        video_id, caption = self.sentences_dict[idx]
        return video_id, caption

    def get_encoded_data(self, video, instruction):
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(
            video_id=0, caption=instruction
        )
        video, video_mask = self._get_rawvideo(choice_video_ids, video_input=video)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        if caption in self.negative_captions:
            pairs_text = np.zeros((1, self.max_words), dtype=np.compat.long)
            pairs_mask = np.zeros_like(pairs_text)
            pairs_segment = np.zeros_like(pairs_text)
            choice_video_ids = [video_id]
            # caption = "" if caption in self.negative_captions else caption
        else:
            pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(
                video_id, caption
            )

        video, video_mask = self._get_rawvideo(choice_video_ids)
        if self.is_test:
            return (
                pairs_text,
                pairs_mask,
                pairs_segment,
                video,
                video_mask,
                video_id,
                caption,
            )
        else:
            return (
                pairs_text,
                pairs_mask,
                pairs_segment,
                video,
                video_mask,
                video_id,
                caption,
            )
