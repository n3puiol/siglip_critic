from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import os
import cv2
from torch.utils.data import Dataset
import numpy as np
import pickle
from .rawvideo_util import RawVideoExtractor


class SSv2_DataLoader(Dataset):
    """Something-Something v2 dataset loader."""

    def __init__(
        self,
        subset,
        data_path,
        features_path,
        tokenizer,
        max_words=30,
        feature_framerate=1.0,
        max_frames=100,
        image_resolution=224,
        frame_order=0,
        slice_framepos=0,
        is_test=False,
        video_max_len=-1,
    ):
        self.video_max_len = video_max_len
        self.is_test = is_test
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        train_path = os.path.join(self.data_path, "something-something-v2-train.json")
        val_path = os.path.join(
            self.data_path, "something-something-v2-validation.json"
        )
        test_path = os.path.join(self.data_path, "something-something-v2-test.json")
        video_id_path_dict["train"] = self._load_split_from_json(train_path)
        video_id_path_dict["val"] = self._load_split_from_json(val_path)
        video_id_path_dict["test"] = self._load_split_from_json(test_path)
        # Test data doesn't have captions
        # video_id_path_dict["test"] = self._load_split_from_json(os.path.join(self.data_path, "something-something-v2-test.json"))
        captions = self._load_captions_from_json([train_path, val_path])
        # caption_file = os.path.join(self.data_path, "raw-captions.pkl")

        video_ids = [itm.strip() for itm in video_id_path_dict[self.subset]]

        # with open(caption_file, "rb") as f:
        #     captions = pickle.load(f)

        video_dict = {}
        for video_id in video_ids:
            video_dict[video_id] = os.path.join(self.features_path, video_id + ".webm")
        # os.walk is very slow for large dataset.
        # for root, dub_dir, video_files in os.walk(self.features_path):
        #     for video_file in video_files:
        #         video_id_ = ".".join(video_file.split(".")[:-1])
        #         if video_id_ not in video_ids:
        #             continue
        #         file_path_ = os.path.join(root, video_file)
        #         video_dict[video_id_] = file_path_
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

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }
        self.video_cache = {}

    def _load_split_from_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        data = [v["id"] for v in data]
        return data

    def _load_captions_from_json(self, paths):
        captions = {}
        for path in paths:
            with open(path, "r") as f:
                data = json.load(f)
            for vid in data:
                captions[vid["id"]] = [vid["label"].split(" ")]
        return captions

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
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

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
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
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
            dtype=np.float,
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
            else:
                video_path = video_input

            # Do not cache in validation dataloader since it's done in do_train.
            # if self.subset == "train" and video_path in self.video_cache:
            if video_path in self.video_cache:
                raw_video_data = self.video_cache[video_path]
            else:
                raw_video_data = self.rawVideoExtractor.get_video_data(
                    video_path, start_time=start_time, end_time=end_time
                )
                raw_video_data = raw_video_data["video"]
            # if video_path not in self.video_cache:
            #    self.video_cache[video_path] = raw_video_data

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(
                    raw_video_data_clip
                )
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[: self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames :, ...]
                    else:
                        sample_indx = np.linspace(
                            0,
                            raw_video_slice.shape[0] - 1,
                            num=self.max_frames,
                            dtype=int,
                        )
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(
                    video_slice, frame_order=self.frame_order
                )

                slice_len = video_slice.shape[0]
                max_video_length[i] = (
                    max_video_length[i]
                    if max_video_length[i] > slice_len
                    else slice_len
                )
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

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
            return pairs_text, pairs_mask, pairs_segment, video, video_mask
