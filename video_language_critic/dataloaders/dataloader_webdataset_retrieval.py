from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import glob
import os
import string
from torch.utils.data import IterableDataset
import numpy as np
from .rawvideo_util import RawVideoExtractor

import webdataset as wds


class WebDatasetLoader(IterableDataset):
    """WebDataset data loader."""

    def __init__(
        self,
        subset,
        data_path,
        tokenizer,
        max_words=30,
        feature_framerate=1.0,
        max_frames=100,
        image_resolution=224,
        frame_order=0,
        slice_framepos=0,
        is_test=False,
        video_max_len=-1,
        seed=0,
        augment_images=False,
    ):
        self.video_max_len = video_max_len
        self.is_test = is_test
        self.data_path = data_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.negative_captions = ()
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate,
            size=image_resolution,
            random_augment=not is_test and augment_images,
        )
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

        with open(os.path.join(self.data_path, f"{subset}_size.txt")) as f:
            split_size = int(f.read().strip())
        self.split_size = split_size

        self.np_random = np.random.RandomState(seed)
        self.np_random2 = np.random.RandomState(seed + 1)
        urls = glob.glob(os.path.join(self.data_path, f"{subset}-*.tar"))
        dataset = (
            wds.WebDataset(urls)
            .decode(wds.torch_video)
            .decode(self.bytes_decoder, partial=True)
        )
        # if self.is_test:
        # TODO: Multiple caption evaluation.
        self.multi_sentence_per_video = False  # !!! important tag for eval
        dataset = dataset.map(self.sample_viewpoint)
        dataset = dataset.map(self.sample_caption)

        dataset = dataset.map(self._get_text)
        dataset = dataset.map(self._get_rawvideo)

        if self.is_test:
            dataset = dataset.to_tuple(
                "pairs_text",
                "pairs_mask",
                "pairs_segment",
                "video",
                "video_mask",
                "video_id",
                "caption",
            )
        else:
            dataset = dataset.to_tuple(
                "pairs_text", "pairs_mask", "pairs_segment", "video", "video_mask"
            )

        self.dataset = dataset

    def bytes_decoder(self, key, value):
        if key == ".video_id":
            return value.decode()
        else:
            return value

    def sample_viewpoint(self, sample):
        viewpoints = [
            k for k in sample.keys() if k.startswith("images") and k.endswith(".mp4")
        ]
        if self.is_test:
            key = viewpoints[0]
        key = self.np_random.choice(viewpoints)
        # Decoded video is a tuple with video, empty array, and fps dictionary.
        sample["video"] = sample[key][0]
        for k in viewpoints:
            del sample[k]
        return sample

    def sample_caption(self, sample):
        caption_ks = [
            k for k in sample.keys() if k.startswith("caption") and k.endswith(".txt")
        ]
        if self.is_test:
            if self.multi_sentence_per_video:
                sample["caption"] = [sample[k] for k in caption_ks]
            else:
                sample["caption"] = sample[caption_ks[0]]
        else:
            key = self.np_random2.choice(caption_ks)
            sample["caption"] = sample[key]
        for k in caption_ks:
            del sample[k]
        return sample

    def __len__(self):
        return self.split_size

    def filter_punctuation(self, caption):
        punctuations = set(
            [
                '"',
                "!",
                "\\",
                "`",
                " ",
                ".",
                ",",
                "-",
                "'",
                "?",
                ":",
                ";",
                "(",
                ")",
            ]
        )
        while caption[0] in punctuations:
            caption = caption[1:]
        while caption[-1] in punctuations:
            caption = caption[:-1]
        return caption

    def _get_text(self, sample):
        caption = sample["caption"]
        video_id = sample["video_id"]
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.compat.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.compat.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.compat.long)

        for i, video_id in enumerate(choice_video_ids):
            # TODO: Save this preprocessing to the dataset.
            caption = self.filter_punctuation(caption)
            # Log captions with unexpected punctuation.
            # if (
            #     # Allowed punctuation.
            #     caption.replace("'", "")
            #     .replace(",", "")
            #     .replace("-", "")
            #     .replace(".", "")
            #     .replace("/", "")
            #     != caption.translate(str.maketrans("", "", string.punctuation))
            # ):
            #     print(caption)
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

        sample["pairs_text"] = pairs_text
        sample["pairs_mask"] = pairs_mask
        sample["pairs_segment"] = pairs_segment
        sample["caption"] = caption
        return sample

    def _get_rawvideo(self, sample):
        choice_video_ids = [sample["video_id"]]
        video_input = sample["video"]
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
            else:
                video_path = video_input

            raw_video_data = self.rawVideoExtractor.get_video_data(
                video_path, start_time=start_time, end_time=end_time
            )
            raw_video_data = raw_video_data["video"]

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

        sample["video"] = video
        sample["video_mask"] = video_mask
        return sample

    # def get_encoded_data(self, video, instruction):
    #     pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(
    #         video_id=0, caption=instruction
    #     )
    #     video, video_mask = self._get_rawvideo(choice_video_ids, video_input=video)

    #     return pairs_text, pairs_mask, pairs_segment, video, video_mask

    def __getattr__(self, name):
        return getattr(self.dataset, name)
