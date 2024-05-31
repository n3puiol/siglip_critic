from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import functools
import numpy as np
import torch

from torch.utils.data import IterableDataset
from torchdata.datapipes.iter import FileLister, FileOpener
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from .rawvideo_util import RawVideoExtractor


class TFRecord_DataLoader(IterableDataset):
    """TFRecord dataset loader."""

    def __init__(
        self,
        subset,
        data_files,
        data_length,
        tokenizer,
        max_words=30,
        feature_framerate=1,
        max_frames=100,
        image_resolution=224,
        frame_order=0,
        slice_framepos=0,
        frame_indices_to_use=None,
        is_test=False,
        video_max_len=-1,
        use_failures_as_negatives_only=False,
        augment_images=False,
    ):
        self.video_max_len = video_max_len
        self.is_test = is_test
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames_for_sampler = max_frames
        self.max_frames = len(frame_indices_to_use) if frame_indices_to_use is not None else max_frames
        self.frame_indices_to_use = frame_indices_to_use
        self.tokenizer = tokenizer
        # self.negative_captions = (
        #     ("Perform failed grasp", "No caption")
        #     if use_failures_as_negatives_only
        #     else ("No caption",)
        # )
        self.negative_captions = tuple()
        #     ("Perform failed grasp",) if use_failures_as_negatives_only else ()
        # )
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly,
        # 3: sample from uniform intervals.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2, 3]
        self.augment_images = not is_test and augment_images
        framerate = 0 if self.slice_framepos == 3 else feature_framerate

        self.rawVideoExtractor = RawVideoExtractor(
            framerate=framerate,
            size=image_resolution,
            random_augment=self.augment_images,
        )
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }
        self.subset = subset
        self.data_files = data_files
        self.data_length = data_length

    def __len__(self):
        if self.data_length is None:
            raise TypeError("TFRecord_Dataloader doesn't have a valid length.")
        else:
            return self.data_length

    def drop_unused_fields(self, episode, randomize=False):
        def _get_one_of_fields(episode, keys, randomize=False):
            # Exact matches.
            for k in keys:
                if k in episode:
                    return episode[k], k
            # Multiple variants.
            for k in keys:
                short_k = k.replace("steps/observation/", "")
                matching_keys = [
                    ek
                    for ek in episode.keys()
                    if ek.startswith(f"{k}_") or ek.endswith(f"_{short_k}")
                ]
                if matching_keys:
                    if randomize:
                        key = np.random.choice(matching_keys)
                    else:
                        key = sorted(matching_keys)[0]
                    return episode[key], key

            return None, None

        def decode_instruction(inst):
            return bytes(inst[np.where(inst != 0)].tolist())

        episode_subset = {}
        if "episode_id" in episode:
            episode_subset["video_id"] = episode["episode_id"][0].decode("utf-8")

        video, _ = _get_one_of_fields(
            episode,
            [
                "steps/observation/rgb",
                "steps/observation/image",
                # "steps/observation/front_rgb",
                # "steps/observation/hand_image",
            ],
            randomize=randomize,
        )
        if video is None:
            raise KeyError(f"No image field found. Fields: {episode.keys()}")
        episode_subset["video"] = video

        caption, caption_key = _get_one_of_fields(
            episode,
            [
                "steps/language_instruction",
                "steps/observation/natural_language_instruction",
                "steps/observation/instruction",
            ],
        )
        if caption is None:
            raise KeyError(f"No caption field found. Fields: {episode.keys()}")
        if caption_key == "steps/observation/instruction":
            caption = caption.reshape([-1, 512])
            caption = np.array([decode_instruction(c) for c in caption])
        if randomize:
            caption = np.random.choice(caption)
        else:
            caption = caption[-1]
        caption = caption.decode("utf-8")
        episode_subset["caption"] = caption
        return episode_subset

    def _process_text(self, caption):
        k = 1
        pairs_text = np.zeros((k, self.max_words), dtype=np.compat.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.compat.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.compat.long)

        i = 0
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

        return pairs_text, pairs_mask, pairs_segment

    def _process_rawvideo(self, raw_video):
        video_mask = np.zeros((1, self.max_frames), dtype=np.compat.long)
        max_video_length = [0]

        # Pair x L x T x 3 x H x W
        video = np.zeros(
            (
                1,
                self.max_frames,
                1,
                3,
                self.rawVideoExtractor.size,
                self.rawVideoExtractor.size,
            ),
            dtype=float,
        )

        i = 0
        video_slice = self.rawVideoExtractor.sample_video_data(
            video_path=None,
            max_frames=self.max_frames_for_sampler,
            slice_framepos=self.slice_framepos,
            frame_order=self.frame_order,
            start_time=None,
            end_time=None,
            raw_video_data=raw_video,
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

    def get_encoded_data(self, video, instruction):
        pairs_text, pairs_mask, pairs_segment = self._process_text(instruction)
        video, video_mask = self._process_rawvideo(video)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        print(worker_info)
        data_files = self.data_files

        if worker_info is None:
            files_chunk = data_files
        else:
            n_workers = worker_info.num_workers
            n_files = len(data_files)
            chunk_size = n_files // n_workers
            print("Chunk size:", chunk_size)

            chunk_start = chunk_size * worker_info.id
            files_chunk = data_files[chunk_start : chunk_start + chunk_size]
            # Distribute remaining files evenly.
            if chunk_size * n_workers + worker_info.id < n_files:
                files_chunk.append(data_files[chunk_size * n_workers + worker_info.id])
            print(f"Worker {worker_info.id}: {len(files_chunk)}/{n_files} files.")
            if len(files_chunk) < 10:
                print("chunk:", files_chunk)

        datapipe1 = FileLister(files_chunk)
        datapipe2 = FileOpener(datapipe1, mode="b")
        dataset = datapipe2.load_from_tfrecord()
        select_fields = functools.partial(
            self.drop_unused_fields, randomize=self.slice_framepos == 3
        )
        dataset = dataset.map(select_fields)
        inner_dataset = dataset.map(self.process_sample)
        if self.subset == "train":
            dataset = ShufflerIterDataPipe(inner_dataset, buffer_size=100)
        else:
            dataset = inner_dataset

        return iter(dataset)

    def process_sample(self, episode):
        video = episode["video"]
        caption = episode["caption"]
        video, video_mask = self._process_rawvideo(video)
        video_id = episode["video_id"] if "video_id" in episode else ""

        if caption in self.negative_captions:
            pairs_text = np.zeros((1, self.max_words), dtype=np.compat.long)
            pairs_mask = np.zeros_like(pairs_text)
            pairs_segment = np.zeros_like(pairs_text)
        else:
            pairs_text, pairs_mask, pairs_segment = self._process_text(caption)

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
