from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from video_language_critic.dataloaders.dataloader_vlm_retrieval import VLM_DataLoader
import numpy as np
import pickle
from .rawvideo_util import RawVideoExtractor


class VLM_ProgressDataLoader(VLM_DataLoader):
    """VLM dataset loader for segments of the form [0, t]."""

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
        is_test=False,
        video_max_len=-1,
        use_failures_as_negatives_only=False,
        n_progress_steps=4,
    ):
        self.n_progress_steps = n_progress_steps
        super().__init__(
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words,
            feature_framerate,
            max_frames,
            image_resolution,
            frame_order=0,
            slice_framepos=2,
            is_test=is_test,
            video_max_len=video_max_len,
            use_failures_as_negatives_only=use_failures_as_negatives_only,
        )

    def _get_rawvideo(self, choice_video_ids, video_input=None):
        video_mask = np.zeros(
            (len(choice_video_ids), self.max_frames, self.n_progress_steps),
            dtype=np.compat.long,
        )
        # Pair x L x T x P x 3 x H x W
        video = np.zeros(
            (
                len(choice_video_ids),
                self.max_frames,
                self.n_progress_steps,
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
                lengths = np.linspace(
                    0,
                    len(raw_video_slice),
                    num=self.n_progress_steps + 1,
                    dtype=int,
                )[1:]
                lengths = np.maximum(1, lengths)
                for j, length in enumerate(lengths):
                    progress_video_slice = raw_video_slice[:length]
                    if self.max_frames < progress_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = progress_video_slice[: self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = progress_video_slice[-self.max_frames :, ...]
                        else:
                            sample_indx = np.linspace(
                                0,
                                progress_video_slice.shape[0] - 1,
                                num=self.max_frames,
                                dtype=int,
                            )
                            video_slice = progress_video_slice[sample_indx, ...]
                    else:
                        video_slice = progress_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(
                        video_slice, frame_order=self.frame_order
                    )

                    slice_len = video_slice.shape[0]
                    if slice_len < 1:
                        pass
                    else:
                        video[i, :slice_len, j : j + 1, ...] = video_slice
                        video_mask[i, :slice_len, j] = [1] * slice_len

            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        return video, video_mask
