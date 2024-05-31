import io
import torch as th
import numpy as np
from PIL import Image

# pip install albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# pip install opencv-python
import cv2


def convert_to_rgb(image):
    return image.convert("RGB")


class RawVideoExtractorCV2:
    def __init__(
        self,
        centercrop=False,
        size=224,
        framerate=-1,
        random_augment=False,
        efficient_subsample=False,
    ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.efficient_subsample = efficient_subsample
        if random_augment:
            self.transform = self._albumentations_random_transform(self.size)
        else:
            self.transform = self._transform(self.size)
        self.preprocess_together = random_augment

    def _transform(self, n_px):
        return Compose(
            [
                Resize(n_px, interpolation=Image.Resampling.BICUBIC),
                CenterCrop(n_px),
                convert_to_rgb,
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def _albumentations_random_transform(self, n_px):
        # Use n_px + 25% center crop.
        centercrop_size = int(np.round(n_px * 1.25))
        min_max_height = (centercrop_size * 0.85, centercrop_size * 1)
        return A.Compose(
            [
                A.Rotate(),
                A.GridDistortion(),
                A.Perspective(scale=0.07),
                A.SmallestMaxSize(centercrop_size),
                A.CenterCrop(width=centercrop_size, height=centercrop_size),
                # Randomize aspect ratio and zoom with probability p, otherwise keep center crop.
                A.Sequential(
                    [
                        # Randomly changes aspect ratio.
                        A.RandomResizedCrop(
                            height=centercrop_size,
                            width=centercrop_size,
                            scale=(0.85, 1.0),
                            ratio=(0.7, 1.4),
                        ),
                        # Randomly changes zoom.
                        A.RandomSizedCrop(
                            height=n_px,
                            width=n_px,
                            min_max_height=min_max_height,
                            p=0.9,
                        ),
                    ],
                    p=0.9,
                ),
                A.Resize(height=n_px, width=n_px),
                # Color variations.
                A.RGBShift(),
                A.HueSaturationValue(),
                A.RandomToneCurve(),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.3),
                A.Posterize(p=0.1),
                # Blur variations.
                A.OneOf(
                    [
                        A.Blur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.MotionBlur(blur_limit=5),
                        A.ZoomBlur(max_factor=1.2),
                    ],
                    p=0.2,
                ),
                A.GaussNoise(var_limit=(10.0, 80.0)),
                A.ToFloat(max_value=255),
                A.Normalize(max_pixel_value=1.0),
                ToTensorV2(),
            ],
            additional_targets={f"image{i}": "image" for i in range(200)},
        )

    def load_video(
        self,
        video_file,
        sample_fp=0,
        start_time=None,
        end_time=None,
    ):
        if start_time is not None or end_time is not None:
            assert (
                isinstance(start_time, int)
                and isinstance(end_time, int)
                and start_time > -1
                and end_time > start_time
            )
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = (
                start_time,
                end_time if end_time <= total_duration else total_duration,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0:
            interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret:
                break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                images.append(frame_rgb)

        cap.release()
        images = np.stack(images)
        return images

    def load_subsampled_video(
        self, video_file, max_frames, slice_framepos, start_time=None, end_time=None, frame_indices_to_use=None,
    ):
        """Load only a subset of video frames.

        Limitation: Currently no support for sample_fp (could be useful with
        slice_framepos 0 or 1).
        """
        if start_time is not None or end_time is not None:
            assert (
                isinstance(start_time, int)
                and isinstance(end_time, int)
                and start_time > -1
                and end_time > start_time
            )

        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frame_count + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = (
                start_time,
                end_time if end_time <= total_duration else total_duration,
            )

        indices_to_load = []
        start_idx = int(start_sec * fps)
        end_idx = min(frame_count, int(end_sec * fps))
        if max_frames < end_idx - start_idx:
            if slice_framepos == 0:
                indices_to_load = np.arange(start_idx, start_idx + max_frames)
            elif slice_framepos == 1:
                indices_to_load = np.arange(end_idx - max_frames, end_idx)
            elif slice_framepos == 2:
                indices_to_load = np.linspace(
                    start_idx,
                    end_idx - 1,
                    num=max_frames,
                    dtype=int,
                )
            else:
                # Split the video into uniform intervals, and sample from each interval.
                linspace = np.linspace(
                    start_idx,
                    end_idx,
                    num=max_frames + 1,
                    dtype=int,
                )
                indices_to_load = np.zeros(max_frames, dtype=int)
                for idx, space in enumerate(linspace[:-1]):
                    indices_to_load[idx] = np.random.randint(space, linspace[idx + 1])
        else:
            indices_to_load = np.arange(start_idx, end_idx)
        if frame_indices_to_use is not None:
            indices_to_load = np.stack([indices_to_load[i] for i in frame_indices_to_use])

        images = []
        for idx in indices_to_load:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(frame_rgb)

        cap.release()
        images = np.stack(images)
        return images

    def transform_video(self, images, preprocess, preprocess_together):
        orig_shape = images.shape
        images = images.reshape(-1, *orig_shape[-3:])
        if preprocess_together:
            images = {
                f"image{i - 1}" if i > 0 else f"image": img
                for i, img in enumerate(images)
            }
            images = preprocess(**images)
            images = np.stack(
                [
                    images[f"image{i - 1}"] if i > 0 else images["image"]
                    for i in range(len(images))
                ]
            )
        else:
            images = np.stack(
                [preprocess(Image.fromarray(img).convert("RGB")) for img in images]
            )
        new_shape = images.shape
        images = images.reshape(*orig_shape[:-3], *new_shape[1:])
        return images

    def convert_video_to_tensor(self, images):
        if len(images) > 0:
            images = th.tensor(np.stack(images))
        else:
            images = th.zeros(1)
        return images

    def get_video_data(self, video_path, start_time=None, end_time=None, augment=True):
        if isinstance(video_path, str):
            image_input = self.load_video(
                video_path,
                sample_fp=self.framerate,
                start_time=start_time,
                end_time=end_time,
            )
            image_input = self.transform_video(
                image_input, self.transform, self.preprocess_together
            )
            image_input = self.convert_video_to_tensor(image_input)
            image_input = {"video": image_input}
        else:
            image_input = self.get_image_input_from_numpy(video_path)
        return image_input

    def subsample_video(self, raw_video_slice, max_frames, slice_framepos, frame_indices_to_use=None):
        # L x T x 3 x H x W
        if max_frames < len(raw_video_slice):
            if slice_framepos == 0:
                video_slice = raw_video_slice[:max_frames, ...]
            elif slice_framepos == 1:
                video_slice = raw_video_slice[-max_frames:, ...]
            elif slice_framepos == 2:
                sample_indx = np.linspace(
                    0,
                    len(raw_video_slice) - 1,
                    num=max_frames,
                    dtype=int,
                )
                video_slice = raw_video_slice[sample_indx, ...]
            else:
                # Split the video into uniform intervals, and sample from each interval.
                linspace = np.linspace(
                    0,
                    len(raw_video_slice),
                    num=max_frames + 1,
                    dtype=int,
                )
                sample_indx = np.zeros(max_frames, dtype=int)
                for idx, space in enumerate(linspace[:-1]):
                    sample_indx[idx] = np.random.randint(space, linspace[idx + 1])
                video_slice = raw_video_slice[sample_indx, ...]
        else:
            video_slice = raw_video_slice
        if frame_indices_to_use is not None:
            video_slice = np.stack([video_slice[i] for i in frame_indices_to_use])
        return video_slice

    def decode_video(self, video_bytes):
        video = np.array(
            [Image.open(io.BytesIO(image_bytes)) for image_bytes in video_bytes]
        )
        return video

    def sample_video_data(
        self,
        video_path,
        max_frames,
        slice_framepos,
        frame_order,
        start_time=None,
        end_time=None,
        raw_video_data=None,
        frame_indices_to_use=None,
    ):
        """Loads, subsamples and augments the video (if applicable)."""
        if raw_video_data is None:
            if self.efficient_subsample:
                video_slice = self.load_subsampled_video(
                    video_path,
                    max_frames,
                    slice_framepos,
                    start_time=start_time,
                    end_time=end_time,
                    frame_indices_to_use=frame_indices_to_use,
                )
                if len(video_slice.shape) <= 3:
                    raise RuntimeError("video path: {} error.".format(video_path))
            else:
                raw_video_data = self.load_video(
                    video_path,
                    sample_fp=self.framerate,
                    start_time=start_time,
                    end_time=end_time,
                )
                if len(raw_video_data.shape) <= 3:
                    raise RuntimeError("video path: {} error.".format(video_path))
                video_slice = self.subsample_video(
                    raw_video_data, max_frames, slice_framepos, frame_indices_to_use
                )
        else:
            raw_video_data = np.array(raw_video_data)
            video_slice = self.subsample_video(
                raw_video_data, max_frames, slice_framepos, frame_indices_to_use
            )
            if video_slice.dtype != np.uint8:
                video_slice = self.decode_video(video_slice)
        video_slice = self.process_raw_data(video_slice)
        video_slice = self.transform_video(
            video_slice,
            self.transform,
            preprocess_together=self.preprocess_together,
        )
        video_slice = self.convert_video_to_tensor(video_slice)
        video_slice = self.process_frame_order(video_slice, frame_order=frame_order)

        return video_slice

    def process_video_data(
        self,
        video,
        max_frames,
        slice_framepos,
        frame_order,
        start_time=None,
        end_time=None,
        frame_indices_to_use=None
    ):
        """Loads, subsamples and augments the video (if applicable)."""
        video_slice = self.subsample_video(video, max_frames, slice_framepos, frame_indices_to_use)
        video_slice = self.transform_video(
            video_slice.astype(np.uint8),
            self.transform,
            preprocess_together=self.preprocess_together,
        )
        video_slice = self.convert_video_to_tensor(video_slice)
        video_slice = self.process_frame_order(video_slice, frame_order=frame_order)

        return video_slice

    def get_image_input_from_numpy(self, video_path):
        video = np.array(video_path)
        video = np.transpose(video, (0, 3, 1, 2))
        return {"video": th.tensor(video)}

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.shape
        tensor = raw_video_data.reshape(
            -1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1]
        )
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data


# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2
