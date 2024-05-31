import glob
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from video_language_critic.dataloaders.dataset_info import openx
from .dataloader_msrvtt_retrieval import MSRVTT_DataLoader
from .dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
from .dataloader_msvd_retrieval import MSVD_DataLoader
from .dataloader_vlm_retrieval import VLM_DataLoader
from .dataloader_vlm_progress import VLM_ProgressDataLoader
from .dataloader_ssv2_retrieval import SSv2_DataLoader
from .dataloader_lsmdc_retrieval import LSMDC_DataLoader
from .dataloader_activitynet_retrieval import ActivityNet_DataLoader
from .dataloader_didemo_retrieval import DiDeMo_DataLoader
from .dataloader_tfrecord_retrieval import TFRecord_DataLoader
from .dataloader_webdataset_retrieval import WebDatasetLoader


def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.test_slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_msvd_train(args, tokenizer):
    msvd_dataset = MSVD_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler


def dataloader_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MSVD_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.test_slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msvd_testset)


def dataloader_lsmdc_train(args, tokenizer):
    lsmdc_dataset = LSMDC_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lsmdc_dataset), train_sampler


def dataloader_lsmdc_test(args, tokenizer, subset="test"):
    lsmdc_testset = LSMDC_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.test_slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        lsmdc_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(lsmdc_testset)


def dataloader_activity_train(args, tokenizer):
    activity_dataset = ActivityNet_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(activity_dataset), train_sampler


def dataloader_activity_test(args, tokenizer, subset="test"):
    activity_testset = ActivityNet_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.test_slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        activity_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(activity_testset)


def dataloader_didemo_train(args, tokenizer):
    didemo_dataset = DiDeMo_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(didemo_dataset), train_sampler


def dataloader_didemo_test(args, tokenizer, subset="test"):
    didemo_testset = DiDeMo_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.test_slice_framepos,
    )
    dataloader_didemo = DataLoader(
        didemo_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_didemo, len(didemo_testset)


def dataloader_vlm_train(args, tokenizer):
    vlm_dataset = VLM_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        frame_indices_to_use=args.frame_indices_to_use,
        efficient_subsample=args.efficient_subsample,
        video_max_len=args.video_max_len,
        use_failures_as_negatives_only=args.use_failures_as_negatives_only,
        augment_images=args.augment_images,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        vlm_dataset, seed=args.seed
    )
    dataloader = DataLoader(
        vlm_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(vlm_dataset), train_sampler


def dataloader_vlm_test(args, tokenizer, subset="test"):
    data_path = args.data_path
    features_path = args.features_path
    if subset == "test" and args.test_data_path is not None:
        data_path = args.test_data_path
    if subset == "test" and args.test_features_path is not None:
        features_path = args.test_features_path
    vlm_testset = VLM_DataLoader(
        subset=subset,
        data_path=data_path,
        features_path=features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.test_slice_framepos,
        frame_indices_to_use=args.frame_indices_to_use,
        efficient_subsample=args.efficient_subsample,
        is_test=True,
        video_max_len=args.video_max_len,
        use_failures_as_negatives_only=args.use_failures_as_negatives_only,
    )
    dataloader_vlm = DataLoader(
        vlm_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_vlm, len(vlm_testset)


def dataloader_vlm_prog_train(args, tokenizer):
    if args.augment_images:
        raise NotImplementedError(
            "Random frame sampling not implemented for VLM_ProgressDataLoader."
        )
    vlm_dataset = VLM_ProgressDataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        # frame_order=args.train_frame_order,
        # slice_framepos=args.slice_framepos,
        video_max_len=args.video_max_len,
        use_failures_as_negatives_only=args.use_failures_as_negatives_only,
        augment_images=args.augment_images,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(vlm_dataset)
    dataloader = DataLoader(
        vlm_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(vlm_dataset), train_sampler


def dataloader_vlm_prog_test(args, tokenizer, subset="test"):
    data_path = args.data_path
    features_path = args.features_path
    if subset == "test" and args.test_data_path is not None:
        data_path = args.test_data_path
    if subset == "test" and args.test_features_path is not None:
        features_path = args.test_features_path
    vlm_testset = VLM_ProgressDataLoader(
        subset=subset,
        data_path=data_path,
        features_path=features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        # frame_order=args.eval_frame_order,
        # slice_framepos=args.test_slice_framepos,
        is_test=True,
        video_max_len=args.video_max_len,
        use_failures_as_negatives_only=args.use_failures_as_negatives_only,
    )
    dataloader_vlm = DataLoader(
        vlm_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_vlm, len(vlm_testset)


def dataloader_ssv2_train(args, tokenizer):
    if args.augment_images:
        raise NotImplementedError(
            "Image augmentations not implemented for SSv2_DataLoader."
        )
    vlm_dataset = SSv2_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        # frame_order=args.train_frame_order,
        # slice_framepos=args.slice_framepos,
        video_max_len=args.video_max_len,
        augment_images=args.augment_images,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(vlm_dataset)
    dataloader = DataLoader(
        vlm_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(vlm_dataset), train_sampler


def dataloader_ssv2_test(args, tokenizer, subset="test"):
    # Test set does not include captions: not supported.
    if subset == "test":
        subset = "val"
    vlm_testset = SSv2_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        # frame_order=args.eval_frame_order,
        # slice_framepos=args.test_slice_framepos,
        is_test=True,
        video_max_len=args.video_max_len,
    )
    dataloader_vlm = DataLoader(
        vlm_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_vlm, len(vlm_testset)


class WebDatasetSampler:
    def set_epoch(self, epoch):
        os.environ["WDS_EPOCH"] = str(epoch)


def dataloader_bridge_train(args, tokenizer, subset="train"):
    if args.augment_images:
        raise NotImplementedError(
            "Image augmentations not implemented for WebDatasetLoader."
        )
    dataset = WebDatasetLoader(
        subset=subset,
        data_path=args.data_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        # frame_order=args.train_frame_order,
        # slice_framepos=args.slice_framepos,
        video_max_len=args.video_max_len,
        augment_images=args.augment_images,
    )
    dataset_size = len(dataset)
    batch_size = args.batch_size // args.n_gpu
    dataset = dataset.shuffle(500)
    dataset = dataset.with_length(dataset_size)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_thread_reader,
        batch_size=batch_size,
        drop_last=True,
    )

    # Is not actually used for sampling but only for setting the epoch.
    sampler = WebDatasetSampler()
    return dataloader, dataset_size, sampler


def dataloader_bridge_test(args, tokenizer, subset="test"):
    data_path = args.data_path
    if subset == "test" and args.test_data_path is not None:
        data_path = args.test_data_path
    dataset = WebDatasetLoader(
        subset=subset,
        data_path=data_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        # frame_order=args.eval_frame_order,
        # slice_framepos=args.test_slice_framepos,
        is_test=True,
        video_max_len=args.video_max_len,
    )
    dataset_size = len(dataset)
    batch_size = args.batch_size_val // args.n_gpu
    dataset = dataset.with_length(dataset_size)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_thread_reader,
        batch_size=batch_size,
        drop_last=False,
    )

    return dataloader, dataset_size


def split_tfrecord_datasets(builder_dirs, subset):
    files = []
    for builder_dir in builder_dirs:
        data_files = sorted(glob.glob(os.path.join(builder_dir, "*.tfrecord*")))
        file_count = len(data_files)
        val_count = max(1, int(np.round(0.05 * file_count)))
        test_count = val_count
        train_count = file_count - val_count - test_count
        if subset == "train":
            files.extend(data_files[:train_count])
        elif subset == "val":
            files.extend(data_files[train_count : train_count + val_count])
        else:
            files.extend(data_files[train_count + val_count :])
    return files


def dataloader_openx_train(args, tokenizer):
    datasets = openx.LANG_DATASETS
    version = {d: openx.dataset2version(d) for d in datasets}
    datasets = [os.path.join(f"{d}_{version[d]}", version[d]) for d in datasets]
    builder_dirs = [os.path.join(args.data_path, d) for d in datasets]
    data_files = split_tfrecord_datasets(builder_dirs, "train")
    top_dir = args.data_path.rstrip("/") + "/"
    data_length = sum(
        [openx.DATA_FILE_LENGTHS[f.replace(top_dir, "")] for f in data_files]
    )

    dataset = TFRecord_DataLoader(
        subset="train",
        data_files=data_files,
        data_length=data_length,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        frame_indices_to_use=args.frame_indices_to_use,
        video_max_len=args.video_max_len,
        use_failures_as_negatives_only=args.use_failures_as_negatives_only,
        augment_images=args.augment_images,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=True,
    )

    return dataloader, data_length, None


def dataloader_openx_test(args, tokenizer, subset="test"):
    datasets = openx.LANG_DATASETS
    version = {d: openx.dataset2version(d) for d in datasets}
    datasets = [os.path.join(f"{d}_{version[d]}", version[d]) for d in datasets]
    builder_dirs = [os.path.join(args.data_path, d) for d in datasets]
    data_files = split_tfrecord_datasets(builder_dirs, subset)
    top_dir = args.data_path.rstrip("/") + "/"
    data_length = sum(
        [openx.DATA_FILE_LENGTHS[f.replace(top_dir, "")] for f in data_files]
    )

    dataset = TFRecord_DataLoader(
        subset=subset,
        data_files=data_files,
        data_length=data_length,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.test_slice_framepos,
        frame_indices_to_use=args.frame_indices_to_use,
        is_test=True,
        video_max_len=args.video_max_len,
        use_failures_as_negatives_only=args.use_failures_as_negatives_only,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, data_length


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {
    "train": dataloader_msrvtt_train,
    "val": dataloader_msrvtt_test,
    "test": None,
}
DATALOADER_DICT["msvd"] = {
    "train": dataloader_msvd_train,
    "val": dataloader_msvd_test,
    "test": dataloader_msvd_test,
}
DATALOADER_DICT["lsmdc"] = {
    "train": dataloader_lsmdc_train,
    "val": dataloader_lsmdc_test,
    "test": dataloader_lsmdc_test,
}
DATALOADER_DICT["activity"] = {
    "train": dataloader_activity_train,
    "val": dataloader_activity_test,
    "test": None,
}
DATALOADER_DICT["didemo"] = {
    "train": dataloader_didemo_train,
    "val": dataloader_didemo_test,
    "test": dataloader_didemo_test,
}
DATALOADER_DICT["vlm"] = {
    "train": dataloader_vlm_train,
    "val": dataloader_vlm_test,
    "test": dataloader_vlm_test,
}
DATALOADER_DICT["mw"] = {
    "train": dataloader_vlm_train,
    "val": dataloader_vlm_test,
    "test": dataloader_vlm_test,
}
DATALOADER_DICT["vlm_prog"] = {
    "train": dataloader_vlm_prog_train,
    "val": dataloader_vlm_prog_test,
    "test": dataloader_vlm_prog_test,
}
DATALOADER_DICT["ssv2"] = {
    "train": dataloader_ssv2_train,
    "val": dataloader_ssv2_test,
    "test": None,
}
DATALOADER_DICT["bridge"] = {
    "train": dataloader_bridge_train,
    "val": dataloader_bridge_test,
    "test": dataloader_bridge_test,
}
DATALOADER_DICT["openx"] = {
    "train": dataloader_openx_train,
    "val": dataloader_openx_test,
    "test": dataloader_openx_test,
}
