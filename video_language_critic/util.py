import argparse
import glob
import torch
import torch.nn as nn
import threading

import json
import numpy as np
import random
import os
import pickle
import shutil
from .metrics import (
    compute_metrics,
    compute_symmetric_loss,
    evaluate_auc,
    get_vlm_accuracy,
    tensor_text_to_video_metrics,
    tensor_video_to_text_sim,
)

import time
from torch._utils import ExceptionWrapper
import logging
import wandb
from video_language_critic import ckpt_utils
from .modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from .modules.modeling import CLIP4Clip
from .modules.optimization import BertAdam

from .dataloaders.data_dataloaders import DATALOADER_DICT

SAVE_LOG = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

global logger


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(fct, model, inputs, device_ids):
    modules = nn.parallel.replicate(model, device_ids)
    assert len(modules) == len(inputs)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input):
        torch.set_grad_enabled(grad_enabled)
        device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = fct(module, *input)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device)
                )

    if len(modules) > 1:
        threads = [
            threading.Thread(target=_worker, args=(i, module, input))
            for i, (module, input) in enumerate(zip(modules, inputs))
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


def get_logger(filename=None):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)
    return logger


def get_args(description="CLIP4Clip on Retrieval Task", from_command=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--do_pretrain", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval",
        default=True,
        action="store_true",
        help="Whether to run eval.",
    )
    parser.add_argument(
        "--eval_on_val",
        default=False,
        action="store_true",
        help="If True, run do_eval evaluation on the validation set. Else run on the test set.",
    )

    parser.add_argument("--train_csv", type=str, default="data/msrvtt/train.csv", help="MSRVTT train data CSV")
    parser.add_argument("--val_csv", type=str, default="data/msrvtt/val.csv", help="MSRVTT validation data CSV.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/metaworld/mw50_videos",
        help="Path to data directory containing data splits and captions.",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default="data/metaworld/mw50_videos",
        help="Path to data directory containing videos.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="Path to data directory containing data splits and captions for test data, if different from --data_path.",
    )
    parser.add_argument(
        "--test_features_path",
        type=str,
        default=None,
        help="Path to data directory containing videos, if different from --features_path.",
    )
    parser.add_argument(
        "--evaluate_test_accuracy",
        default=False,
        action="store_true",
        help="If True, evaluate accuracy on test data.",
    )
    parser.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="If True, use only a few batches of data in training and validation for development purposes.",
    )

    parser.add_argument("--num_thread_reader", type=int, default=1, help="")
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="initial learning rate"
    )
    parser.add_argument("--epochs", type=int, default=20, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--batch_size_val", type=int, default=16, help="batch size eval"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.9, help="Learning rate exp epoch decay"
    )
    parser.add_argument(
        "--n_display", type=int, default=100, help="Information display frequence"
    )
    parser.add_argument(
        "--video_dim", type=int, default=1024, help="video feature dimension"
    )
    parser.add_argument(
        "--video_max_len", type=int, default=-1, help="Max length of video"
    )
    parser.add_argument(
        "--deduplicate_captions",
        action="store_true",
        default=False,
        help="Whether to remove duplicate captions in each batch.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_words", type=int, default=32, help="")
    parser.add_argument("--max_frames", type=int, default=12, help="")
    parser.add_argument("--feature_framerate", type=int, default=5, help="")
    parser.add_argument("--margin", type=float, default=0.1, help="margin for loss")
    parser.add_argument(
        "--hard_negative_rate",
        type=float,
        default=0.5,
        help="rate of intra negative sample",
    )
    parser.add_argument(
        "--augment_images",
        action="store_true",
        default=False,
        help="Whether to add image augmentations.",
    )
    parser.add_argument(
        "--add_reversed_negatives",
        action="store_true",
        default=False,
        help="Whether to use the videos played in reverse as negative examples.",
    )
    parser.add_argument(
        "--test_on_reversed_negatives",
        action="store_true",
        default=False,
        help="Whether to use the videos played in reverse as negative examples also at evaluation time.",
    )
    parser.add_argument(
        "--use_failures_as_negatives_only",
        action="store_true",
        default=False,
        help="Whether to only use the videos demonstrating failures as negative examples for other tasks.",
    )
    parser.add_argument(
        "--success_data_only",
        action="store_true",
        default=False,
        help="Whether the dataset includes only successful trajectories. If True, do not compute strict AUC.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy",
        help="Type of loss function to use",
    )
    parser.add_argument(
        "--dist_type",
        type=str,
        default="cosine",
        help="Type of distance function to use between embeddings",
    )
    parser.add_argument(
        "--triplet_margin",
        type=float,
        default=0.2,
        help="Margin to enforce between positive and negative pairs in triplet loss",
    )
    parser.add_argument(
        "--progress_margin",
        type=float,
        default=None,
        help="Margin to enforce between successively longer segments of the video.",
    )
    parser.add_argument(
        "--ranking_loss_weight",
        type=float,
        default=1.0,
        help="Weight of the ranking loss component (if loss_type == sequence_ranking_loss).",
    )
    parser.add_argument(
        "--main_eval_metric",
        type=str,
        default="loss",
        help="Which metric to use for model selection",
    )
    parser.add_argument(
        "--other_eval_metrics",
        type=str,
        default=None,
        help="Other eval metrics for which to keep best checkpoints.",
    )
    parser.add_argument(
        "--n_ckpts_to_keep",
        type=int,
        default=1,
        help="The number of checkpoints to keep in addition to the best epoch. Set to -1 to keep all.",
    )
    parser.add_argument(
        "--keep_ckpt_freq",
        type=int,
        default=0,
        help="How often to keep a checkpoint in addition to the --n_ckpts_to_keep most recent "
        "checkpoints.",
    )
    parser.add_argument(
        "--keep_last_optimizer_only",
        action="store_true",
        default=False,
        help="Whether to only keep the optimizer state of the last checkpoint.",
    )

    parser.add_argument(
        "--negative_weighting",
        type=int,
        default=1,
        help="Weight the loss for intra negative",
    )
    parser.add_argument(
        "--n_pair", type=int, default=1, help="Num of pair to output from data loader"
    )

    parser.add_argument(
        "--output_dir",
        default="experiments",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--wandb_entity",
        default="",
        type=str,
        help="Name of the weights & biases entity to use, if any.",
    )
    parser.add_argument(
        "--wandb_project",
        default="",
        type=str,
        help="Name of the weights & biases project to use, if any.",
    )
    parser.add_argument(
        "--cross_model",
        default="cross-base",
        type=str,
        required=False,
        help="Cross module",
    )
    parser.add_argument(
        "--init_model",
        default=None,
        type=str,
        required=False,
        help="Initial model.",
    )
    parser.add_argument(
        "--resume_model",
        default=None,
        type=str,
        required=False,
        help="Resume train model.",
    )
    parser.add_argument(
        "--resume_from_latest",
        action="store_true",
        default=False,
        help="Whether to resume from the latest checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite experiment in output_dir.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--n_gpu", type=int, default=1, help="Changed in the execute process."
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--datatype", default="vlm", type=str, help="Format of the dataset to train on."
    )
    parser.add_argument(
        "--test_datatype",
        default=None,
        type=str,
        help="Format of the dataset to test on (if different from --datatype)."
    )
    parser.add_argument(
        "--test_set_name",
        default="test",
        type=str,
        help="Test data type added to output path when saving eval results.",
    )

    parser.add_argument("--world_size", default=0, type=int, help="distributed training")
    parser.add_argument("--local_rank", default=0, type=int, help="distributed training")
    parser.add_argument("--rank", default=0, type=int, help="distributed training")
    parser.add_argument(
        "--coef_lr", type=float, default=1.0, help="coefficient for bert branch."
    )

    parser.add_argument(
        "--text_num_hidden_layers", type=int, default=12, help="Layer NO. of text."
    )
    parser.add_argument(
        "--visual_num_hidden_layers", type=int, default=12, help="Layer NO. of visual."
    )
    parser.add_argument(
        "--cross_num_hidden_layers", type=int, default=4, help="Layer NO. of cross."
    )

    parser.add_argument(
        "--loose_type",
        action="store_true",
        help="Default using tight type for retrieval.",
    )
    parser.add_argument("--expand_msrvtt_sentences", action="store_true", help="")

    parser.add_argument(
        "--train_frame_order",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.",
    )
    parser.add_argument(
        "--eval_frame_order",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.",
    )

    parser.add_argument(
        "--freeze_layer_num",
        type=int,
        default=0,
        help="Layer NO. of CLIP need to freeze.",
    )
    parser.add_argument(
        "--slice_framepos",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly; 3: sample randomly from uniform intervals.",
    )
    parser.add_argument(
        "--frame_indices_to_use",
        type=int,
        nargs="+",
        default=None,
        help="If set, use these indices to choose final frames (after slice_framepos is applied)."
    )
    parser.add_argument(
        "--test_slice_framepos",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly; 3: sample randomly from uniform intervals.",
    )
    parser.add_argument(
        "--efficient_subsample",
        action="store_true",
        default=True,
        help="If yes, subsample frames before loading, else subsample after loading. False by default to maintain backwards compatibility.",
    )
    parser.add_argument(
        "--only_keep_final_similarity",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--linear_patch",
        type=str,
        default="2d",
        choices=["2d", "3d"],
        help="linear projection of flattened patches.",
    )
    parser.add_argument(
        "--sim_header",
        type=str,
        default="meanP",
        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
        help="Type of similarity header.",
    )

    parser.add_argument(
        "--pretrained_clip_name",
        default="ViT-B/32",
        type=str,
        help="Choose a CLIP version",
    )
    if from_command is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(from_command)
    if args.frame_indices_to_use is not None:
        print('Raw frame indices to use', args.frame_indices_to_use)
        args.frame_indices_to_use = [int(i) for i in args.frame_indices_to_use]
        print('Frame indices to use', args.frame_indices_to_use)

    args.loose_type = args.sim_header != "tightTransf"
    args.return_sequence = args.loss_type in ["sequence_ranking_loss"]
    if args.test_datatype is None:
        args.test_datatype = args.datatype

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if (
        os.path.exists(args.output_dir)
        and len(glob.glob(os.path.join(args.output_dir, "pytorch_*.bin.*"))) > 0
    ):
        if args.overwrite:
            assert os.path.basename(args.output_dir).startswith("ckpt_")
            print(f"Removing existing directory {args.output_dir}")
            config = load_config(args)
            if "wandb_run_id" in config:
                wandb_run_id = config["wandb_run_id"]
                api = wandb.Api(
                    overrides={
                        "entity": args.wandb_entity,
                        "project": args.wandb_project,
                    }
                )
                run = api.run(f"{args.wandb_project}/{wandb_run_id}")
                run.delete()
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        elif args.do_train and not args.resume_model and not args.resume_from_latest:
            raise ValueError(
                f"Attempting to overwrite existing experiment {args.output_dir}. "
                "Use `--overwrite` if this was intended."
            )
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    update_slurm_job_id(args)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    args.logger = logger
    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    # logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu
            )
        )

    return device, n_gpu


def init_model(args, device, n_gpu, local_rank, ckpt=None, verbose=True):
    init_model = ckpt if ckpt is not None else args.init_model
    if args.resume_from_latest:
        init_model, init_opt = ckpt_utils.get_latest_checkpoint(args.output_dir)
        model_state_dict = None
        if init_model is not None:
            try:
                model_state_dict = torch.load(init_model, map_location="cpu")
            except RuntimeError as e:
                print(init_model, "is not readable:", e)
                # Sometimes the latest file might have been only partially saved.
                init_model, init_opt = ckpt_utils.get_latest_checkpoint(
                    args.output_dir, skip_last=True
                )
                model_state_dict = torch.load(init_model, map_location="cpu")
            args.init_model = init_model
            args.resume_model = init_opt
            print(f"Initializing from\n{init_model}\n{init_opt}")
    elif init_model:
        model_state_dict = torch.load(init_model, map_location="cpu")
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = (
        args.cache_dir
        if args.cache_dir
        else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")
    )
    model = CLIP4Clip.from_pretrained(
        args.cross_model,
        cache_dir=cache_dir,
        state_dict=model_state_dict,
        task_config=args,
        verbose=verbose,
    )

    model.to(device)

    return model


def prep_optimizer(
    args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.0
):
    if hasattr(model, "module"):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    decay_param_tp = [
        (n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)
    ]
    no_decay_param_tp = [
        (n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)
    ]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [
        (n, p) for n, p in no_decay_param_tp if "clip." not in n
    ]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in decay_clip_param_tp],
            "weight_decay": weight_decay,
            "lr": args.lr * coef_lr,
        },
        {"params": [p for n, p in decay_noclip_param_tp], "weight_decay": weight_decay},
        {
            "params": [p for n, p in no_decay_clip_param_tp],
            "weight_decay": 0.0,
            "lr": args.lr * coef_lr,
        },
        {"params": [p for n, p in no_decay_noclip_param_tp], "weight_decay": 0.0},
    ]

    scheduler = None
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.lr,
        warmup=args.warmup_proportion,
        schedule="warmup_cosine",
        b1=0.9,
        b2=0.98,
        e=1e-6,
        t_total=num_train_optimization_steps,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=args.sim_header
        == "tightTransf",  # needed for tightTransformer
    )

    return optimizer, scheduler, model


def deduplicate_captions(captions, caption_dim_data):
    """Remove duplicate captions and return the <unique caption>-<video> mapping as labels."""

    def _equal_captions(c1, c2):
        if isinstance(c1, str):
            return c1 == c2
        else:
            return torch.equal(c1, c2)

    same_caption = torch.zeros(len(captions), len(captions))
    for i in range(len(captions)):
        for j in range(i + 1, len(captions)):
            # torch.any(same_caption[:, j]) if j is a duplicate.
            same_caption[i, j] = _equal_captions(captions[i], captions[j])

    unique_captions = torch.logical_not(torch.any(same_caption, dim=0))
    if isinstance(captions, list):
        captions = list(np.array(captions)[unique_captions])
    else:
        captions = captions[unique_captions]
    for i in range(len(caption_dim_data)):
        caption_dim_data[i] = caption_dim_data[i][unique_captions]

    new_indices = torch.cumsum(unique_captions, dim=0) - 1
    first_matching_captions = new_indices[torch.argmax(same_caption, dim=0)]
    labels = torch.where(unique_captions, new_indices, first_matching_captions)
    labels = nn.functional.one_hot(labels).T

    return (captions, *caption_dim_data, labels)


def train_epoch(
    epoch,
    args,
    model,
    train_dataloader,
    device,
    n_gpu,
    optimizer,
    scheduler,
    global_step,
    local_rank=0,
):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    total_loss_breakdown = {}
    est_len_train_dataloader = len(train_dataloader)
    real_len_train_dataloader = 0

    for step, batch in enumerate(train_dataloader):
        if args.dev and step > 2:
            break
        _, _, _, _, _, video_id, caption = batch
        batch = tuple(t.to(device) for t in batch[:-2])
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask = batch

        if args.deduplicate_captions:
            input_ids, input_mask, segment_ids, labels = deduplicate_captions(
                input_ids, [input_mask, segment_ids]
            )
        else:
            labels = None

        # Use videos without captions as negative examples only.
        if args.use_failures_as_negatives_only:
            is_caption = torch.any(input_mask, dim=-1)
            is_caption = is_caption.squeeze(1)
            if labels is None:
                labels = torch.eye(len(input_ids), len(video)).to(is_caption.device)
            input_ids = input_ids[is_caption]
            input_mask = input_mask[is_caption]
            segment_ids = segment_ids[is_caption]
            labels = labels[is_caption]

        loss, loss_breakdown = model(
            input_ids,
            segment_ids,
            input_mask,
            video,
            video_mask,
            labels=labels,
            captions=caption,
        )

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
            loss_breakdown = {k: v.mean() for k, v in loss_breakdown.items()}
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        real_len_train_dataloader += 1
        for k, v in loss_breakdown.items():
            total_loss_breakdown[k] = total_loss_breakdown.get(k, 0) + float(v)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, "module"):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info(
                    "Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Breakdown: %s, Time/step: %f",
                    epoch,
                    args.epochs,
                    step + 1,
                    est_len_train_dataloader,
                    "-".join(
                        [
                            str("%.9f" % itm)
                            for itm in sorted(list(set(optimizer.get_lr())))
                        ]
                    ),
                    float(loss),
                    str(loss_breakdown),
                    (time.time() - start_time)
                    / (log_step * args.gradient_accumulation_steps),
                )
                start_time = time.time()

    logger.info(
        f"Estimated train dataloader length: {est_len_train_dataloader} vs actual: {real_len_train_dataloader}"
    )
    total_loss = total_loss / real_len_train_dataloader
    total_loss_breakdown = {
        k: v / real_len_train_dataloader for k, v in total_loss_breakdown.items()
    }
    return total_loss, global_step, total_loss_breakdown


def cast_dict_to_float(d):
    float_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = float(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            v = float(v)
        float_dict[k] = v
    return float_dict


def _run_on_single_gpu(
    model,
    batch_list_t,
    batch_list_v,
    batch_sequence_output_list,
    batch_visual_output_list,
    only_keep_final_similarity=False,
):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(
                sequence_output,
                visual_output,
                input_mask,
                video_mask,
                loose_type=model.loose_type,
            )
            if only_keep_final_similarity:
                shift_video_mask = torch.zeros_like(video_mask)
                shift_video_mask[:, :, :-1] = video_mask[:, :, 1:]
                is_last_step = video_mask - shift_video_mask
                is_last_step_repeat = is_last_step.repeat([1, b1b2_logits.shape[0], 1])
                is_last_step_repeat = is_last_step_repeat.permute([1, 0, 2])
                b1b2_logits = b1b2_logits[is_last_step_repeat.bool()].reshape(
                    (b1b2_logits.shape[0], b1b2_logits.shape[1])
                )

            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=1)
        sim_matrix.append(each_row)
    return sim_matrix


def eval_epoch(
    args,
    model,
    test_dataloader,
    datatype,
    device,
    n_gpu,
    save_eval_result=False,
    ckpt_epoch=None,
):
    if hasattr(model, "module"):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if (
        hasattr(test_dataloader.dataset, "multi_sentence_per_video")
        and test_dataloader.dataset.multi_sentence_per_video
    ):
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning(
            "sentence num: {}, video num: {}".format(sentence_num_, video_num_)
        )

    # eval_metric = args.main_eval_metric
    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0
        video_ids = []
        captions = []
        all_labels = []
        video_masks = []
        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            if args.dev and bid >= 3:
                break
            _, _, _, _, _, video_id, caption = batch
            video_ids.append(video_id)
            captions.extend(caption)

            batch = tuple(t.to(device) for t in batch[:-2])
            input_ids, input_mask, segment_ids, video, video_mask = batch

            labels = torch.eye(len(input_ids), len(video)).to(input_ids.device)

            # Use videos without captions as negative examples only.
            if args.use_failures_as_negatives_only:
                is_caption = torch.any(input_mask, dim=-1).to(input_ids.device)
                is_caption = is_caption.squeeze(1)
                input_ids = input_ids[is_caption]
                input_mask = input_mask[is_caption]
                segment_ids = segment_ids[is_caption]
                labels = labels[is_caption]
                # caption = [c for i, c in enumerate(caption) if is_caption[i]]

            # Note: This way of adding reversed videos is a simple but wasteful implementation:
            # 1) Visual output sequences could be reversed after single-frame features have been computed.
            # 2) Not all reversed features need to be compared to all captions.
            if args.test_on_reversed_negatives:
                reversed_video = model.reverse_videos(video, video_mask)
                # Negatives do not match any text.
                labels = torch.cat([labels, torch.zeros_like(labels)], dim=1)
                video = torch.cat([video, reversed_video], dim=0)
                video_mask = torch.cat([video_mask, video_mask], dim=0)

            if datatype in ["vlm_prog"]:
                # Merge progress dimension with the batch dimension.
                video = video.permute(3, 0, 2, 1, 4, 5, 6)
                video = video.flatten(0, 1)
                video = video.unsqueeze(1)
                video_mask = video_mask.permute(3, 0, 1, 2)
                video_mask = video_mask.flatten(0, 1)
            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output = model.get_sequence_output(
                    input_ids, segment_ids, input_mask
                )
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append(
                    (
                        input_mask,
                        segment_ids,
                    )
                )

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [
                    itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_
                ]

                if len(filter_inds) > 0:
                    video, video_mask = (
                        video[filter_inds, ...],
                        video_mask[filter_inds, ...],
                    )
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, visual_output = model.get_sequence_visual_output(
                    input_ids, segment_ids, input_mask, video, video_mask
                )

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append(
                    (
                        input_mask,
                        segment_ids,
                    )
                )

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            video_masks.append(video_mask.cpu().numpy().squeeze(1))
            all_labels.append(labels)
            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device("cuda:{}".format(str(dev_id)))
                    devc_batch_list = [
                        tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]
                    ]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [
                        tuple(t.to(devc) for t in b) for b in batch_list_v
                    ]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [
                        b.to(devc) for b in batch_sequence_output_list[s_:e_]
                    ]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [
                (
                    batch_list_t_splits[dev_id],
                    batch_list_v_splits[dev_id],
                    batch_t_output_splits[dev_id],
                    batch_v_output_splits[dev_id],
                )
                for dev_id in device_ids
            ]
            parallel_outputs = parallel_apply(
                _run_on_single_gpu, model, parameters_tuple_list, device_ids
            )
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            video_masks = np.concatenate(video_masks, axis=0)
        else:
            sim_matrix = _run_on_single_gpu(
                model,
                batch_list_t,
                batch_list_v,
                batch_sequence_output_list,
                batch_visual_output_list,
                only_keep_final_similarity=args.only_keep_final_similarity,
            )
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            video_masks = np.concatenate(video_masks, axis=0)

    if multi_sentence_:
        logger.info(
            "before reshape, sim matrix size: {} x {}".format(
                sim_matrix.shape[0], sim_matrix.shape[1]
            )
        )
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max(
            [
                e_ - s_
                for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)
            ]
        )
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(
                np.concatenate(
                    (
                        sim_matrix[s_:e_],
                        np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf),
                    ),
                    axis=0,
                )
            )
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info(
            "after reshape, sim matrix size: {} x {} x {}".format(
                sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]
            )
        )

        # tv_metrics = {}
        # if eval_metric != "loss":
        if datatype in ["vlm_prog"]:
            # For a non-square sim matrix, take its last square part.
            assert (
                not args.failures_as_negatives_only
            ), "Failures as negatives only not implemented for progress data."
            # TODO: if both are on, we need to divide dim. 0 by n_progress_steps.
            last_sim_matrix = sim_matrix[:, -sim_matrix.shape[0] :]
        else:
            last_sim_matrix = sim_matrix
        tv_metrics = tensor_text_to_video_metrics(last_sim_matrix)
        if not isinstance(captions[0], str):
            captions = [item for caption_batch in captions for item in caption_batch]
        labels = np.zeros((sim_matrix.shape[0], sim_matrix.shape[1]))
        assert np.sum([b.shape[0] for b in all_labels]) == sim_matrix.shape[0]
        assert np.sum([b.shape[1] for b in all_labels]) == sim_matrix.shape[1]
        acc_y, acc_x = 0
        for batch_labels in all_labels:
            labels[
                acc_y : acc_y + batch_labels.shape[0],
                acc_x : acc_x + batch_labels.shape[1],
            ] = batch_labels
            acc_y += batch_labels.shape[0]
            acc_x += batch_labels.shape[1]

        tv_metrics = evaluate_auc(
            args, last_sim_matrix, captions, labels, datatype, tv_metrics
        )

        if args.deduplicate_captions:
            raise NotImplementedError(
                "Caption deduplication not implemented for multi_sentence."
            )
            video_has_label = np.any(labels, axis=0)
            # Corresponds to the first dimension of sim_matrix.
            sim_matrix_captions = [
                caption for v, caption in enumerate(captions) if video_has_label[v]
            ]
            captions, sim_matrix, labels = deduplicate_captions(captions, [sim_matrix])
        tv_metrics["loss"], val_loss_breakdown = compute_symmetric_loss(
            args, model, sim_matrix, labels, captions, device, video_masks
        )
        tv_metrics.update(val_loss_breakdown)
        # vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info(
            "sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1])
        )

        if datatype in ["vlm_prog"]:
            # For a non-square sim matrix, take its last square part.
            assert (
                not args.failures_as_negatives_only
            ), "Failures as negatives only not implemented for progress data."
            # TODO: if both are on, we need to divide dim. 0 by n_progress_steps.
            last_sim_matrix = sim_matrix[:, -sim_matrix.shape[0] :]
        elif model.return_sequence and len(sim_matrix.shape) > 2:
            # TODO: Use the video mask to get the final step.
            last_sim_matrix = sim_matrix[:, :, -1]
        else:
            last_sim_matrix = sim_matrix

        last_sim_matrix_auc = last_sim_matrix
        if args.test_on_reversed_negatives:
            # Only the loss computation and retrieval metrics use reversed examples,
            # auc does not.
            n_vs = sim_matrix.shape[1]
            assert n_vs % 2 == 0
            last_sim_matrix_auc = sim_matrix[:, : n_vs // 2]

        labels = np.zeros((sim_matrix.shape[0], sim_matrix.shape[1]))
        assert np.sum([b.shape[0] for b in all_labels]) == sim_matrix.shape[0]
        assert np.sum([b.shape[1] for b in all_labels]) == sim_matrix.shape[1]
        acc_y, acc_x = 0, 0
        for batch_labels in all_labels:
            batch_labels = batch_labels.cpu().detach().numpy()
            labels[
                acc_y : acc_y + batch_labels.shape[0],
                acc_x : acc_x + batch_labels.shape[1],
            ] = batch_labels
            acc_y += batch_labels.shape[0]
            acc_x += batch_labels.shape[1]

        # tv_metrics = {}
        # if eval_metric == "loss":
        if SAVE_LOG:
            tv_metrics = compute_metrics(last_sim_matrix, video_ids, captions)
        else:
            tv_metrics = compute_metrics(last_sim_matrix, labels=labels)
        metrics = {f"tv_{k}": v for k, v in tv_metrics.items()}
        if not isinstance(captions[0], str):
            captions = [item for caption_batch in captions for item in caption_batch]

        auc_metrics = evaluate_auc(
            args, last_sim_matrix_auc, captions, labels, datatype
        )
        metrics.update(auc_metrics)

        if args.deduplicate_captions:
            captions, sim_matrix, labels = deduplicate_captions(captions, [sim_matrix])
        metrics["loss"] = 0
        val_loss_breakdown = {}
        if not args.only_keep_final_similarity:
            metrics["loss"], val_loss_breakdown = compute_symmetric_loss(
                args, model, sim_matrix, labels, captions, device, video_masks
            )
        metrics.update(val_loss_breakdown)
        vt_metrics = compute_metrics(last_sim_matrix.T, labels=labels.T)
        metrics.update({f"vt_{k}": v for k, v in vt_metrics.items()})
        logger.info(
            "\t Length-T: {}, Length-V:{}".format(len(sim_matrix), len(sim_matrix[0]))
        )

    logger.info("Text-to-Video:")
    if "R1" in tv_metrics:
        logger.info(
            "\t>>>  Loss: {:.4f} - AUC {:.4f} - Labeled AUC {:.4f} - Strict AUC {:.4f} - R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".format(
                metrics["loss"],
                metrics["auc"],
                metrics["labeled_auc"],
                metrics["strict_auc"],
                tv_metrics["R1"],
                tv_metrics["R5"],
                tv_metrics["R10"],
                tv_metrics["MedianR"],
                tv_metrics["MeanR"],
            )
        )
        logger.info(val_loss_breakdown)
    else:
        logger.info("\t>>>  Loss: {:.4f}".format(metrics["loss"]))
    logger.info("========================================")
    logger.info("Video-to-Text:")
    logger.info(
        "\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}".format(
            vt_metrics["R1"],
            vt_metrics["R5"],
            vt_metrics["R10"],
            vt_metrics["MedianR"],
            vt_metrics["MeanR"],
        )
    )

    # metric = tv_metrics[main_metric]
    # return metric
    if save_eval_result:
        if ckpt_epoch is None:
            ckpt_path = args.init_model
            ckpt_epoch = int(os.path.basename(ckpt_path).split(".")[-1])
        split = "val" if args.eval_on_val else "test"

        out_path = os.path.join(
            args.output_dir,
            args.test_set_name,
            "eval",
            f"{split}_model_{ckpt_epoch}.pkl",
        )
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        metrics_to_save = cast_dict_to_float(metrics)
        result = {
            "sim_matrix": sim_matrix,
            "video_ids": video_ids,
            "captions": captions,
            "metrics": metrics_to_save,
            "args": args,
        }
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"Saved eval to {out_path}")
        if args.test_datatype == "vlm":
            metrics["vlm_correct_pick_accuracy"] = get_vlm_accuracy(
                out_path,
                condition=lambda x: "step_id_1" not in x,
                correct_condition=lambda x: "correct_pick" in x,
                verbose=False,
            )

        metrics_to_save = cast_dict_to_float(metrics)
        eval_metrics_path = out_path.replace(".pkl", "_metrics.json")
        with open(eval_metrics_path, "w") as f:
            json.dump(metrics_to_save, f)
        logger.info(f"Saved eval metrics to {eval_metrics_path}")

    return metrics


def freeze_model(model, args):
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if (
                name.find("ln_final.") == 0
                or name.find("text_projection") == 0
                or name.find("logit_scale") == 0
                or name.find("visual.ln_post.") == 0
                or name.find("visual.proj") == 0
            ):
                continue  # need to train
            elif (
                name.find("visual.transformer.resblocks.") == 0
                or name.find("transformer.resblocks.") == 0
            ):
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue  # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False


def get_val_test_dataloaders(args, tokenizer):
    assert args.datatype in DATALOADER_DICT
    assert args.test_datatype in DATALOADER_DICT

    assert (
        DATALOADER_DICT[args.test_datatype]["test"] is not None
        or DATALOADER_DICT[args.datatype]["val"] is not None
    )

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.test_datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.test_datatype]["test"](
            args, tokenizer
        )

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](
            args, tokenizer, subset="val"
        )
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length
    return test_dataloader, test_length, val_dataloader, val_length


def load_config(args):
    config_path = os.path.join(args.output_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    return config


def update_wandb_run_id(args, run_id):
    config = load_config(args)
    config["wandb_run_id"] = run_id
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f)


def update_slurm_job_id(args):
    config = load_config(args)
    slurm_id = None
    if "SLURM_JOB_ID" in os.environ:
        slurm_id = os.environ["SLURM_JOB_ID"]
    if "SLURM_ARRAY_TASK_ID" in os.environ and os.environ["SLURM_ARRAY_TASK_ID"]:
        slurm_id += "_" + os.environ["SLURM_ARRAY_TASK_ID"]
    if "slurm_job_id" not in config:
        config["slurm_job_id"] = []
    config["slurm_job_id"].append(slurm_id)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f)


def init_wandb_logger(args):
    wandb_logger = None
    if args.wandb_project:
        run_id = None
        if args.resume_model:
            config = load_config(args)
            run_id = config["wandb_run_id"]
        run_name = os.path.basename(args.output_dir)
        if run_name[:5] == "ckpt_":
            run_name = run_name[5:]
        run_name = os.path.join(
            os.path.basename(os.path.dirname(args.output_dir)), run_name
        )
        wandb_logger = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=args,
            resume="must" if args.resume_model is not None else None,
            id=run_id,
        )
        if run_id is None:
            update_wandb_run_id(args, wandb_logger.id)
    return wandb_logger


def do_train(
    args,
    tokenizer,
    model,
    device,
    n_gpu,
    val_dataloader,
    test_dataloader,
):
    train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype][
        "train"
    ](args, tokenizer)
    num_train_optimization_steps = (
        int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
        / args.gradient_accumulation_steps
    ) * args.epochs

    coef_lr = args.coef_lr
    optimizer, scheduler, model = prep_optimizer(
        args,
        model,
        num_train_optimization_steps,
        device,
        n_gpu,
        args.local_rank,
        coef_lr=coef_lr,
    )

    if args.local_rank == 0:
        if train_length is not None:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info(
                "  Num steps = %d",
                num_train_optimization_steps * args.gradient_accumulation_steps,
            )

    main_eval_metric = args.main_eval_metric
    eval_metrics = [main_eval_metric] + [
        metric.strip() for metric in args.other_eval_metrics.split(",")
    ]
    best_scores = {
        metric: np.inf if ckpt_utils.lower_is_better(metric) else -np.inf
        for metric in eval_metrics
    }
    best_model_files = {metric: None for metric in eval_metrics}
    ## ##############################################################
    # resume optimizer state besides loss to continue train
    ## ##############################################################
    resumed_epoch = 0
    if args.resume_model:
        checkpoint = torch.load(args.resume_model, map_location="cpu")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_scores = checkpoint["best_scores"]
        best_model_files = checkpoint["best_model_files"]
        resumed_epoch = checkpoint["epoch"]
        logger.info(f"Resumed from epoch {resumed_epoch}:")
        for k, v in best_model_files.items():
            logger.info(f"{k.ljust(20)} {best_scores[k]:.8f} {os.path.basename(v)}")
    # if args.datatype == "ssv2":
    #     logger.info("Loading val data into RAM...")
    #     val_data = []
    #     for b, batch in enumerate(val_dataloader):
    #         if b % 500 == 0 or b == 1:
    #             logger.info(f"Loaded {b}/{len(val_dataloader)} batches")
    #         val_data.append(batch)
    #     logger.info("Done loading val data into RAM.")
    # else:
    wandb_logger = init_wandb_logger(args)
    val_data = val_dataloader

    global_step = 0
    for epoch in range(resumed_epoch + 1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr_loss, global_step, tr_loss_breakdown = train_epoch(
            epoch,
            args,
            model,
            train_dataloader,
            device,
            n_gpu,
            optimizer,
            scheduler,
            global_step,
            local_rank=args.local_rank,
        )
        if args.local_rank == 0:
            logger.info(
                "Epoch %d/%s Finished, Train Loss: %f, Breakdown %s",
                epoch,
                args.epochs,
                tr_loss,
                str(tr_loss_breakdown),
            )

        val_metrics = eval_epoch(args, model, val_data, args.datatype, device, n_gpu)
        test_metrics = eval_epoch(
            args,
            model,
            test_dataloader,
            args.test_datatype,
            device,
            n_gpu,
            save_eval_result=True,
            ckpt_epoch=epoch,
        )
        metrics = {
            **{f"train_{k}": v for k, v in tr_loss_breakdown.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }

        if args.local_rank == 0:
            if wandb_logger is not None:
                wandb_logger.log(metrics, step=epoch, commit=True)
            (
                best_scores,
                best_model_files,
            ) = ckpt_utils.save_and_keep_best_checkpoints(
                args,
                val_metrics,
                epoch,
                best_scores,
                best_model_files,
                model,
                optimizer,
                tr_loss,
                logger,
            )
            logger.info(f"Epoch {epoch}:")
            for k, v in best_model_files.items():
                logger.info(f"{k.ljust(20)} {best_scores[k]:.8f} {os.path.basename(v)}")
            logger.info(
                "The best model is: {}, the {} is: {:.4f}".format(
                    best_model_files[main_eval_metric],
                    main_eval_metric,
                    best_scores[main_eval_metric],
                )
            )

    return best_model_files[main_eval_metric]
    ## Uncomment if want to test on the best checkpoint
    # if args.local_rank == 0:
    #     model = ckpt_utils.load_model(-1, args, n_gpu, device, model_file=best_output_model_file, logger=logger)
    #     eval_epoch(args, model, test_dataloader, device, n_gpu)
