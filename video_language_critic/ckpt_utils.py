import glob
import os
import torch
from .modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from .modules.modeling import CLIP4Clip


def get_model_path(args, epoch, type_name=""):
    return os.path.join(
        args.output_dir,
        "pytorch_model.bin.{}{}".format(
            "" if type_name == "" else type_name + ".", epoch
        ),
    )


def get_optimizer_path(args, epoch, type_name=""):
    return os.path.join(
        args.output_dir,
        "pytorch_opt.bin.{}{}".format(
            "" if type_name == "" else type_name + ".", epoch
        ),
    )


def get_latest_checkpoint(output_dir, skip_last=False):
    """Get the latest model and optimizer checkpoints from output_dir."""
    ckpts = glob.glob(os.path.join(output_dir, "pytorch_model.bin.*"))
    ckpts = [f for f in ckpts if os.path.isfile(f) and not f.endswith(".pkl")]
    epoch_to_ckpt = {int(c.split(".")[-1]): c for c in ckpts}
    latest_ckpts = sorted(epoch_to_ckpt.items(), reverse=True)
    print("Latest checkpoints:", latest_ckpts)
    if skip_last:
        latest_ckpts = latest_ckpts[1:]
    if len(latest_ckpts) == 0:
        print(f"No checkpoints found in {output_dir}")
        return None, None
    ckpt = latest_ckpts[0][1]
    opt_ckpt = ckpt.replace("pytorch_model.bin", "pytorch_opt.bin")
    return ckpt, opt_ckpt


def save_checkpoint(
    model_to_save,
    optimizer,
    epoch,
    tr_loss,
    best_scores,
    best_model_files,
    output_model_file,
    optimizer_state_file,
    logger=None,
):
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": tr_loss,
            "best_scores": best_scores,
            "best_model_files": best_model_files,
        },
        optimizer_state_file,
    )
    if logger:
        logger.info("Model saved to %s", output_model_file)
        logger.info("Optimizer saved to %s", optimizer_state_file)


def remove_checkpoint(output_model_file):
    if os.path.exists(output_model_file):
        os.remove(output_model_file)
    optimizer_path = output_model_file.replace("pytorch_model", "pytorch_opt")
    if os.path.exists(optimizer_path):
        os.remove(optimizer_path)


def save_model(
    epoch,
    args,
    model,
    optimizer,
    tr_loss,
    best_scores,
    best_model_files,
    type_name="",
    logger=None,
):
    """Save the model.

    Keep the args.n_ckpts_to_keep most recent checkpoints and remove older ones, unless they are in best_model_files.
    """
    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = get_model_path(args, epoch, type_name)
    optimizer_state_file = get_optimizer_path(args, epoch, type_name)

    save_checkpoint(
        model_to_save,
        optimizer,
        epoch,
        tr_loss,
        best_scores,
        best_model_files,
        output_model_file,
        optimizer_state_file,
        logger,
    )

    prev_model_file = get_model_path(args, epoch - args.n_ckpts_to_keep, type_name)
    prev_model_epoch = epoch - args.n_ckpts_to_keep
    prev_optimizer_file = get_optimizer_path(
        args, epoch - args.n_ckpts_to_keep, type_name
    )
    if os.path.exists(prev_model_file):
        if (
            prev_model_file in best_model_files.values()
            or prev_model_epoch % args.keep_ckpt_freq == 0
        ):
            print(f"Not removing checkpoint {os.path.basename(prev_model_file)}")
        else:
            os.remove(prev_model_file)
            if os.path.exists(prev_optimizer_file):
                os.remove(prev_optimizer_file)
            print(f"Removed checkpoint {os.path.basename(prev_model_file)}")
    if args.keep_last_optimizer_only:
        prev_optimizer_file = get_optimizer_path(args, epoch - 1, type_name)
        if os.path.exists(prev_optimizer_file):
            os.remove(prev_optimizer_file)
            print(f"Removed optimizer {os.path.basename(prev_optimizer_file)}")

    return output_model_file


def save_and_keep_best_checkpoints(
    args,
    current_metrics,
    epoch,
    best_scores,
    best_model_files,
    model,
    optimizer,
    tr_loss,
    logger=None,
):
    (
        best_scores,
        best_model_files,
        ckpts_to_remove,
    ) = update_best_scores(
        args,
        current_metrics,
        epoch,
        best_scores,
        best_model_files,
    )
    # Save model, optimizer and best scores & files.
    save_model(
        epoch,
        args,
        model,
        optimizer,
        tr_loss,
        best_scores,
        best_model_files,
        type_name="",
        logger=logger,
    )

    for ckpt in ckpts_to_remove:
        logger.info(f"Removing {ckpt}, no longer the best checkpoint for any metric")
        remove_checkpoint(ckpt)

    return best_scores, best_model_files


def lower_is_better(eval_metric):
    ms = eval_metric.split("_")
    return "loss" in ms or "MR" in ms or "MedianR" in ms or "MeanR" in ms


def update_best_scores(args, current_metrics, epoch, best_scores, best_model_files):
    """Keep track of the best scoring checkpoint per metric and return a list of older checkpoints."""
    output_model_file = get_model_path(args, epoch)
    ckpts_to_remove = set()
    ckpts_to_keep = set()
    for metric, best_file in best_model_files.items():
        best_score = best_scores[metric]
        current_score = current_metrics[metric]
        matches_keep_freq = False
        if best_file is not None:
            prev_best_epoch = int(best_file.split(".")[-1])
            if args.keep_ckpt_freq > 0:
                matches_keep_freq = prev_best_epoch % args.keep_ckpt_freq == 0

        # Current score is better than the existing best score.
        if (lower_is_better(metric) and best_score > current_score) or (
            not lower_is_better(metric) and best_score < current_score
        ):
            if best_file is not None:
                if (
                    args.n_ckpts_to_keep >= 0
                    and prev_best_epoch <= epoch - args.n_ckpts_to_keep
                    and not matches_keep_freq
                ):
                    ckpts_to_remove.add(best_file)
            best_scores[metric] = current_score
            best_model_files[metric] = output_model_file
        else:
            ckpts_to_keep.add(best_file)
    ckpts_to_remove = ckpts_to_remove - ckpts_to_keep

    return best_scores, best_model_files, ckpts_to_remove


def load_model(epoch, args, n_gpu, device, model_file=None, logger=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location="cpu")
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
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
        )

        model.to(device)
    else:
        model = None
    return model
