from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from transformers import AutoProcessor
from video_language_critic.util import (
    get_args,
    set_seed_logger,
    init_device,
    freeze_model,
    init_siglip_model,
    get_val_test_dataloaders,
    do_train,
    eval_epoch,
)

torch.distributed.init_process_group(backend="nccl")

global logger


def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    logger = args.logger
    device, n_gpu = init_device(args, args.local_rank)

    # Use SigLIP processor instead of CLIP tokenizer
    pretrained_siglip_name = "google/siglip-base-patch16-224"
    if hasattr(args, "pretrained_siglip_name"):
        pretrained_siglip_name = args.pretrained_siglip_name
    
    processor = AutoProcessor.from_pretrained(pretrained_siglip_name)
    tokenizer = processor.tokenizer  # Extract tokenizer for compatibility

    model = init_siglip_model(args, device, n_gpu, args.local_rank)

    freeze_model(model, args)

    test_dataloader, test_length, val_dataloader, val_length = get_val_test_dataloaders(
        args, tokenizer
    )

    if args.local_rank == 0:
        if test_length is not None:
            logger.info("***** Running test *****")
            logger.info("  Num examples = %d", test_length)
            logger.info("  Batch size = %d", args.batch_size_val)
            logger.info("  Num steps = %d", len(test_dataloader))
        if val_length is not None:
            logger.info("***** Running val *****")
            logger.info("  Num examples = %d", val_length)

    if args.do_train:
        best_ckpt = do_train(
            args,
            tokenizer,
            model,
            device,
            n_gpu,
            val_dataloader,
            test_dataloader,
        )
    elif args.do_eval:
        if args.local_rank == 0:
            dataloader = val_dataloader if args.eval_on_val else test_dataloader
            eval_epoch(args, model, dataloader, device, n_gpu, save_eval_result=True)


if __name__ == "__main__":
    main()