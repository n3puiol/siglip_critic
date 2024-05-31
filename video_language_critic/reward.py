import torch
from .util import init_device, init_model, get_args
from video_language_critic.modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from .dataloaders.data_dataloaders import DATALOADER_DICT
from video_language_critic.dataloaders.dataloader_vlm_retrieval import VLM_DataLoader


class RewardCalculator:
    def __init__(self, args=None, val_stats={"mean": 0, "std": 1}):
        if args is None:
            print("No args specified. Using default.")
            self.args = get_args()
        else:
            args = self.set_missing_args_to_default_values(args)
            self.args = args
        self.val_stats = val_stats
        self.tokenizer = ClipTokenizer()
        self.dataloader = self.get_dataloader()
        self.model, self.device = self.get_model_device()

    def set_missing_args_to_default_values(self, args):
        try:
            frame_indices_to_use = args.frame_indices_to_use
        except AttributeError:
            args.frame_indices_to_use = None
        return args

    def get_dataloader(self):
        args = self.args
        print(self.args)
        dataloader = VLM_DataLoader(
            subset="test",
            data_path=args.data_path,
            features_path=args.features_path,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            tokenizer=self.tokenizer,
            max_frames=args.max_frames,
            frame_order=args.eval_frame_order,
            slice_framepos=args.test_slice_framepos,
            frame_indices_to_use=args.frame_indices_to_use,
            video_max_len=args.video_max_len,
            use_failures_as_negatives_only=args.use_failures_as_negatives_only,
            augment_images=False,
            load_data=False,
        )

        return dataloader

    def get_model_device(self, args=None):
        if args is None:
            args = self.args

        device, n_gpu = init_device(args, args.local_rank)

        model = init_model(args, device, n_gpu, args.local_rank)
        model.eval()  # if not in eval mode, will be torch.distributed

        return model, device

    def get_reward(self, video, instruction):
        batch = self.dataloader.get_encoded_data(video, instruction)
        batch = tuple(torch.tensor(t) for t in batch)
        batch = tuple(t.to(self.device) for t in batch)

        (
            input_ids,
            input_mask,
            segment_ids,
            video,
            video_mask,
        ) = batch
        if len(video.shape) == 6:
            video = video.squeeze()
        """input_ids.shape, segment_ids.shape, input_mask.shape, video.shape, video_mask.shape

        >> (torch.Size([16, 1, 32]), torch.Size([16, 1, 32]), torch.Size([16, 1, 32]), torch.Size([16, 1, 12, 1, 3, 224, 224]), torch.Size([16, 1, 12]))"""

        print(
            input_ids.shape,
            segment_ids.shape,
            input_mask.shape,
            video.shape,
            video_mask.shape,
        )
        sequence_output, visual_output = self.model.get_sequence_visual_output(
            input_ids, segment_ids, input_mask, video, video_mask, shaped=True
        )
        reward, *_tmp = self.model.get_similarity_logits(
            sequence_output,
            visual_output,
            input_mask,
            video_mask,
            loose_type=self.model.loose_type,
        )

        reward = (
            reward.cpu().detach().numpy()[0][0] - self.val_stats["mean"]
        ) / self.val_stats["std"]
        return reward
