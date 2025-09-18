from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
from torch import nn
from transformers import SiglipModel, SiglipProcessor

from .until_module import (
    PreTrainedModel,
    AllGather,
    BinaryCrossEn,
    CrossEn,
    SequentialRankingLoss,
    SemiHardNegativeTripletLoss,
)
from .module_cross import CrossModel, CrossConfig, Transformer as TransformerClip
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)
allgather = AllGather.apply


class SigLIP4SigLIPPreTrainedModel(PreTrainedModel, nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    def __init__(self, cross_config, *inputs, **kwargs):
        super(SigLIP4SigLIPPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.siglip = None
        self.cross = None

    @classmethod
    def from_pretrained(
        cls,
        cross_model_name,
        state_dict=None,
        cache_dir=None,
        type_vocab_size=2,
        *inputs,
        **kwargs,
    ):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None:
            state_dict = {}

        # Load SigLIP model
        pretrained_siglip_name = "google/siglip-base-patch16-224"
        if hasattr(task_config, "pretrained_siglip_name"):
            pretrained_siglip_name = task_config.pretrained_siglip_name
        
        siglip_model = SiglipModel.from_pretrained(pretrained_siglip_name)
        
        # Extract SigLIP state dict and add siglip prefix
        siglip_state_dict = siglip_model.state_dict()
        for key, val in siglip_state_dict.items():
            new_key = "siglip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(
            cross_model_name,
            cache_dir,
            type_vocab_size,
            state_dict=None,
            task_config=task_config,
        )

        model = cls(cross_config, siglip_state_dict, *inputs, **kwargs)

        # Initialization tricks similar to CLIP4Clip
        if model.sim_header == "tightTransf":
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                # Initialize cross-attention from SigLIP text encoder
                for key, val in siglip_state_dict.items():
                    if key == "text_model.embeddings.position_embedding.weight":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.startswith("text_model.encoder.layers"):
                        layer_num = int(key.split(".")[3])
                        if layer_num < task_config.cross_num_hidden_layers:
                            new_key = key.replace("text_model.encoder.layers", "cross.transformer.resblocks")
                            state_dict[new_key] = val.clone()

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in siglip_state_dict.items():
                    if key == "text_model.embeddings.position_embedding.weight":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if (
                        model.sim_header == "seqTransf"
                        and key.startswith("text_model.encoder.layers")
                    ):
                        layer_num = int(key.split(".")[3])
                        if layer_num < task_config.cross_num_hidden_layers:
                            new_key = key.replace("text_model.encoder.layers", "transformerClip.layers")
                            state_dict[new_key] = val.clone()

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(
    target_name,
    target_config,
    target_attr_name,
    source_config,
    source_attr_name,
    default_value=None,
):
    if hasattr(source_config, source_attr_name):
        if (
            default_value is None
            or getattr(source_config, source_attr_name) != default_value
        ):
            setattr(
                target_config,
                target_attr_name,
                getattr(source_config, source_attr_name),
            )
            show_log(
                source_config,
                "Set {}.{}: {}.".format(
                    target_name,
                    target_attr_name,
                    getattr(target_config, target_attr_name),
                ),
            )
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class SigLIP4SigLIP(SigLIP4SigLIPPreTrainedModel):
    def __init__(self, cross_config, siglip_state_dict, task_config, verbose=True):
        super(SigLIP4SigLIP, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1
        
        # Get SigLIP model configuration
        pretrained_siglip_name = "google/siglip-base-patch16-224"
        if hasattr(task_config, "pretrained_siglip_name"):
            pretrained_siglip_name = task_config.pretrained_siglip_name
        
        # Initialize SigLIP model
        self.siglip = SiglipModel.from_pretrained(pretrained_siglip_name).float()
        
        # Extract model dimensions from SigLIP config
        siglip_config = self.siglip.config
        embed_dim = siglip_config.text_config.hidden_size
        vision_config = siglip_config.vision_config
        text_config = siglip_config.text_config
        
        # Set max position embeddings
        max_position_embeddings = text_config.max_position_embeddings
        assert (
            self.task_config.max_words + self.task_config.max_frames
            <= max_position_embeddings
        )

        self._stage_one = True
        self._stage_two = False

        if verbose:
            show_log(
                task_config,
                "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two),
            )

        self.loose_type = False
        if self._stage_one and check_attr("loose_type", self.task_config):
            self.loose_type = True
            if verbose:
                show_log(task_config, "Test retrieval by loose type.")

        self.return_sequence = check_attr("return_sequence", self.task_config)

        if verbose:
            show_log(task_config, "\t embed_dim: {}".format(embed_dim))
            show_log(task_config, "\t image_size: {}".format(vision_config.image_size))
            show_log(task_config, "\t patch_size: {}".format(vision_config.patch_size))
            show_log(task_config, "\t vision_hidden_size: {}".format(vision_config.hidden_size))
            show_log(task_config, "\t text_hidden_size: {}".format(text_config.hidden_size))
            show_log(task_config, "\t vocab_size: {}".format(text_config.vocab_size))
            show_log(task_config, "\t max_position_embeddings: {}".format(max_position_embeddings))

        # Similarity header configuration
        self.sim_header = "meanP"
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf":
            assert self.loose_type is False

        cross_config.max_position_embeddings = max_position_embeddings
        if self.loose_type is False:
            # Cross Encoder
            cross_config = update_attr(
                "cross_config",
                cross_config,
                "num_hidden_layers",
                self.task_config,
                "cross_num_hidden_layers",
            )
            cross_config.return_sequence = self.return_sequence
            self.cross = CrossModel(cross_config)
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim
            )
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(
                width=embed_dim,
                layers=self.task_config.cross_num_hidden_layers,
                heads=text_config.num_attention_heads,
            )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(
                input_size=embed_dim,
                hidden_size=embed_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
            )

        # Loss configuration
        self.loss_type = self.task_config.loss_type
        self.dist_type = self.task_config.dist_type
        self.add_reversed_negatives = self.task_config.add_reversed_negatives
        self.retain_ts = self.task_config.datatype == "vlm_prog"
        
        if self.loss_type == "semihard_triplet":
            margin = self.task_config.triplet_margin
            progress_margin = self.task_config.progress_margin
            self.loss_fct = SemiHardNegativeTripletLoss(
                margin=margin, progress_margin=progress_margin
            )
        elif self.loss_type == "binary_cross_entropy":
            self.loss_fct = BinaryCrossEn()
        elif self.loss_type == "sequence_ranking_loss":
            self.loss_fct = CrossEn()
            self.ranking_loss_fct = SequentialRankingLoss()
            self.ranking_loss_weight = self.task_config.ranking_loss_weight
        else:
            self.loss_fct = CrossEn()

        self.apply(self.init_weights)

    def reverse_videos(self, video, video_mask):
        reversed_video = torch.zeros_like(video)
        for i in range(video.shape[0]):
            reversed_video[i, video_mask[i].bool()] = torch.flip(
                video[i, video_mask[i].bool()], [0]
            )
        return reversed_video

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        video,
        video_mask=None,
        return_loss=None,
        labels=None,
        captions=None,
    ):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        if self.retain_ts:
            # Add progress steps to the batch dimension.
            video = video.permute(3, 0, 2, 1, 4, 5, 6)
            video = video.flatten(0, 1)
            video = video.unsqueeze(1)
            video_mask = video_mask.permute(3, 0, 2, 1)
            video_mask = video_mask.flatten(0, 1)
            video_mask = video_mask.squeeze(-1)
        else:
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        # T x 3 x H x W
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, visual_output = self.get_sequence_visual_output(
            input_ids,
            attention_mask,
            video,
            video_mask,
            shaped=True,
            video_frame=video_frame,
        )
        
        if self.add_reversed_negatives:
            reversed_visual_output = self.reverse_videos(visual_output, video_mask)
            if labels is None:
                assert sequence_output.shape[0] == visual_output.shape[0]
                labels = torch.eye(visual_output.shape[0]).to(visual_output.device)
            # Negatives do not match any text.
            labels = torch.cat([labels, torch.zeros_like(labels)], dim=1)
            visual_output = torch.cat([visual_output, reversed_visual_output], dim=0)
            video_mask = torch.cat([video_mask, video_mask], dim=0)

        if self.training or return_loss:
            loss = 0.0
            loss_breakdown = {}
            if self.dist_type == "squared_euclidean":
                pdist_matrix = self.get_pairwise_squared_distances(
                    sequence_output,
                    visual_output,
                    attention_mask,
                    video_mask,
                    shaped=True,
                )
                sim_loss, breakdown = self.loss_fct(pdist_matrix)
                loss += sim_loss
                loss_breakdown.update(breakdown)
            else:
                sim_matrix, *_tmp = self.get_similarity_logits(
                    sequence_output,
                    visual_output,
                    attention_mask,
                    video_mask,
                    shaped=True,
                    loose_type=self.loose_type,
                )
                # Sim matrix for the full videos.
                last_sim_matrix = sim_matrix
                if self.return_sequence:
                    last_sim_matrix = sim_matrix[:, :, -1]
                
                if self.loss_type == "sequence_ranking_loss":
                    # Use both sequential and non-sequential losses.
                    sim_loss1, breakdown1 = self.ranking_loss_fct(
                        sim_matrix, labels, video_mask, captions=captions
                    )
                    sim_loss1 *= self.ranking_loss_weight
                    loss += sim_loss1
                    loss_breakdown.update({f"tv_{k}": v for k, v in breakdown1.items()})

                if self.loss_type == "binary_cross_entropy":
                    sim_loss1, breakdown1 = self.loss_fct(
                        last_sim_matrix, labels, captions=captions
                    )
                    loss += sim_loss1
                    loss_breakdown.update({f"tv_{k}": v for k, v in breakdown1.items()})
                else:
                    sim_loss1, breakdown1 = self.loss_fct(last_sim_matrix, labels)
                    labels_T = labels.T if labels is not None else None
                    sim_loss2, breakdown2 = (
                        self.loss_fct(last_sim_matrix.T, labels_T)
                        if self.training or (return_loss == "v2t")
                        else 0.0
                    )
                    sim_loss = (sim_loss1 + sim_loss2) / 2
                    loss += sim_loss
                    loss_breakdown.update({f"tv_{k}": v for k, v in breakdown1.items()})
                    loss_breakdown.update({f"vt_{k}": v for k, v in breakdown2.items()})

            return loss, loss_breakdown
        else:
            return None

    def get_sequence_output(self, input_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        # Use SigLIP text encoder
        text_outputs = self.siglip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_hidden = text_outputs.last_hidden_state.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        # Use SigLIP vision encoder
        vision_outputs = self.siglip.vision_model(pixel_values=video)
        visual_hidden = vision_outputs.last_hidden_state.float()
        
        # Reshape to match expected format
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(
        self,
        input_ids,
        attention_mask,
        video,
        video_mask,
        shaped=False,
        video_frame=-1,
    ):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(
            input_ids, attention_mask, shaped=True
        )
        visual_output = self.get_visual_output(
            video, video_mask, shaped=True, video_frame=video_frame
        )

        return sequence_output, visual_output

    def _get_cross_output(
        self, sequence_output, visual_output, attention_mask, video_mask
    ):
        concat_features = torch.cat(
            (sequence_output, visual_output), dim=1
        )  # concatenate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(
            concat_features, concat_type, concat_mask, output_all_encoded_layers=True
        )
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.0
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(
            attention_mask_un, dim=1, dtype=torch.float
        )
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.0] = 1.0
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(
        self, sequence_output, visual_output, attention_mask, video_mask
    ):
        text_out = self._mean_pooling_for_similarity_sequence(
            sequence_output, attention_mask
        )
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(
        self,
        sequence_output,
        visual_output,
        attention_mask,
        video_mask,
        sim_header="meanP",
    ):
        sequence_output, visual_output = (
            sequence_output.contiguous(),
            visual_output.contiguous(),
        )

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(
                visual_output,
                torch.sum(video_mask, dim=-1).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training:
                self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat(
                (
                    visual_output,
                    visual_output_original[
                        :, visual_output.size(1) :, ...
                    ].contiguous(),
                ),
                dim=1,
            )
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=visual_output.device
            )
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(
            visual_output, video_mask
        )
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1) if sequence_output.dim() > 2 else sequence_output
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        # SigLIP uses different scaling mechanism
        retrieve_logits = torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def _cross_similarity(
        self, sequence_output, visual_output, attention_mask, video_mask
    ):
        sequence_output, visual_output = (
            sequence_output.contiguous(),
            visual_output.contiguous(),
        )

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text  # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # For SigLIP text output, we use the pooled output
        attention_mask = torch.ones(sequence_output.size(0), 1).to(
            device=attention_mask.device, dtype=attention_mask.dtype
        )

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(
                1, b_visual, 1, 1
            )
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = self._get_cross_output(
                sequence_output_l, visual_output_r, attention_mask_l, video_mask_r
            )
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1)
            if self.return_sequence:
                retrieve_logits_row = retrieve_logits_row.view(
                    step_truth, b_visual, pooled_output.size(1)
                )
                # Drop the text feature.
                retrieve_logits_row = retrieve_logits_row[:, :, 1:]
            else:
                retrieve_logits_row = retrieve_logits_row.view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(
        self,
        sequence_output,
        visual_output,
        attention_mask,
        video_mask,
        shaped=False,
        loose_type=False,
    ):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(
                sequence_output,
                visual_output,
                attention_mask,
                video_mask,
                sim_header=self.sim_header,
            )
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(
                sequence_output,
                visual_output,
                attention_mask,
                video_mask,
            )

        return retrieve_logits, contrastive_direction

    def get_pairwise_squared_distances(
        self,
        sequence_output,
        visual_output,
        attention_mask,
        video_mask,
        shaped=False,
    ):
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(
            visual_output, video_mask
        )
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1) if sequence_output.dim() > 2 else sequence_output
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        dists = (
            torch.sum(torch.square(sequence_output), axis=1, keepdims=True)
            + torch.sum(torch.square(visual_output), axis=1, keepdims=True).T
            - 2 * torch.matmul(sequence_output, visual_output.T)
        )
        dists = torch.maximum(dists, 0.0)
        return dists