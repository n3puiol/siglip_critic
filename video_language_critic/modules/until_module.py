# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import logging
import functools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from .until_config import PretrainedConfig

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class PreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if "beta" in dir(module) and "gamma" in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="")

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info(
                    "Weights of {} not initialized from pretrained model: {}".format(
                        model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)
                    )
                )
            if len(unexpected_keys) > 0:
                logger.info(
                    "Weights from pretrained model not used in {}: {}".format(
                        model.__class__.__name__,
                        "\n   " + "\n   ".join(unexpected_keys),
                    )
                )
            if len(error_msgs) > 0:
                logger.error(
                    "Weights from pretrained model cause errors in {}: {}".format(
                        model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)
                    )
                )

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [
                    (k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)
                ]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model


##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(
        self,
    ):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix, labels=None):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        if labels is None:
            logpt = torch.diag(logpt)
            nce_loss = -logpt
            sim_loss = nce_loss.mean()
        else:
            sim_loss = (-logpt * labels).sum() / labels.sum()

        return sim_loss, {}


class BinaryCrossEn(nn.Module):
    def __init__(
        self,
    ):
        super(BinaryCrossEn, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, sim_matrix, labels=None, captions=None):
        pos = sim_matrix[labels.bool()]
        n_ts, n_vs = sim_matrix.shape
        assert n_vs % 2 == 0
        tiled_captions = np.tile(captions, [len(captions), 1])
        identical_captions = torch.Tensor(
            np.equal(tiled_captions, tiled_captions.T)
        ).to(sim_matrix.device)

        # Make sure the negative caption is different.
        possible_negatives = torch.logical_not(identical_captions).int()
        negs = torch.zeros([n_ts, n_vs // 2]).to(sim_matrix.device)
        # Deterministically pick the next index in batch out of possible negatives.
        pos_caption_idx = 0
        for i in range(n_vs // 2):
            # If caption is positive.
            if torch.any(labels[:, i]):
                if torch.any(possible_negatives[i, i + 1 :]):
                    negs[
                        pos_caption_idx,
                        i + 1 + torch.argmax(possible_negatives[i, i + 1 :]),
                    ] = 1
                elif torch.any(possible_negatives[i, :i]):
                    negs[pos_caption_idx, torch.argmax(possible_negatives[i, :i])] = 1
                pos_caption_idx += 1

        neg = sim_matrix[:, : n_vs // 2][negs.bool()]
        # Reversed negatives.
        rev_negs = sim_matrix[:, n_vs // 2 :][labels[:, : n_vs // 2].bool()]

        targets = torch.cat(
            [
                torch.ones(len(pos)),
                torch.zeros(len(neg)),
                torch.zeros(len(rev_negs)),
            ]
        ).to(sim_matrix.device)
        preds = torch.cat([pos, neg, rev_negs])
        sim_loss = self.bce(preds, targets)

        return sim_loss, {}


class MILNCELoss(nn.Module):
    def __init__(
        self,
        batch_size=1,
        n_pair=1,
    ):
        super(MILNCELoss, self).__init__()
        self.batch_size = batch_size
        self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix):
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair)))
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (
            self.n_pair // 2
        )
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(
            logpt_choice.to(dtype=self.bool_dtype)
        ).mean()
        return sim_loss


class SemiHardNegativeTripletLoss(nn.Module):
    def __init__(self, margin=1.0, progress_margin=None):
        super(SemiHardNegativeTripletLoss, self).__init__()
        self.margin = margin
        self.progress_margin = progress_margin

    def _max_in_subset(self, data, subset, dim=1):
        axis_min, _ = torch.min(data, dim, keepdim=True)
        set_max, _ = torch.max(
            torch.multiply(data - axis_min, subset), dim, keepdim=True
        )
        set_max += axis_min
        return set_max.view(-1)

    def _min_in_subset(self, data, subset, dim=1):
        axis_max, _ = torch.max(data, dim, keepdim=True)
        set_min, _ = torch.min(
            torch.multiply(data - axis_max, subset), dim, keepdim=True
        )
        set_min += axis_max
        return set_min.view(-1)

    def progress_loss(self, sim_matrix1, sim_matrix2, labels=None):
        """Penalize sim_matrix1[i, i] + a > sim_matrix2[i, i]."""
        if labels is None:
            loss = torch.relu(
                torch.diag(sim_matrix1) + self.progress_margin - torch.diag(sim_matrix2)
            )
        else:
            loss = torch.relu(
                torch.masked_select(sim_matrix1, labels)
                + self.progress_margin
                - torch.masked_select(sim_matrix2, labels)
            )
        return loss.mean()

    def semihardneg_progress_loss(self, sim_matrix, labels=None):
        """Add the SHN loss to max(0, S_k + a < S_{k+1}) for each progress step k.

        Where S_k = sim_matrix[lang, v_k].
        """
        transform = lambda x: x
        transformed_labels = labels
        if sim_matrix.shape[0] > sim_matrix.shape[1]:
            sim_matrix = sim_matrix.transpose(1, 0)
            transform = functools.partial(torch.transpose, dim0=1, dim1=0)
            transformed_labels = None if labels is None else transform(labels)
        n_text = sim_matrix.shape[0]
        n_video = sim_matrix.shape[1]
        n_progress_steps = n_video // n_text
        # Apply triplet loss per progress step to avoid short videos being far from everything.
        loss_breakdown = {}
        shn_loss = torch.concat(
            [
                self.semihardneg_loss(
                    transform(sim_matrix[:, i * n_text : (i + 1) * n_text]),
                    transformed_labels,
                ).view(1)
                for i in range(n_progress_steps)
            ]
        )
        for i in range(n_progress_steps):
            loss_breakdown[f"shn_loss_{i}"] = shn_loss[i]
        shn_loss = torch.mean(shn_loss)
        prog_loss = torch.concat(
            [
                self.progress_loss(
                    transform(sim_matrix[:, i * n_text : (i + 1) * n_text]),
                    transform(sim_matrix[:, (i + 1) * n_text : (i + 2) * n_text]),
                    transformed_labels,
                ).view(1)
                for i in range(n_progress_steps - 1)
            ]
        )
        for i in range(n_progress_steps - 1):
            loss_breakdown[f"prog_loss_{i}"] = prog_loss[i]
        prog_loss = torch.mean(prog_loss)

        loss_breakdown["shn_loss"] = shn_loss
        loss_breakdown["prog_loss"] = prog_loss
        loss_breakdown = {
            k: v.cpu().detach().numpy() for k, v in loss_breakdown.items()
        }
        for k, v in loss_breakdown.items():
            loss_breakdown[k] = v.item() if v.shape == () else v
        return shn_loss + prog_loss, loss_breakdown

    def semihardneg_loss(self, sim_matrix, labels=None):
        if labels is None:
            pos = torch.diag(sim_matrix)
            neg_mask = torch.logical_not(
                torch.eye(sim_matrix.shape[0]).to(sim_matrix.device)
            )
        else:
            pos = self._min_in_subset(sim_matrix, labels)
            neg_mask = torch.logical_not(labels)
        # Find i, j that are a worse pair than i and its positive pair.
        shn_cond = torch.less(sim_matrix, pos.view(-1, 1))
        # if torch.any(torch.diag(shn_cond)):
        #     # The positive itself should not be smaller than the positive.
        #     import pdb

        #     pdb.set_trace()
        # Choose the largest j that is a worse pair than i and its positive pair.
        shn_largest = self._max_in_subset(sim_matrix, shn_cond)
        # If there is no worse pair, use the smallest j as negative.
        shn_smallest = self._min_in_subset(sim_matrix, neg_mask)
        shn = torch.where(torch.any(shn_cond, dim=1), shn_largest, shn_smallest)
        triplet_loss = torch.relu(shn + self.margin - pos).mean()
        return triplet_loss, {}

    def forward(self, sim_matrix, labels=None):
        h = sim_matrix.shape[0]
        w = sim_matrix.shape[1]
        # Remove rows that have no positive label.
        if labels is not None:
            keep_row = labels.sum(dim=1) > 0
            sim_matrix = sim_matrix[keep_row]
            labels = labels[keep_row]
        if self.progress_margin is None:
            return self.semihardneg_loss(sim_matrix, labels)
        elif (h > w and h % w == 0) or (w > h and w % h == 0):
            return self.semihardneg_progress_loss(sim_matrix, labels)
        else:
            raise RuntimeError(
                f"Incompatible similarity matrix shape: {sim_matrix.shape}."
            )


class SequentialRankingLoss(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, x, labels, x_mask, *args, **kwargs):
        if labels is None:
            labels = torch.eye(x.size(0))
        x = x[labels.bool()]
        keep_mask_row = labels.sum(dim=0) > 0
        x_mask = x_mask[keep_mask_row]

        next_x = x[:, 1:]
        diffs = torch.relu(x[:, :-1] - next_x + self.margin)
        has_next = torch.concat(
            [
                x_mask[:, 1:],
                torch.zeros(x_mask.shape[0], 1).to(device=x_mask.device, dtype=int),
            ],
            dim=1,
        )
        diffs = diffs * has_next[:, :-1]
        col_diffs = diffs.sum(dim=0)
        has_next_cols = has_next[:, :-1].sum(dim=0)
        col_losses = col_diffs / has_next_cols
        loss = col_diffs.sum() / has_next_cols.sum()

        loss_breakdown = {"ranking_loss": float(loss)}
        for i, l in enumerate(col_losses):
            loss_breakdown[f"ranking_loss_t{i}"] = float(l)
        return loss, loss_breakdown


class MaxMarginRankingLoss(nn.Module):
    def __init__(
        self,
        margin=1.0,
        negative_weighting=False,
        batch_size=1,
        n_pair=1,
        hard_negative_rate=0.5,
    ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + F.relu(
            self.margin + x - d.view(1, -1)
        )
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )
