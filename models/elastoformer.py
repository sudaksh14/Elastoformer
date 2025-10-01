
# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""Elastoformer ViT model built upon code at: """

from transformers.models.vit.modeling_vit import (
    ViTPreTrainedModel,
    ViTModel,
    ViTEncoder,
    ViTLayer,
    ViTAttention,
    ViTSelfAttention,
    ViTSelfOutput,
    ViTConfig,
    ViTForImageClassification,
)
import torch
from torch import nn
from typing import Dict, List, Optional, Set, Tuple, Union
import math


class ElasticViTConfig(ViTConfig):
    def __init__(self, *args, pruned_dim=768, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruned_dim = pruned_dim


class ElasticViTSelfAttention(ViTSelfAttention):
    def __init__(self, config: ElasticViTConfig) -> None:
        super().__init__(config)
        if config.pruned_dim % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.pruned_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        # x shape: (batch, seq_len, all_head_size)
        new_shape = x.size()[:-1] + (num_heads, head_dim)  # (B, T, h, d)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)  # (B, h, T, d)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        # Compute Q, K, V
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Transpose to shape (batch, heads, seq_len, head_dim)
        query_layer = self.transpose_for_scores(query_layer, self.num_attention_heads, self.attention_head_size)
        key_layer = self.transpose_for_scores(key_layer, self.num_attention_heads, self.attention_head_size)
        value_layer = self.transpose_for_scores(value_layer, self.num_attention_heads, self.attention_head_size)

        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Mask heads if needed
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Compute context
        context_layer = torch.matmul(attention_probs, value_layer)  # (B, h, T, d)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (B, T, h, d)
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)  # (B, T, h*d)
        context_layer = context_layer.view(new_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
    
    
class ElasticViTSelfOutput(ViTSelfOutput):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ElasticViTConfig) -> None:
        super().__init__(config)
        self.dense = nn.Linear(config.pruned_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    
class ElasticViTAttention(ViTAttention):
    def __init__(self, config: ElasticViTConfig) -> None:
        super().__init__(config)
        self.attention = ElasticViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

# ----------------------------
# Custom Encoder using Elastic Attention
# ----------------------------
class ElasticViTLayer(ViTLayer):
    """
    Inherits ViTLayer but replaces ViTAttention with ElasticViTAttention
    """
    def __init__(self, config: ElasticViTConfig):
        super().__init__(config)
        # Replace the attention module with Elastic
        self.attention = ElasticViTAttention(config)

class ElasticViTEncoder(ViTEncoder):
    """
    Inherits ViTEncoder but uses ElasticViTLayer
    """
    def __init__(self, config: ElasticViTConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([ElasticViTLayer(config) for _ in range(config.num_hidden_layers)])

# ----------------------------
# Custom ViT Model
# ----------------------------
class ElasticViTModel(ViTModel):
    def __init__(self, config: ElasticViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config, add_pooling_layer=add_pooling_layer, use_mask_token=use_mask_token)
        # Replace encoder with elastic encoder
        self.encoder = ElasticViTEncoder(config)

# ----------------------------
# Custom ViTForImageClassification
# ----------------------------
class ElasticViTForImageClassification(ViTForImageClassification):
    def __init__(self, config: ElasticViTConfig):
        super().__init__(config)
        # Replace the base vit model with our elastic variant
        self.vit = ElasticViTModel(config, add_pooling_layer=False)


def from_pretrained_elastic(
    pretrained_model_name_or_path: str,
    elastic_config: ElasticViTConfig,
    map_location: str = None,
    strict: bool = True,
    **kwargs
) -> ElasticViTForImageClassification:
    """
    Loads a HF ViT checkpoint into the ElasticViTForImageClassification model.

    Args:
        pretrained_model_name_or_path (str): HF model identifier or local checkpoint.
        elastic_config (ElasticViTConfig): Your custom elastic configuration.
        map_location (str, optional): Device mapping for torch.load.
        strict (bool, optional): Whether to strictly enforce key matching when loading state_dict.
        kwargs: Additional kwargs passed to HF `from_pretrained`.

    Returns:
        ElasticViTForImageClassification: The model with HF weights loaded.
    """

    # Initialize your elastic model
    model = ElasticViTForImageClassification(elastic_config)

    # Load HF state dict
    hf_model = ViTForImageClassification.from_pretrained(pretrained_model_name_or_path, **kwargs)
    hf_state_dict = hf_model.state_dict()

    # Remove any incompatible keys or adjust shapes if needed
    elastic_state_dict = model.state_dict()
    mapped_state_dict = {}
    for k, v in hf_state_dict.items():
        if k in elastic_state_dict and v.shape == elastic_state_dict[k].shape:
            mapped_state_dict[k] = v
        else:
            print(f"[Elastic Loader] Skipping incompatible key: {k}, HF shape: {v.shape}, Elastic shape: {elastic_state_dict.get(k).shape if k in elastic_state_dict else 'N/A'}")

    # Load state dict into elastic model
    model.load_state_dict(mapped_state_dict, strict=strict)

    if map_location:
        model = model.to(map_location)

    print(f"[Elastic Loader] Loaded pretrained weights from {pretrained_model_name_or_path}")
    return model