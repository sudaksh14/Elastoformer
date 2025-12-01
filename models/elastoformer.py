
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
"""Elastoformer model is built on Huggingface ViT: https://github.com/huggingface/transformers/tree/main/src/transformers/models/vit"""

from transformers.models.vit.modeling_vit import (
    ViTPreTrainedModel,
    ViTModel,
    ViTEncoder,
    ViTIntermediate,
    ViTOutput,
    ViTConfig,
    ViTForImageClassification,
    ViTEmbeddings,
    ViTPooler,
)
from transformers.modeling_outputs import ImageClassifierOutput
import torch
from torch import nn
from typing import Dict, List, Optional, Set, Tuple, Union
import math

# -----------------------------
# Elastic Config
# -----------------------------
class ElasticViTConfig(ViTConfig):
    def __init__(self, *args, pruned_dim=768, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruned_dim = pruned_dim


# -----------------------------
# Elastic Self-Attention
# -----------------------------
class ElasticViTSelfAttention(nn.Module):
    def __init__(self, config: ElasticViTConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.pruned_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context = torch.matmul(attention_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))

        outputs = (context, attention_probs) if output_attentions else (context,)
        return outputs

# -----------------------------
# Elastic Self-Output
# -----------------------------
class ElasticViTSelfOutput(nn.Module):
    def __init__(self, config: ElasticViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.pruned_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# -----------------------------
# Elastic Attention Block
# -----------------------------
class ElasticViTAttention(nn.Module):
    def __init__(self, config: ElasticViTConfig):
        super().__init__()
        self.attention = ElasticViTSelfAttention(config)
        self.output = ElasticViTSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask=head_mask, output_attentions=output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

# -----------------------------
# Elastic ViT Layer
# -----------------------------
class ElasticViTLayer(nn.Module):
    """Elastic version of ViTLayer using pruned dimensions for attention."""
    
    def __init__(self, config: ElasticViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ElasticViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

# ----------------------------
# Custom Encoder using Elastic Attention
# ----------------------------
class ElasticViTEncoder(ViTEncoder):
    """
    Elastic encoder that skips ViTEncoder's init check
    but keeps all inherited behavior.
    """
    def __init__(self, config: ElasticViTConfig):
        # skip ViTEncoder init to avoid hidden_size % num_heads check
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList(
            [ElasticViTLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

# ----------------------------
# Custom ViT Model
# ----------------------------
class ElasticViTModel(ViTModel):
    """
    Elastic ViT backbone — replaces encoder with ElasticViTEncoder
    """
    def __init__(self, config: ElasticViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        # do NOT call ViTModel.__init__ to bypass checks
        nn.Module.__init__(self)
        self.config = config

        # Patch embedding and encoder
        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ElasticViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if getattr(config, "use_final_layernorm", True) else None
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

# ----------------------------
# Custom ViTForImageClassification
# ----------------------------
class ElasticViTForImageClassification(ViTForImageClassification):
    """
    Classification head for Elastic ViT — uses ElasticViTModel backbone.
    """
    def __init__(self, config: ElasticViTConfig):
        # skip ViTForImageClassification init to avoid internal checks
        nn.Module.__init__(self)
        self.num_labels = config.num_labels
        self.config = config

        # Elastic backbone
        self.vit = ElasticViTModel(config, add_pooling_layer=False)

        # Classification head (same as HF)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ImageClassifierOutput]:
        """
        Forward pass for ElasticViTForImageClassification.
        Mirrors Hugging Face ViTForImageClassification.forward(), 
        but calls ElasticViTModel.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # Get [CLS] token representation
        pooled_output = outputs[0][:, 0]  # shape: (batch_size, hidden_size)
        # pooled_output = self.layernorm(pooled_output)
        logits = self.classifier(pooled_output)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output

        return ImageClassifierOutput(
            loss=None,  # you can add loss calculation if needed
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )



def from_pretrained_elastic(pretrained_model_name_or_path, elastic_config, **kwargs):
    model = ElasticViTForImageClassification(elastic_config)
    hf_model = ViTForImageClassification.from_pretrained(pretrained_model_name_or_path, **kwargs)

    # Copy compatible weights
    hf_state_dict = hf_model.state_dict()
    model_state_dict = model.state_dict()
    matched_weights = {k: v for k, v in hf_state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(matched_weights, strict=False)

    print(f"[Elastic Loader] Loaded pretrained weights from {pretrained_model_name_or_path}")
    return model
