import torch
from torch import nn
import math
from typing import Optional, Tuple, Union
from transformers import ViTForImageClassification

# ----------------------------
# Config
# ----------------------------
class ElasticViTConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        pruned_dim=768,
        num_labels=1000,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.pruned_dim = pruned_dim
        self.num_labels = num_labels

# ----------------------------
# Self-Attention
# ----------------------------
class ElasticViTSelfAttention(nn.Module):
    def __init__(self, config: ElasticViTConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.pruned_dim // self.num_heads
        self.all_head_dim = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_dim)
        self.key = nn.Linear(config.hidden_size, self.all_head_dim)
        self.value = nn.Linear(config.hidden_size, self.all_head_dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, H, T, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        if head_mask is not None:
            probs = probs * head_mask

        context = torch.matmul(probs, v)  # (B, H, T, D)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(hidden_states.size(0), hidden_states.size(1), self.all_head_dim)

        return (context, probs) if output_attentions else (context,)

# ----------------------------
# Self-Output
# ----------------------------
class ElasticViTSelfOutput(nn.Module):
    def __init__(self, config: ElasticViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.pruned_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# ----------------------------
# Attention Block
# ----------------------------
class ElasticViTAttention(nn.Module):
    def __init__(self, config: ElasticViTConfig):
        super().__init__()
        self.attention = ElasticViTSelfAttention(config)
        self.output = ElasticViTSelfOutput(config)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

# ----------------------------
# Encoder Layer
# ----------------------------
class ElasticViTLayer(nn.Module):
    def __init__(self, config: ElasticViTConfig):
        super().__init__()
        self.attention = ElasticViTAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, head_mask=None, output_attentions=False, **kwargs):
        # Attention
        self_attention_outputs = self.attention(self.layernorm_before(hidden_states), head_mask, output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = hidden_states + attention_output

        # Feedforward
        layer_output = self.layernorm_after(hidden_states)
        intermediate_output = self.intermediate(layer_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)

        hidden_states = hidden_states + layer_output

        return (hidden_states,) + outputs

# ----------------------------
# Encoder
# ----------------------------
class ElasticViTEncoder(nn.Module):
    def __init__(self, config: ElasticViTConfig):
        super().__init__()
        self.layer = nn.ModuleList([ElasticViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, head_mask=None, output_attentions=False, **kwargs):
        all_attentions = []
        for layer in self.layer:
            layer_outputs = layer(hidden_states, head_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions.append(layer_outputs[1])
        return (hidden_states, all_attentions) if output_attentions else (hidden_states,)

# ----------------------------
# Full Model
# ----------------------------
class ElasticViTModel(nn.Module):
    def __init__(self, config: ElasticViTConfig):
        super().__init__()
        self.config = config
        self.encoder = ElasticViTEncoder(config)

    def forward(self, hidden_states, head_mask=None, output_attentions=False, **kwargs):
        return self.encoder(hidden_states, head_mask, output_attentions)

# ----------------------------
# ViT for Image Classification
# ----------------------------
class ElasticViTForImageClassification(nn.Module):
    def __init__(self, config: ElasticViTConfig):
        super().__init__()
        self.config = config
        self.vit = ElasticViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values, head_mask=None, output_attentions=False, **kwargs):
        hidden_states, attentions = self.vit(pixel_values, head_mask, output_attentions)
        cls_token = hidden_states[:, 0]
        logits = self.classifier(cls_token)
        return logits
    

def from_pretrained_elastic(pretrained_model_name_or_path: str, config: ElasticViTConfig, device: str = "cpu"):
    """
    Loads HF ViT/DeiT weights into ElasticViTForImageClassification.

    Args:
        pretrained_model_name_or_path (str): HF model name or local checkpoint.
        config (ElasticViTConfig): Your custom ElasticViT config.
        device (str): Device to map the model to.

    Returns:
        ElasticViTForImageClassification: Model with HF weights loaded where compatible.
    """

    # Initialize your custom Elastic model
    model = ElasticViTForImageClassification(config)

    # Load Hugging Face ViTForImageClassification
    hf_model = ViTForImageClassification.from_pretrained(pretrained_model_name_or_path)
    hf_state_dict = hf_model.state_dict()
    elastic_state_dict = model.state_dict()

    # Map compatible weights only
    mapped_state_dict = {}
    for k, v in hf_state_dict.items():
        if k in elastic_state_dict:
            if v.shape == elastic_state_dict[k].shape:
                mapped_state_dict[k] = v
            else:
                # Skip weights that don’t match (e.g., pruned attention heads)
                print(f"[Elastic Loader] Skipping incompatible key: {k}, HF shape: {v.shape}, Elastic shape: {elastic_state_dict[k].shape}")
        else:
            print(f"[Elastic Loader] Key not found in Elastic model: {k}")

    # Load compatible weights
    model.load_state_dict(mapped_state_dict, strict=False)

    # Move model to device
    model.to(device)

    print(f"[Elastic Loader] Loaded pretrained weights from {pretrained_model_name_or_path} onto ElasticViT")
    return model
