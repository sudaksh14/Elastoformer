import timm
import torch
from transformers import ViTForImageClassification, ViTConfig, AutoImageProcessor
import os

def convert_timm_to_hf(timm_model_name, save_dir="./hf_model"):
    """
    Convert a TIMM DeiT/ViT model to Hugging Face ViTForImageClassification.
    Works for deit3_base_patch16_224.fb_in22k_ft_in1k.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1️⃣ Load TIMM model
    timm_model = timm.create_model(timm_model_name, pretrained=True)
    timm_model.eval()
    
    # 2️⃣ Create HF config
    config = ViTConfig(
        hidden_size=timm_model.embed_dim,
        num_hidden_layers=len(timm_model.blocks),
        num_attention_heads=timm_model.blocks[0].attn.num_heads,
        intermediate_size=timm_model.blocks[0].mlp.fc1.out_features,
        image_size=224,
        patch_size=timm_model.patch_embed.patch_size[0],
        num_channels=3,
        num_labels=timm_model.head.out_features,
    )
    
    # 3️⃣ Initialize HF model
    hf_model = ViTForImageClassification(config)
    
    # 4️⃣ Transfer patch embeddings
    hf_model.vit.embeddings.patch_embeddings.projection.weight.data.copy_(timm_model.patch_embed.proj.weight.data)
    hf_model.vit.embeddings.patch_embeddings.projection.bias.data.copy_(timm_model.patch_embed.proj.bias.data)
    
    # 5️⃣ Transfer class token & position embeddings
    hf_model.vit.embeddings.cls_token.data.copy_(timm_model.cls_token.data)
    hf_model.vit.embeddings.position_embeddings.data[:, 1:, :].copy_(timm_model.pos_embed.data)
    
    # 6️⃣ Transfer encoder blocks
    for i, block in enumerate(timm_model.blocks):
        hf_block = hf_model.vit.encoder.layer[i]
        
        D = timm_model.embed_dim  # hidden_dim, e.g., 768

        qkv = block.attn.qkv.weight.data       # [3*D, D]
        qkv_bias = block.attn.qkv.bias.data    # [3*D]

        # Split Q, K, V
        hf_block.attention.attention.query.weight.data.copy_(qkv[0:D, :])
        hf_block.attention.attention.key.weight.data.copy_(qkv[D:2*D, :])
        hf_block.attention.attention.value.weight.data.copy_(qkv[2*D:3*D, :])

        # Split bias
        hf_block.attention.attention.query.bias.data.copy_(qkv_bias[0:D])
        hf_block.attention.attention.key.bias.data.copy_(qkv_bias[D:2*D])
        hf_block.attention.attention.value.bias.data.copy_(qkv_bias[2*D:3*D])

        
        # Attention output projection
        hf_block.attention.output.dense.weight.data.copy_(block.attn.proj.weight.data)
        hf_block.attention.output.dense.bias.data.copy_(block.attn.proj.bias.data)
        
        # MLP
        hf_block.intermediate.dense.weight.data.copy_(block.mlp.fc1.weight.data)
        hf_block.intermediate.dense.bias.data.copy_(block.mlp.fc1.bias.data)
        hf_block.output.dense.weight.data.copy_(block.mlp.fc2.weight.data)
        hf_block.output.dense.bias.data.copy_(block.mlp.fc2.bias.data)
        
        # LayerNorms at block level
        hf_block.layernorm_before.weight.data.copy_(block.norm1.weight.data)
        hf_block.layernorm_before.bias.data.copy_(block.norm1.bias.data)
        hf_block.layernorm_after.weight.data.copy_(block.norm2.weight.data)
        hf_block.layernorm_after.bias.data.copy_(block.norm2.bias.data)
    
    # 7️⃣ Transfer classifier
    hf_model.classifier.weight.data.copy_(timm_model.head.weight.data)
    hf_model.classifier.bias.data.copy_(timm_model.head.bias.data)
    
    # 8️⃣ Save HF model & processor
    hf_model.save_pretrained(save_dir)
    
    print(f"✅ Conversion complete! Hugging Face model saved to {save_dir}")
    return hf_model

# Example usage
if __name__ == "__main__":
    convert_timm_to_hf("deit3_base_patch16_224.fb_in22k_ft_in1k", save_dir="./hf_deit3_base_patch16_224")
