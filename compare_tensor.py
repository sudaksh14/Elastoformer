import torch
import torch.nn as nn
from main import create_vit_general 
from utils.prune_utils import get_vit_info
from utils.cnn_prune_utils import resnet_generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

torch.manual_seed(42)

def check_overlap(big_tensor, small_tensor):
    """
    Check if there's any overlap between two tensors.

    Args:
        big_tensor: The larger tensor.
        small_tensor: The smaller tensor.

    Returns:
        A boolean indicating whether there's overlap.
    """

    # print(f"Small Tensor: {small_tensor[:10, :10]}")
    # print(f"Big Tensor Shape: {big_tensor[:10, :10]}")

    print(f"Small Tensor Shape: {small_tensor.shape}")
    print(f"Big Tensor Shape: {big_tensor.shape}")

    overlap = torch.isin(small_tensor, big_tensor)

    print(f"Overlap found: {overlap.any().item()}")  # True if there's at least one overlap
    print(f"Matching values: {small_tensor[overlap].shape}")  # Prints matching values
    print(f"All values Nested: {small_tensor.flatten().shape == small_tensor[overlap].shape}")

def check_similarity(t1, t2):
        # Check shape
        print(f"  - Shape 1: {t1.shape}, Shape 2: {t2.shape}")
        # Compute cosine similarity or L2 norm difference
        diff = (t1 - t2).abs().mean().item()
        print(f"  - Mean Absolute Difference: {diff:.6f}")
        cos_sim = torch.nn.functional.cosine_similarity(t1.flatten(), t2.flatten(), dim=0).item()
        print(f"  - Cosine Similarity: {cos_sim:.6f}")

def compare_weights(sample_layer="vit.encoder.layer.0.attention.attention.key"):
    
        path_core = "./saves/state_dicts/Vit_b_16_Pruned_0.25_state_dict_ViT_Adaptivity_selective_clipping_full_data.pth"
        path_rebuilt = "./saves/state_dicts/Vit_b_16_Rebuilt_0.25_state_dict_ViT_Adaptivity_selective_clipping_full_data.pth"

        prune_dict = torch.load(path_core, map_location=device)
        rebuilt_dict = torch.load(path_rebuilt, map_location=device)

        print(prune_dict.keys())
        print(prune_dict['vit.encoder.layer.1.attention.attention.query.weight'].shape)
        print(prune_dict['vit.encoder.layer.1.intermediate.dense.weight'].shape)


        prune_embed = prune_dict['vit.encoder.layer.0.attention.attention.query.weight'].shape[0]
        prune_ff = prune_dict['vit.encoder.layer.0.intermediate.dense.weight'].shape[0]

        rebuilt_embed = rebuilt_dict['vit.encoder.layer.0.attention.attention.query.weight'].shape[0]
        rebuilt_ff = rebuilt_dict['vit.encoder.layer.0.intermediate.dense.weight'].shape[0]

        vit_core = create_vit_general(embed_dim=prune_embed, output_dim=prune_embed, ff_hidden_dim=prune_ff, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        vit_rebuilt = create_vit_general(embed_dim=rebuilt_embed, output_dim=rebuilt_embed, ff_hidden_dim=rebuilt_ff, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)

        vit_core.load_state_dict(prune_dict)
        vit_rebuilt.load_state_dict(rebuilt_dict)

        vit_core.eval()
        vit_rebuilt.eval()

        for name, layer in vit_core.named_modules():
            if isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
                print(f"Checking overlap for layer: {name}")
                layer_core = dict(vit_core.named_modules()).get(name)
                layer_rebuilt = dict(vit_rebuilt.named_modules()).get(name)
                check_overlap(layer_rebuilt.weight.data, layer_core.weight.data)
                print("")


def compare_weights_multilevel(model_paths, prune_steps=3):
    """
    Compare weights between multiple versions (levels) of ViT models.
    model_paths: List of paths to state_dicts [level1, level2, level3, level4]
    """
    models = []
    state_dicts = [torch.load(p, map_location=device) for p in model_paths]

    print(state_dicts[0].keys())
    print(state_dicts[0]['vit.encoder.layer.1.attention.attention.query.weight'].shape)
    print(state_dicts[0]['vit.encoder.layer.1.intermediate.dense.weight'].shape)

    for sd in state_dicts:
        # Extract model innformation from state dict
        model_info = get_vit_info(non_pruned_weights=sd, core_model=True)
        model = create_vit_general(dim_dict=model_info)
        models.append(model)
    
    for model, state in zip(models, state_dicts):
        model.load_state_dict(state)
        model.eval()

    # Compare each level with the next: 1->2, 2->3, 3->4
    for i in range(prune_steps):
        print(f"\n🔍 Comparing Level {i+2} ➡️ Level {i+1}")
        for name, layer in models[i+1].named_modules():
            if isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
                layer_core = dict(models[i].named_modules()).get(name)
                if layer_core is not None:
                    print(f"Checking overlap for layer: {name}")
                    check_overlap(layer.weight.data, layer_core.weight.data)
                    print("")


def get_model_info_resnet(state_dict=None):
    """
    Extracts the channel dimensions for each Conv and BatchNorm layer in a ResNet model.

    Args:
        pruned_weights (dict): Dictionary of pruned weights.
        non_pruned_weights (dict): Dictionary of non-pruned weights.
        core_model (bool): Whether to use core model only (no pruned weights).

    Returns:
        dict: Dictionary with channel sizes for each layer.
    """
    model_info = {}

    for layer_name, layer_weights in state_dict.items():
        layer_info = {}

        if "weight" in layer_name:
            weight = layer_weights
            if len(weight.shape) > 1: 
                layer_info["out_channels"] = weight.shape[0]
                layer_info["in_channels"] = weight.shape[1]
        
        if layer_info:
            model_info[layer_name.replace('.weight', '').replace('.bias', '')] = (layer_info["out_channels"], layer_info["in_channels"]) if "out_channels" in layer_info else (layer_info["num_features"], None)

    return model_info


def compare_resnet_weights_multilevel(model_paths, prune_steps=3, device='cpu'):
    """
    Compare weights between multiple versions (levels) of ResNet models.

    Args:
        model_paths (List[str]): Paths to model state_dicts [level1, level2, level3, ...]
        prune_steps (int): Number of levels to compare
        model_creator (callable): Function that returns a ResNet model instance
        device (str or torch.device): Device to map the models to
    """
    state_dicts = [torch.load(p, map_location=device) for p in model_paths]
    models = []

    # Initialize models using provided model creator
    for sd in state_dicts:
        model_info = get_model_info_resnet(sd)
        model = resnet_generator(arch="resnet50", channel_dict=model_info)
        model.load_state_dict(sd)
        model.eval()
        models.append(model.to(device))

    for i in range(prune_steps):
        print(f"\n🔍 Comparing Level {i+2} ➡️ Level {i+1}")
        model_high = models[i+1]
        model_low = models[i]

        for (name_high, layer_high), (name_low, layer_low) in zip(model_high.named_modules(), model_low.named_modules()):
            if isinstance(layer_high, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                if hasattr(layer_high, 'weight') and hasattr(layer_low, 'weight'):
                    try:
                        print(f"\n📌 Layer: {name_high}")
                        check_overlap(layer_high.weight.data, layer_low.weight.data)

                        # If the layer has bias
                        if hasattr(layer_high, 'bias') and layer_high.bias is not None:
                            check_overlap(layer_high.bias.data, layer_low.bias.data)
                    except Exception as e:
                        print(f"⚠️ Error comparing layer {name_high}: {e}")

     
if __name__ == '__main__':
    # paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_deit_Iter_Adaptivity_lowlr.pth"] + \
    #         [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_deit_Iter_Adaptivity_lowlr.pth" for i in range(2,7)]
    # compare_weights_multilevel(paths, prune_steps=5)

    compare_resnet_weights_multilevel(
        model_paths=["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_Resnet50_Iter_Adaptivity_SGD.pth"] + \
            [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_Resnet50_Iter_Adaptivity_SGD.pth" for i in range(2,7)],
        prune_steps=5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )