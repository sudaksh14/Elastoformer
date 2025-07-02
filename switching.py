import torch
import torch.nn as nn
import time
from ViT_iterative_adaptivity import create_vit_general
from prune_utils import get_vit_info, merge_and_remap_indices
from transformers.models.vit.modeling_vit_pruned import ViTForImageClassification
from cnn_prune_utils import resnet_generator
import torchvision


def load_vit_model(state_dict_path=None, device='cuda'):
    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location=device)
        
    model_info = get_vit_info(non_pruned_weights=state_dict, core_model=True)
    model = create_vit_general(dim_dict=model_info)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

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

def load_resnet_model(state_dict_path=None, device='cuda'):
    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location=device)

    model_info = get_model_info_resnet(state_dict)
    model = resnet_generator(arch="resnet50", channel_dict=model_info)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def initialize_growing_vit(config, 
                                   primary_index_map: dict, 
                                   primary_weights: dict,
                                   secondary_index_map: dict,
                                   secondary_weights: dict):
    """
    Initialize a ViT model from config and selectively fill weights from two sources.

    Args:
        config (ViTConfig): Configuration for the new (larger) model.
        primary_index_map (dict): {new_param_name: (old_param_name, index_tensor)} from primary (e.g., small model).
        primary_weights (dict): Pretrained state_dict from primary source.
        secondary_index_map (dict): Same format as above, for remaining weights.
        secondary_weights (dict): Secondary source of weights.
    
    Returns:
        model (ViTForImageClassification): Initialized model.
    """

    start_time = time.perf_counter()
    model = create_vit_general(dim_dict=config)
    new_state_dict = model.state_dict()

    for new_name, param in new_state_dict.items():
        assigned = False

        # Try copying from primary weights
        if new_name in primary_index_map:
            index_tensor = primary_index_map[new_name]
            try:
                new_state_dict[new_name].copy_(primary_weights[old_name][index_tensor])
                assigned = True
            except Exception as e:
                print(f"[Primary] Failed for {new_name}: {e}")

        # Try copying from secondary weights if not assigned
        if not assigned and new_name in secondary_index_map:
            old_name, index_tensor = secondary_index_map[new_name]
            if old_name in secondary_weights:
                try:
                    new_state_dict[new_name].copy_(secondary_weights[old_name][index_tensor])
                    assigned = True
                except Exception as e:
                    print(f"[Secondary] Failed for {new_name} <- {old_name}: {e}")

        # Else leave as randomly initialized
        if not assigned:
            print(f"[Init] Using random init for: {new_name}")

    end_time = time.perf_counter()
    print(f"✅ Model initialized in {end_time - start_time:.4f} seconds.")

    # Load final state dict
    model.load_state_dict(new_state_dict)
    return model

def update_vit_weights_global(model, pruned_indices_list, non_pruned_indices_list, pruned_weights_dict, non_pruned_weights_dict, include_bias=True, device=None):
    """
    Updates the weights of a Vision Transformer (ViT) model with merged and reindexed indices.

    The pruned and non-pruned indices are combined and reindexed to a contiguous range starting from 0.

    Args:
        model (torch.nn.Module): The ViT model.
        pruned_indices_list (tuple): Tuple of two dicts (in_indices, out_indices).
        non_pruned_indices_list (tuple): Tuple of two dicts (in_indices, out_indices).
        pruned_weights_dict (dict): Dict of {layer_name: {"Weight": ..., "Bias": ...}} for pruned.
        non_pruned_weights_dict (dict): Same as above for non-pruned.
        include_bias (bool): Whether to update biases.
        device (torch.device or str): The device to move tensors to.
    """
    
    pruned_in = pruned_indices_list[0]
    pruned_out = pruned_indices_list[1]

    non_pruned_in = non_pruned_indices_list[0]
    non_pruned_out = non_pruned_indices_list[1]

    pruned_in_mapped = {}
    pruned_out_mapped = {}
    non_pruned_in_mapped = {}
    non_pruned_out_mapped = {}

    for layer_name, layer in model.named_modules():

        if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            continue

        if layer_name not in (pruned_out.keys() | pruned_in.keys()):
            # print(f"Warning: Layer '{layer_name}' not found in the Pruning Metadata")
            continue

        if not hasattr(layer, 'weight'):
            continue

        weight = layer.weight.data

        pruned_weights = pruned_weights_dict[layer_name]

        non_pruned_weights = {
        "Weight": non_pruned_weights_dict.get(f"{layer_name}.weight"),
        "Bias": non_pruned_weights_dict.get(f"{layer_name}.bias")
    }

        if layer_name == 'classifier':
            pruned_dim1, non_pruned_dim1, _ = merge_and_remap_indices(pruned_in[layer_name], non_pruned_in[layer_name])

            weight[:, pruned_dim1] = torch.tensor(pruned_weights["Weight"], device=device)
            weight[:, non_pruned_dim1] = torch.tensor(non_pruned_weights["Weight"], device=device)
            
            pruned_in_mapped[layer_name] = pruned_dim1
            non_pruned_in_mapped[layer_name] = non_pruned_dim1

        elif isinstance(layer, nn.Linear):
            pruned_dim0, non_pruned_dim0, _ = merge_and_remap_indices(pruned_out[layer_name], non_pruned_out[layer_name])
            pruned_dim1, non_pruned_dim1, _ = merge_and_remap_indices(pruned_in[layer_name], non_pruned_in[layer_name])
            
            try:
                rows_pruned = torch.tensor(pruned_dim0, device=device)
                cols_pruned = torch.tensor(pruned_dim1, device=device)

                # Replace pruned weights
                if rows_pruned.numel() > 0 and cols_pruned.numel() > 0:
                    row_idx, col_idx = torch.meshgrid(rows_pruned, cols_pruned, indexing="ij")
                    weight[row_idx, col_idx] = torch.tensor(pruned_weights["Weight"], device=device)

                rows_non_pruned = torch.tensor(non_pruned_dim0, device=device)
                cols_non_pruned = torch.tensor(non_pruned_dim1, device=device)
                
                # Replace non-pruned weights
                if rows_non_pruned.numel() > 0 and cols_non_pruned.numel() > 0:
                    row_idx, col_idx = torch.meshgrid(rows_non_pruned, cols_non_pruned, indexing="ij")
                    weight[row_idx, col_idx] = torch.tensor(non_pruned_weights["Weight"], device=device)
            
            except Exception as e:
                print(f"Failed at: {layer_name}")
                print("Error:", e)

            pruned_out_mapped[layer_name] = pruned_dim0
            non_pruned_out_mapped[layer_name] = non_pruned_dim0
            pruned_in_mapped[layer_name] = pruned_dim1
            non_pruned_in_mapped[layer_name] = non_pruned_dim1

        elif isinstance(layer, (nn.LayerNorm, nn.Conv2d)):
            pruned_dim0, non_pruned_dim0, _ = merge_and_remap_indices(pruned_out[layer_name], non_pruned_out[layer_name])

            weight[torch.tensor(pruned_dim0)] = torch.tensor(pruned_weights["Weight"], device=device)
            weight[torch.tensor(non_pruned_dim0)] = torch.tensor(non_pruned_weights["Weight"], device=device)

            pruned_out_mapped[layer_name] = pruned_dim0
            non_pruned_out_mapped[layer_name] = non_pruned_dim0


        if include_bias and layer.bias is not None:
            if layer_name == 'classifier':
                layer.bias.data = torch.tensor(non_pruned_weights["Bias"], device=device)
            else:
                bias = layer.bias.data
                if len(pruned_dim0) > 0:
                    bias[torch.tensor(pruned_dim0)] = torch.tensor(pruned_weights["Bias"], device=device)
                if len(non_pruned_dim0) > 0:
                    bias[torch.tensor(non_pruned_dim0)] = torch.tensor(non_pruned_weights["Bias"], device=device)

    return model.to(device)

if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_ViT_Iter_Adaptivity_test_multistep.pth"] + \
            [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_ViT_Iter_Adaptivity_test_multistep.pth" for i in range(2,7)]
    
    configs = []
    for sd in paths:
        configs.append(get_vit_info(non_pruned_weights = torch.load(sd, map_location=device), core_model=True))

    metadata_paths = [f"./saves/pruning_metadata/ViT_Iter_Adaptivity_test_multistep_pruning_metadata_Level_{i}.pth" for i in range(2,7)]

    print("------------------------------------CUSTOM INIT--------------------------------------")
    for i in range(4):
        print(f"Initializing model for level {i+1}...")
        print(f"Config: {configs[i+1]}")
        
        core_model = create_vit_general(dim_dict=configs[i])
        metadata = torch.load(metadata_paths[-1-i], map_location=device)

        
        start_time = time.perf_counter()
        for _ in range(50):
            big_model = create_vit_general(dim_dict=configs[i+1]).to(device)
            big_model = update_vit_weights_global(model=big_model, pruned_indices_list=metadata["pruned_index"], non_pruned_indices_list=metadata["non_pruned_index"],
                                                    pruned_weights_dict=metadata["weights"], non_pruned_weights_dict=core_model.state_dict(), device=device)
            
        end_time = time.perf_counter()
        print(f"✅ Model initialized in {(end_time - start_time):.4f} seconds.")

    print("------------------------------------STATE DICTS--------------------------------------")
    for state_dict_path in paths:
        print(f"Loading model from: {state_dict_path}")
        start_time = time.perf_counter()
        for _ in range(50):
            model = load_vit_model(state_dict_path, device=device)
        end_time = time.perf_counter()
        print(f"✅ Model initialized in {(end_time - start_time):.4f} seconds.")
    






