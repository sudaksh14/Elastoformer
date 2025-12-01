import torch
import torch.nn as nn
import time
from utils.prune_utils import get_vit_info, create_vit_general
import json
import pickle

def merge_and_remap_indices(pruned, non_pruned):
    # Flatten tensors
    pruned = pruned.flatten()
    non_pruned = non_pruned.flatten()

    # Merge and sort unique indices
    merged = torch.unique(torch.cat((pruned, non_pruned))).sort().values

    # Remap *consecutively*: 0, 1, 2, ..., len(merged)-1
    remap = {int(old): new for new, old in enumerate(merged)}

    # Apply remapping
    pruned_mapped = torch.tensor([remap[int(idx)] for idx in pruned], dtype=torch.long)
    non_pruned_mapped = torch.tensor([remap[int(idx)] for idx in non_pruned], dtype=torch.long)

    # Return the new merged list, which is simply 0..len(merged)-1
    merged_new = torch.arange(len(merged), dtype=torch.long)

    return pruned_mapped, non_pruned_mapped, merged_new

def load_vit_model(state_dict_path=None, device='cuda'):
    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=False)
        
    model_info = get_vit_info(non_pruned_weights=state_dict, core_model=True)
    model = create_vit_general(dim_dict=model_info)
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

def update_vit_weights_from_checkpoint(
    model,
    checkpoint,
    include_bias=True,
    device="cpu",
):
    """
    Loads pruned/non-pruned weights from checkpoint and updates the ViT model.

    Args:
        model (nn.Module): ViT model instance.
        checkpoint (dict): Loaded checkpoint containing:
            - pruned_weights
            - non_pruned_weights
            - pruned_index_in/out
            - non_pruned_index_in/out
        include_bias (bool): Whether to update biases.
        device (str/torch.device): Device to load weights on.
    """

    # ---- Extract metadata ----
    pruned_weights_dict = checkpoint["pruned_weights"]
    non_pruned_weights_dict = checkpoint["non_pruned_weights"]
    
    print(non_pruned_weights_dict.keys())

    pruned_in = checkpoint["pruned_index_in"]
    pruned_out = checkpoint["pruned_index_out"]

    non_pruned_in = checkpoint["non_pruned_index_in"]
    non_pruned_out = checkpoint["non_pruned_index_out"]

    model = model.to(device)

    # Dictionaries for bookkeeping
    pruned_in_mapped, pruned_out_mapped = {}, {}
    non_pruned_in_mapped, non_pruned_out_mapped = {}, {}

    # ---- Iterate over model layers ----
    for layer_name, layer in model.named_modules():

        if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            continue

        if layer_name not in pruned_out and layer_name not in pruned_in:
            continue

        # Ensure weight exists
        if not hasattr(layer, "weight"):
            continue

        weight = layer.weight.data

        # Safely get pruned and non-pruned weight dicts
        pW = pruned_weights_dict.get(layer_name, {})
        nW = non_pruned_weights_dict.get(layer_name, {})

        # ******* CLASSIFIER SPECIAL CASE *******
        if layer_name == "classifier":
            pruned_dim_in, non_pruned_dim_in, _ = merge_and_remap_indices(
                pruned_in[layer_name], non_pruned_in[layer_name]
            )

            weight[:, pruned_dim_in] = pW["Weight"].to(device)
            weight[:, non_pruned_dim_in] = nW["Weight"].to(device)

            pruned_in_mapped[layer_name] = pruned_dim_in
            non_pruned_in_mapped[layer_name] = non_pruned_dim_in
            continue

        # ******* LINEAR LAYER *******
        if isinstance(layer, nn.Linear):

            pruned_out_idx, non_pruned_out_idx, _ = merge_and_remap_indices(
                pruned_out[layer_name], non_pruned_out[layer_name]
            )
            pruned_in_idx, non_pruned_in_idx, _ = merge_and_remap_indices(
                pruned_in[layer_name], non_pruned_in[layer_name]
            )

            # ---- Update pruned weights ----
            if len(pruned_out_idx) > 0 and len(pruned_in_idx) > 0:
                row_idx, col_idx = torch.meshgrid(
                    torch.tensor(pruned_out_idx, device=device),
                    torch.tensor(pruned_in_idx, device=device),
                    indexing="ij"
                )
                weight[row_idx, col_idx] = pW["Weight"].to(device)

            # ---- Update non-pruned weights ----
            if len(non_pruned_out_idx) > 0 and len(non_pruned_in_idx) > 0:
                row_idx, col_idx = torch.meshgrid(
                    torch.tensor(non_pruned_out_idx, device=device),
                    torch.tensor(non_pruned_in_idx, device=device),
                    indexing="ij"
                )
                weight[row_idx, col_idx] = nW["Weight"].to(device)

            pruned_out_mapped[layer_name] = pruned_out_idx
            non_pruned_out_mapped[layer_name] = non_pruned_out_idx
            pruned_in_mapped[layer_name] = pruned_in_idx
            non_pruned_in_mapped[layer_name] = non_pruned_in_idx

        # ******* LAYERNORM / CONV2D *******
        elif isinstance(layer, (nn.LayerNorm, nn.Conv2d)):
            print(layer_name)
            print(pruned_out[layer_name].shape)
            print(non_pruned_out[layer_name].shape)
            print(pW["Weight"].shape)
            print(nW["Weight"].shape)

            pruned_out_idx, non_pruned_out_idx, merged = merge_and_remap_indices(
                pruned_out[layer_name], non_pruned_out[layer_name]
            )
            print(pruned_out_idx.shape)
            print(non_pruned_out_idx.shape)
            print(merged.shape)

            weight[torch.tensor(pruned_out_idx)] = pW["Weight"].to(device)
            weight[torch.tensor(non_pruned_out_idx)] = nW["Weight"].to(device)

            pruned_out_mapped[layer_name] = pruned_out_idx
            non_pruned_out_mapped[layer_name] = non_pruned_out_idx

        # ---- Handle bias ----
        if include_bias and layer.bias is not None:

            if layer_name == "classifier":
                layer.bias.data = nW["Bias"].to(device)
            else:
                bias = layer.bias.data
                pruned_out_idx, non_pruned_out_idx, merged = merge_and_remap_indices(
                    pruned_out[layer_name], non_pruned_out[layer_name]
                )
                if layer_name in pruned_out:
                    bias[torch.tensor(pruned_out_idx)] = pW["Bias"].to(device)
                if layer_name in non_pruned_out:
                    bias[torch.tensor(non_pruned_out_idx)] = nW["Bias"].to(device)

    return model

def update_vit_weights(model,
    core_model,
    checkpoint,
    include_bias=True,
    device="cpu",
):
    """
    Loads pruned/non-pruned weights from checkpoint and updates the ViT model.

    Args:
        model (nn.Module): ViT model instance.
        checkpoint (dict): Loaded checkpoint containing:
            - pruned_weights
            - non_pruned_weights
            - pruned_index_in/out
            - non_pruned_index_in/out
        include_bias (bool): Whether to update biases.
        device (str/torch.device): Device to load weights on.
    """

    # ---- Extract metadata ----
    pruned_weights_dict = checkpoint["pruned_weights"]
    # non_pruned_weights_dict = checkpoint["non_pruned_weights"]
    non_pruned_weights_dict = core_model.state_dict()
    
    # Convert keys to sets
    keys1 = set(non_pruned_weights_dict.keys())
    keys2 = set(checkpoint["non_pruned_weights"].keys())

    # Keys in non_pruned_weights_dict but not in checkpoint
    only_in_dict1 = keys1 - keys2
    # Keys in checkpoint but not in non_pruned_weights_dict
    only_in_dict2 = keys2 - keys1

    print("Keys only in non_pruned_weights_dict:", only_in_dict1)
    print("Keys only in checkpoint['non_pruned_weights']:", only_in_dict2)

    pruned_in = checkpoint["pruned_index_in"]
    pruned_out = checkpoint["pruned_index_out"]

    non_pruned_in = checkpoint["non_pruned_index_in"]
    non_pruned_out = checkpoint["non_pruned_index_out"]

    model = model.to(device)

    # Dictionaries for bookkeeping
    pruned_in_mapped, pruned_out_mapped = {}, {}
    non_pruned_in_mapped, non_pruned_out_mapped = {}, {}

    # ---- Iterate over model layers ----
    for layer_name, layer in model.named_modules():

        if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            continue

        if layer_name not in pruned_out and layer_name not in pruned_in:
            continue

        # Ensure weight exists
        if not hasattr(layer, "weight"):
            continue

        weight = layer.weight.data

        # Safely get pruned and non-pruned weight dicts
        pW = pruned_weights_dict.get(layer_name, {})
        nW = non_pruned_weights_dict.get(layer_name, {})

        # ******* CLASSIFIER SPECIAL CASE *******
        if layer_name == "classifier":
            pruned_dim_in, non_pruned_dim_in, _ = merge_and_remap_indices(
                pruned_in[layer_name], non_pruned_in[layer_name]
            )

            weight[:, pruned_dim_in] = pW["Weight"].to(device)
            weight[:, non_pruned_dim_in] = nW["Weight"].to(device)

            pruned_in_mapped[layer_name] = pruned_dim_in
            non_pruned_in_mapped[layer_name] = non_pruned_dim_in
            continue

        # ******* LINEAR LAYER *******
        if isinstance(layer, nn.Linear):

            pruned_out_idx, non_pruned_out_idx, _ = merge_and_remap_indices(
                pruned_out[layer_name], non_pruned_out[layer_name]
            )
            pruned_in_idx, non_pruned_in_idx, _ = merge_and_remap_indices(
                pruned_in[layer_name], non_pruned_in[layer_name]
            )

            # ---- Update pruned weights ----
            if len(pruned_out_idx) > 0 and len(pruned_in_idx) > 0:
                row_idx, col_idx = torch.meshgrid(
                    torch.tensor(pruned_out_idx, device=device),
                    torch.tensor(pruned_in_idx, device=device),
                    indexing="ij"
                )
                weight[row_idx, col_idx] = pW["Weight"].to(device)

            # ---- Update non-pruned weights ----
            if len(non_pruned_out_idx) > 0 and len(non_pruned_in_idx) > 0:
                row_idx, col_idx = torch.meshgrid(
                    torch.tensor(non_pruned_out_idx, device=device),
                    torch.tensor(non_pruned_in_idx, device=device),
                    indexing="ij"
                )
                weight[row_idx, col_idx] = nW["Weight"].to(device)

            pruned_out_mapped[layer_name] = pruned_out_idx
            non_pruned_out_mapped[layer_name] = non_pruned_out_idx
            pruned_in_mapped[layer_name] = pruned_in_idx
            non_pruned_in_mapped[layer_name] = non_pruned_in_idx

        # ******* LAYERNORM / CONV2D *******
        elif isinstance(layer, (nn.LayerNorm, nn.Conv2d)):

            pruned_out_idx, non_pruned_out_idx, _ = merge_and_remap_indices(
                pruned_out[layer_name], non_pruned_out[layer_name]
            )

            print(pruned_out_idx.shape)
            print(non_pruned_out_idx.shape)
            print(pW["Weight"].shape)
            weight[torch.tensor(pruned_out_idx)] = pW["Weight"].to(device)
            weight[torch.tensor(non_pruned_out_idx)] = nW["Weight"].to(device)

            pruned_out_mapped[layer_name] = pruned_out_idx
            non_pruned_out_mapped[layer_name] = non_pruned_out_idx

        # ---- Handle bias ----
        if include_bias and layer.bias is not None:

            if layer_name == "classifier":
                layer.bias.data = nW["Bias"].to(device)
            else:
                bias = layer.bias.data
                if layer_name in pruned_out:
                    bias[torch.tensor(pruned_out[layer_name])] = pW["Bias"].to(device)
                if layer_name in non_pruned_out:
                    bias[torch.tensor(non_pruned_out[layer_name])] = nW["Bias"].to(device)

    return model


if __name__ == "__main__":
    
    iter = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_Switching.pth"] + \
            [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_Switching.pth" for i in range(2,7)]
            
    # ---------------------TESTING CONFIGS AND LOADING-----------------------
    # for i, state_dict_path in enumerate(paths):
    #     print(f"Loading model from: {state_dict_path}")
    #     model = load_vit_model(state_dict_path, device=device)
    #     # torch.save(model, f"./saves/state_dicts/Elastoformer_Level_{i+1}.pth")
    #     print(f"✅ Model saved as Level-{i+1}")
    
    # configs = []
    # for sd in paths:
        # configs.append(get_vit_info(non_pruned_weights = torch.load(sd, map_location=device), core_model=True))
        
    # json_path = "./saves/Switching_configs.json"
    # pkl_path = "./saves/Switching_configs.pkl"
    
    # with open(json_path, "w") as f:
    #     json.dump(configs, f)
        
    # with open(pkl_path, "wb") as f:
    #     pickle.dump(configs, f)
        
    # exit()

    # metadata_paths = [f"./saves/pruning_metadata/Switching_pruning_metadata_Level_{i}.pth" for i in range(2,7)]
    
    with open("./saves/Switching_configs.pkl", "rb") as f:
        configs = pickle.load(f)

    print("------------------------------------ELASTOFORMER--------------------------------------")
    for i in range(1,6):
        print(f"Initializing model for level {i+1}...")
        print(f"Config: {configs[i]}")
        
        # metadata = torch.load(metadata_paths[i+1], map_location="cpu", weights_only=False)
        metadata = torch.load(f"./saves/pruning_metadata/Switching_pruning_metadata_Level_{i+1}.pth", map_location="cpu", weights_only=False)
        
        # core_model_config = create_vit_general(dim_dict=configs[0])
        # print(core_model_config.config.pruned_dim)
        
        core_model = torch.load("./saves/pruning_metadata/core_model.pt", map_location="cpu", weights_only=False)
        # print(core_model.config.pruned_dim)
        # print(get_vit_info(non_pruned_weights=core_model.state_dict(), core_model=True))
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(iter):
            # big_model = create_vit_general(dim_dict=configs[i+1]).to(device)
            big_model = create_vit_general(dim_dict=configs[i]).to("cpu")
            big_model = update_vit_weights_from_checkpoint(big_model, metadata, device="cpu")
            # big_model = update_vit_weights(big_model, core_model, metadata, device="cpu")
            big_model.eval()
        
        torch.cuda.synchronize()    
        end_time = time.perf_counter()
        print(f"✅ Model initialized in {(end_time - start_time)/iter:.4f} seconds.")

    print("------------------------------------INDEPENDENT MODELS--------------------------------------")
    model_paths = [f"/var/scratch/skalra/elastoformer_saves/state_dicts/Elastoformer_Level_{i}.pth" for i in range(1,7)]
            
    for path in model_paths:
        print(f"Loading model from: {path}")
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(iter):
            model = torch.load(path, weights_only=False).to(device)
            model.eval()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        print(f"✅ Model initialized in {(end_time - start_time)/iter:.4f} seconds.")
    
    






