import torch 
import torch.nn as nn 
from layerwrapper import WrappedLayer 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import numpy as np
from partial_freezing import freeze_linear_params, freeze_conv2d_params, freeze_layernorm_params


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    subset = find_layers(model, layers=[nn.Linear])
    zero_cnt = 0
    fc_params = 0
    for name in subset:
        W = subset[name].weight.data
        if W.shape[0] == 1000:
            continue 
        zero_cnt += (W==0).sum().item()
        fc_params += W.numel()
    return float(zero_cnt) / fc_params

def compute_mask(W_metric, prune_granularity, sparsity):
    if prune_granularity == "layer":
        thres = torch.sort(W_metric.flatten().cuda())[0][int(W_metric.numel() * sparsity)].cpu()
        W_mask = (W_metric <= thres)
        return W_mask 
    elif prune_granularity == "row":
        W_mask = (torch.zeros_like(W_metric)==1)
        sort_res = torch.sort(W_metric, dim=-1, stable=True)

        indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity)]
        W_mask.scatter_(1, indices, True)
        return W_mask 
    
def prune_magnitude(args, model, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.blocks 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_deit(args, model, calib_data, device):
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda"])

    metric_stats = []
    for blk in model.blocks:
        subset = find_layers(blk)
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    thresh = None 
    #####################################
    inps = model.patch_embed(inps)

    cls_tokens = model.cls_token.expand(bs, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    dist_token = model.dist_token.expand(bs, -1, -1)
    inps = torch.cat((cls_tokens, dist_token, inps), dim=1)

    inps = inps + model.pos_embed
    inps = model.pos_drop(inps)

    for block_id, blk in enumerate(model.blocks):
        subset = find_layers(blk)

        if require_forward:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            if bs > 256:
                tmp_res = []
                for i1 in range(0, bs, 256):
                    j1 = min(i1+256, bs)
                    tmp_res.append(blk(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = blk(inps)

            for h in handles:
                h.remove()

        ################# pruning ###################
        for name in subset:
            if args.prune_metric == "wanda":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, args.sparsity)

            subset[name].weight.data[W_mask] = 0

def prune_vit(args, model, calib_data, device):
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda"])

    metric_stats = []
    for blk in model.blocks:
        subset = find_layers(blk)
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    thresh = None 
    #####################################
    inps = model.patch_embed(inps)

    cls_tokens = model.cls_token.expand(bs, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    inps = torch.cat((cls_tokens, inps), dim=1)
    inps = inps + model.pos_embed
    inps = model.pos_drop(inps)

    for block_id, blk in enumerate(model.blocks):
        print(f"block {block_id}")
        subset = find_layers(blk)

        if require_forward:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            if bs > 256:
                tmp_res = []
                for i1 in range(0, bs, 256):
                    j1 = min(i1+256, bs)
                    tmp_res.append(blk(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = blk(inps)

            for h in handles:
                h.remove()

        ################# pruning ###################
        for name in subset:
            if args.prune_metric == "wanda":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, args.sparsity)

            subset[name].weight.data[W_mask] = 0
        ##############################################

def prune_convnext(args, model, calib_data, device):
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda"])

    ##############################################################
    metric_stats = []
    for block_id in range(4):
        subset = find_layers(model.stages[block_id])
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)
    ##############################################################

    thresh = None 
    for block_id in range(4):
        print(f"block {block_id}")
        subset = find_layers(model.stages[block_id])

        if require_forward:
            layer = model.downsample_layers[block_id]
            if bs > 1024:
                tmp_res = []
                for i1 in range(0, bs, 512):
                    j1 = min(i1+512, bs)
                    tmp_res.append(layer(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = layer(inps)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
               handles.append(subset[name].register_forward_hook(add_batch(name)))
            layer = model.stages[block_id]
            if bs > 1024:
                tmp_res = []
                for i1 in range(0, bs, 512):
                    j1 = min(i1+512, bs)
                    tmp_res.append(layer(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = layer(inps)
            for h in handles:
                h.remove()

        ################# pruning ###################
        for name in subset:
            if args.prune_metric == "wanda":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, args.sparsity)

            subset[name].weight.data[W_mask] = 0
        ##############################################

##########################################################################################
def extract_layer_index(layer_name):
    """Extracts the numeric index from a layer name string."""
    match = re.search(r'\d+', layer_name)  # Find first number in the string
    return int(match.group()) if match else float('inf')  # Default to large number if no match


def visualize_weight_matrix(vit_model, layer_name, plot_name):
    """
    Visualizes the weight matrix of a given linear layer in a Vision Transformer (ViT) model.
    
    Parameters:
    vit_model (torch.nn.Module): The Vision Transformer model.
    layer_name (str): The name of the linear layer to visualize.
    """
    # Extract the specified layer's weights
    weight_matrix = dict(vit_model.named_parameters()).get(layer_name, None)

    if weight_matrix is None:
        print(f"Layer '{layer_name}' not found in the model.")
        return
    
    weight_matrix = weight_matrix.detach().cpu().numpy()
    
    # Plot the weight matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(weight_matrix, cmap='coolwarm', center=0)
    plt.title(f'Weight Matrix of {layer_name}')
    plt.xlabel('Input Features')
    plt.ylabel('Output Features')
    plt.savefig(f'weight_matrix_{layer_name}_{plot_name}.png')
    
def get_pruned_indices(vit_model):
    pruned_indices = {}
    
    for name, param in vit_model.named_parameters():
        if 'weight' in name and param.requires_grad:  # Focus on trainable weight matrices
            zero_indices = (param == 0).nonzero(as_tuple=True)  # Find zero-value indices
            pruned_indices[name] = list(zip(*zero_indices))  # Convert to list of index tuples
    
    return pruned_indices

def get_hidden_dim(model):
    vit_hidden_info = {}

    print(model.vit.embeddings.patch_embeddings.projection.weight.shape)

    # Loop through each transformer block in ViT
    for layer_idx, block in enumerate(model.vit.encoder.layer):
        layer_name = f"Layer_{layer_idx}"

        # Extract QKV weights (ViT uses a single weight matrix for Q, K, and V)
        hidden_dim = block.attention.attention.key.weight.shape[1]

        # print(block.attention.attention.key.weight.shape)
        print(block.attention.attention.query.weight.shape)
        
        # Split into individual Q, K, V weights
        # q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3, dim=0)

        attn_proj_dim = block.attention.output.dense.weight.shape[0]
        print(block.attention.output.dense.weight.shape)

        # Extract Feed-Forward (FFN) hidden dim
        ffn_hidden_dim = block.intermediate.dense.weight.shape[0]
        print(block.intermediate.dense.weight.shape)
        print(block.output.dense.weight.shape)

        # Extract LayerNorm hidden dimensions
        norm1_dim = block.layernorm_before.weight.shape[0]
        norm2_dim = block.layernorm_after.weight.shape[0]


        # Store in dictionary
        vit_hidden_info[layer_name] = {
            "attn_hidden_dim": hidden_dim,
            "FFN_hidden_Dim": ffn_hidden_dim,
            "LayerNorm1_Dim": norm1_dim,
            "LayerNorm2_Dim": norm2_dim, 
        }

    return vit_hidden_info

def get_layer_dimensions(state_dict):
    """
    Extract hidden dimensions of QKV, FFN, LayerNorm, embeddings, and classifier layers
    from a ViT model's state_dict, including pruned dimensions if provided.

    Args:
        state_dict (dict): The state_dict of a Vision Transformer model.
        pruning_info (dict, optional): A dictionary containing pruned indices for the layers
                                       (dim 0 and dim 1). If None, the function assumes no pruning.

    Returns:
        dict: Dictionary containing hidden dimensions for each transformer layer and model components.
    """
    vit_hidden_info = {}

    # Identify transformer block layers dynamically
    layer_indices = set()
    for key in state_dict.keys():
        parts = key.split(".")
        if "encoder" in parts and any(part.isdigit() for part in parts):
            layer_indices.add(int(parts[parts.index("encoder") + 2]))
        
        if "embeddings" in parts:
            # Embeddings dimension (e.g., input embedding)
            embedding_dim_key = "vit.embeddings.patch_embeddings.projection.weight"
            vit_hidden_info["embedding_dim"] = state_dict[embedding_dim_key].shape[0] if embedding_dim_key in state_dict else None

        if "layernorm" in parts:
            LN_key = "vit.layernorm.weight"
            vit_hidden_info["LN_dim"] = state_dict[LN_key].shape[0] if LN_key in state_dict else None
        
        if "classifier" in parts:
            # Classifier dimension (e.g., the final linear layer for classification)
            classifier_dim_key = "classifier.weight"
            vit_hidden_info["classifier_dim"] = state_dict[classifier_dim_key].shape[1] if classifier_dim_key in state_dict else None

    for layer_idx in sorted(layer_indices):
        layer_name = f"Encoder_Layer_{layer_idx}"
        
        try:
            # QKV dimensions (shared weight for Q, K, V)
            qkv_weight_key = f"vit.encoder.layer.{layer_idx}.attention.attention.key.weight"
            qkv_bias_key = f"vit.encoder.layer.{layer_idx}.attention.attention.key.bias"
            attn_hidden_dim0 = state_dict[qkv_weight_key].shape[0] if qkv_weight_key in state_dict else None
            attn_hidden_dim1 = state_dict[qkv_weight_key].shape[1] if qkv_weight_key in state_dict else None

            # Projection hidden dimension
            proj_weight1_key = f"vit.encoder.layer.{layer_idx}.attention.output.dense.weight"
            proj_hidden_dim0 = state_dict[proj_weight1_key].shape[0] if proj_weight1_key in state_dict else None
            proj_hidden_dim1 = state_dict[proj_weight1_key].shape[1] if proj_weight1_key in state_dict else None

            # FFN1 hidden dimension
            ffn1_weight_key = f"vit.encoder.layer.{layer_idx}.intermediate.dense.weight"
            ffn_inter_dim0 = state_dict[ffn1_weight_key].shape[0] if ffn1_weight_key in state_dict else None
            ffn_inter_dim1 = state_dict[ffn1_weight_key].shape[1] if ffn1_weight_key in state_dict else None

            # FFN2 hidden dimension
            ffn2_weight_key = f"vit.encoder.layer.{layer_idx}.output.dense.weight"
            ffn_out_dim0 = state_dict[ffn2_weight_key].shape[0] if ffn2_weight_key in state_dict else None
            ffn_out_dim1 = state_dict[ffn2_weight_key].shape[1] if ffn1_weight_key in state_dict else None

            # LayerNorm dimensions
            norm1_key = f"vit.encoder.layer.{layer_idx}.layernorm_before.weight"
            norm2_key = f"vit.encoder.layer.{layer_idx}.layernorm_after.weight"
            norm1_dim = state_dict[norm1_key].shape[0] if norm1_key in state_dict else None
            norm2_dim = state_dict[norm2_key].shape[0] if norm2_key in state_dict else None

            vit_hidden_info[layer_name] = {
                "attn_layers": (attn_hidden_dim0, attn_hidden_dim1),
                "Proj_layer": (proj_hidden_dim0, proj_hidden_dim1),
                "FF1_layer": (ffn_inter_dim0, ffn_inter_dim1),
                "FF2_layer": (ffn_out_dim0, ffn_out_dim1),
                "LayerNorm1_Dim": norm1_dim,
                "LayerNorm2_Dim": norm2_dim,
            }

        except KeyError as e:
            print(f"Warning: Missing expected key {e} in state_dict for layer {layer_idx}")

    return vit_hidden_info


def get_layer_size(state_dict):
    vit_hidden_info = {}
    
    # q_weight_shape = state_dict["vit.encoder.layer.0.attention.attention.query.weight"]
    # num_heads = q_weight_shape.shape[0] // q_weight_shape.shape[1]
    # vit_hidden_info["num_heads"] = num_heads

    # Identify transformer block layers dynamically
    layer_indices = set()
    for key in state_dict.keys():
        parts = key.split(".")
        if "encoder" in parts and any(part.isdigit() for part in parts):
            layer_indices.add(int(parts[parts.index("encoder") + 2]))

            for layer_idx in sorted(layer_indices):
                # Attention Weight dimension
                q_weight_key = f"vit.encoder.layer.{layer_idx}.attention.attention.query.weight"
                attn_hidden_dim0 = state_dict[q_weight_key].shape[0] if q_weight_key in state_dict else None
                attn_hidden_dim1 = state_dict[q_weight_key].shape[1] if q_weight_key in state_dict else None
                vit_hidden_info[q_weight_key] = (attn_hidden_dim0, attn_hidden_dim1)

                k_weight_key = f"vit.encoder.layer.{layer_idx}.attention.attention.key.weight"
                attn_hidden_dim0 = state_dict[k_weight_key].shape[0] if k_weight_key in state_dict else None
                attn_hidden_dim1 = state_dict[k_weight_key].shape[1] if k_weight_key in state_dict else None
                vit_hidden_info[k_weight_key] = (attn_hidden_dim0, attn_hidden_dim1)

                v_weight_key = f"vit.encoder.layer.{layer_idx}.attention.attention.value.weight"
                attn_hidden_dim0 = state_dict[v_weight_key].shape[0] if v_weight_key in state_dict else None
                attn_hidden_dim1 = state_dict[v_weight_key].shape[1] if v_weight_key in state_dict else None
                vit_hidden_info[v_weight_key] = (attn_hidden_dim0, attn_hidden_dim1)

                # Projection hidden dimension
                proj_weight_key = f"vit.encoder.layer.{layer_idx}.attention.output.dense.weight"
                proj_hidden_dim0 = state_dict[proj_weight_key].shape[0] if proj_weight_key in state_dict else None
                proj_hidden_dim1 = state_dict[proj_weight_key].shape[1] if proj_weight_key in state_dict else None
                vit_hidden_info[proj_weight_key] = (proj_hidden_dim0, proj_hidden_dim1)

                # FFN1 hidden dimension
                ffn1_weight_key = f"vit.encoder.layer.{layer_idx}.intermediate.dense.weight"
                ffn_inter_dim0 = state_dict[ffn1_weight_key].shape[0] if ffn1_weight_key in state_dict else None
                ffn_inter_dim1 = state_dict[ffn1_weight_key].shape[1] if ffn1_weight_key in state_dict else None
                vit_hidden_info[ffn1_weight_key] = (ffn_inter_dim0, ffn_inter_dim1)

                # FFN2 hidden dimension
                ffn2_weight_key = f"vit.encoder.layer.{layer_idx}.output.dense.weight"
                ffn_out_dim0 = state_dict[ffn2_weight_key].shape[0] if ffn2_weight_key in state_dict else None
                ffn_out_dim1 = state_dict[ffn2_weight_key].shape[1] if ffn1_weight_key in state_dict else None
                vit_hidden_info[ffn2_weight_key] = (ffn_out_dim0, ffn_out_dim1)

                # LayerNorm dimensions
                norm1_key = f"vit.encoder.layer.{layer_idx}.layernorm_before.weight"
                norm2_key = f"vit.encoder.layer.{layer_idx}.layernorm_after.weight"
                norm1_dim = state_dict[norm1_key].shape[0] if norm1_key in state_dict else None
                norm2_dim = state_dict[norm2_key].shape[0] if norm2_key in state_dict else None
                vit_hidden_info[norm1_key] = norm1_dim
                vit_hidden_info[norm2_key] = norm2_dim

        
        elif "embeddings" in parts:
            # Embeddings dimension (e.g., input embedding)
            embedding_dim_key = "vit.embeddings.patch_embeddings.projection.weight"
            vit_hidden_info[embedding_dim_key] = state_dict[embedding_dim_key].shape[0] if embedding_dim_key in state_dict else None

        elif "layernorm" in parts:
            LN_key = "vit.layernorm.weight"
            vit_hidden_info[LN_key] = state_dict[LN_key].shape[0] if LN_key in state_dict else None
        
        elif "classifier" in parts:
            # Classifier dimension (e.g., the final linear layer for classification)
            classifier_dim_key = "classifier.weight"
            vit_hidden_info[classifier_dim_key] = state_dict[classifier_dim_key].shape[1] if classifier_dim_key in state_dict else None

    return vit_hidden_info

def extract_vit_weight_subset(model, out_indices_dict, in_indices_dict):
    """
    Extracts weight subsets from a Vision Transformer model based on specified output and input indices.

    Args:
        model (torch.nn.Module): The ViT model.
        out_indices_dict (dict): Dictionary with layer names as keys and lists of output indices as values.
        in_indices_dict (dict): Dictionary with layer names as keys and lists of input indices as values.

    Returns:
        dict: Dictionary storing the selected weights for each layer.
    """
    selected_weights = {}

    for layer_name, layer in model.named_modules():

        if "classifier" in layer_name:
            weight = layer.weight  # Shape: (out_dim, in_dim)
            bias = layer.bias if layer.bias is not None else None

            in_indices = torch.tensor(in_indices_dict[layer_name], dtype=torch.long)
            selected_weight = weight[:, in_indices]

            # Store subset
            selected_weights[layer_name] = {
                "Weight": selected_weight.detach().cpu().numpy(),
                "Bias": bias.detach().cpu().numpy() if bias is not None else None
            }

        elif isinstance(layer, torch.nn.Conv2d):
            weight = layer.weight
            bias = layer.bias if layer.bias is not None else None

            out_indices = torch.tensor(out_indices_dict[layer_name], dtype=torch.long)

            # Store subset
            selected_weights[layer_name] = {
                "Weight": weight[out_indices].detach().cpu().numpy(),
                "Bias": bias[out_indices].detach().cpu().numpy() if bias is not None else None
            }

        elif isinstance(layer, torch.nn.Linear):
            weight = layer.weight  # Shape: (out_dim, in_dim)
            bias = layer.bias if layer.bias is not None else None

            # Get indices
            out_indices = torch.tensor(out_indices_dict[layer_name], dtype=torch.long)
            in_indices = torch.tensor(in_indices_dict[layer_name], dtype=torch.long)

            # Select subset of weights
            selected_weight = weight[out_indices][:, in_indices]

            # Store results
            selected_weights[layer_name] = {
                "Weight": selected_weight.detach().cpu().numpy(),
                "Bias": bias[out_indices].detach().cpu().numpy() if bias is not None else None
            }

        elif isinstance(layer, torch.nn.LayerNorm):
            weight = layer.weight  # Shape: (dim,)
            bias = layer.bias  # Shape: (dim,)

            out_indices = torch.tensor(out_indices_dict[layer_name], dtype=torch.long)

            # Store subset
            selected_weights[layer_name] = {
                "Weight": weight[out_indices].detach().cpu().numpy(),
                "Bias": bias[out_indices].detach().cpu().numpy() if bias is not None else None
            }

    print("Pruned weights extracted successfully")

    return selected_weights
            

def extract_vit_core_weights(model):
    """
    Extracts weights from the core Vision Transformer model based on specified output and input indices.

    Args:
        model (torch.nn.Module): The ViT model.
        out_indices_dict (dict): Dictionary with layer names as keys and lists of output indices as values.
        in_indices_dict (dict): Dictionary with layer names as keys and lists of input indices as values.

    Returns:
        dict: Dictionary storing the selected weights for each layer.
    """
    selected_weights = {}

    for layer_name, layer in model.named_modules():

        if "classifier" in layer_name:
            weight = layer.weight  # Shape: (out_dim, in_dim)
            bias = layer.bias if layer.bias is not None else None

            selected_weight = weight

            # Store subset
            selected_weights[layer_name] = {
                "Weight": selected_weight.detach().cpu().numpy(),
                "Bias": bias.detach().cpu().numpy() if bias is not None else None
            }

        elif isinstance(layer, torch.nn.Conv2d):
            weight = layer.weight
            bias = layer.bias if layer.bias is not None else None

            # Store subset
            selected_weights[layer_name] = {
                "Weight": weight.detach().cpu().numpy(),
                "Bias": bias.detach().cpu().numpy() if bias is not None else None
            }

        elif isinstance(layer, torch.nn.Linear):
            weight = layer.weight  # Shape: (out_dim, in_dim)
            bias = layer.bias if layer.bias is not None else None

            # Select subset of weights
            selected_weight = weight

            # Store results
            selected_weights[layer_name] = {
                "Weight": selected_weight.detach().cpu().numpy(),
                "Bias": bias.detach().cpu().numpy() if bias is not None else None
            }

        elif isinstance(layer, torch.nn.LayerNorm):
            weight = layer.weight  # Shape: (dim,)
            bias = layer.bias  # Shape: (dim,)

            selected_weights[layer_name] = {
                "Weight": weight.detach().cpu().numpy(),
                "Bias": bias.detach().cpu().numpy() if bias is not None else None
            }

    print("Core weights extracted successfully")

    return selected_weights

def get_unpruned_indices(total_idxs, pruned_idxs, dim="out"):

    if isinstance(total_idxs, int):
        tensor1 = torch.arange(total_idxs)
    else:
        if dim == "out":
            tensor1 = torch.arange(total_idxs[0])
        elif dim == "in":
            tensor1 = torch.arange(total_idxs[1])
        else:
            raise ValueError("Incorrect Dimension")
    tensor2 = torch.tensor(pruned_idxs)

    unpruned_idxs = tensor1[~torch.isin(tensor1, tensor2)]

    return unpruned_idxs.tolist()


def update_vit_weights(model, pruned_indices_list, non_pruned_indices_list, pruned_weights_dict, non_pruned_weights_dict, include_bias=True):
    """
    Updates the weights of a Vision Transformer (ViT) model based on given indices and corresponding weights.
    
    Args:
        model (torch.nn.Module): The ViT model.
        pruned_indices_dict (dict): Dictionary with layer names as keys and pruned weight indices as values.
        non_pruned_indices_dict (dict): Dictionary with layer names as keys and non-pruned weight indices as values.
        pruned_weights_dict (dict): Dictionary with layer names as keys and pruned weights as values.
        non_pruned_weights_dict (dict): Dictionary with layer names as keys and non-pruned weights as values.
    """
    
    pruned_in = pruned_indices_list[0]
    pruned_out = pruned_indices_list[1]

    non_pruned_in = non_pruned_indices_list[0]
    non_pruned_out = non_pruned_indices_list[1]

    for layer_name, layer in model.named_modules():

        if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            continue
        else:
            if layer_name not in (pruned_out.keys() | pruned_in.keys()):
                print(f"Warning: Layer '{layer_name}' not found in the Pruning Metadata")
                continue

        # print("Layer Found: ", layer_name, layer)
        
        # Ensure the layer has weights
        if not hasattr(layer, 'weight'):
            print(f"Warning: Layer '{layer_name}' does not have 'weight' attribute.")
            continue
        
        weight = layer.weight.data
        
        # Get corresponding weights
        pruned_weights = pruned_weights_dict[layer_name]
        non_pruned_weights = non_pruned_weights_dict[layer_name]

        if layer_name == 'classifier':
            pruned_dim1 = pruned_in[layer_name]
            non_pruned_dim1 = non_pruned_in[layer_name]

            # Replace pruned weights
            weight[:, pruned_dim1] = torch.tensor(pruned_weights["Weight"], device=weight.device)
            # Replace non-pruned weights
            weight[:, non_pruned_dim1] = torch.tensor(non_pruned_weights["Weight"], device=weight.device)

            # print(f"Updated weights for layer: {layer_name}")


        elif isinstance(layer, nn.Linear):

            # Get pruned and unpruned indices
            pruned_dim0 = torch.tensor(pruned_out[layer_name], dtype=torch.long)  # dim 0 indices
            pruned_dim1 = torch.tensor(pruned_in[layer_name], dtype=torch.long)   # dim 1 indices

            non_pruned_dim0 = torch.tensor(non_pruned_out[layer_name], dtype=torch.long)
            non_pruned_dim1 = torch.tensor(non_pruned_in[layer_name], dtype=torch.long)
        
            # Replace pruned weights
            weight[pruned_dim0[:, None], pruned_dim1] = torch.tensor(pruned_weights["Weight"], device=weight.device)
            # Replace non-pruned weights
            weight[non_pruned_dim0[:, None], non_pruned_dim1] = torch.tensor(non_pruned_weights["Weight"], device=weight.device)

            # print(f"Updated weights for layer: {layer_name}")

        elif isinstance(layer, (nn.LayerNorm, nn.Conv2d)):

            # Get pruned and unpruned indices
            pruned_dim0 = pruned_out[layer_name]  # dim 0 indices
            non_pruned_dim0 = non_pruned_out[layer_name] # dim 0 indices

            # Replace pruned weights
            weight[pruned_dim0] = torch.tensor(pruned_weights["Weight"], device=weight.device)
            # Replace non-pruned weights
            weight[non_pruned_dim0] = torch.tensor(non_pruned_weights["Weight"], device=weight.device)

            # print(f"Updated weights for layer: {layer_name}")

        if include_bias:
            if layer.bias is not None and "Bias" in pruned_weights.keys():
                bias = layer.bias.data

                if layer_name == 'classifier':
                    bias = torch.tensor(non_pruned_weights["Bias"], device=weight.device)
                
                else:
                    # Replace pruned weights
                    bias[pruned_dim0] = torch.tensor(pruned_weights["Bias"], device=weight.device)
                    # Replace non-pruned weights
                    bias[non_pruned_dim0] = torch.tensor(non_pruned_weights["Bias"], device=weight.device)

    return model

def zero_out_gradients(model, in_indices, out_indices):

    for layer_name, layer in model.named_modules():
        
        if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            continue
        else:
            if layer_name not in (in_indices.keys() | out_indices.keys()):
                print(f"Warning: Layer '{layer_name}' not found in the Pruning Metadata")
                continue

        if layer.weight.grad is None:
            print(f"Warning: Layer '{layer_name}' does not have gradients.")
            continue

        if 'classifier' in layer_name:
            continue
        
            freeze_dim1 = in_indices[layer_name]
            # Freeze non-pruned Gradients
            with torch.no_grad():
                layer.weight.grad[:, freeze_dim1] = 0

        elif isinstance(layer, nn.Linear):
            # Get frozen indices
            freeze_dim0 = torch.tensor(out_indices[layer_name], dtype=torch.long, device=layer.weight.grad.device)  # dim 0 indices
            freeze_dim1 = torch.tensor(in_indices[layer_name], dtype=torch.long, device=layer.weight.grad.device)   # dim 1 indices

            # Freeze non-pruned Gradients
            with torch.no_grad():
                layer.weight.grad[freeze_dim0[:, None], freeze_dim1] = 0
                if layer.bias is not None:
                    layer.bias.grad[freeze_dim0] = 0

            # if layer_name == "vit.encoder.layer.0.attention.attention.query":
            #     print("Query Layer Gradients")
            #     print("Dim0:", freeze_dim0)
            #     print("Dim1:", freeze_dim1)
            #     print(layer.weight.grad[freeze_dim0[0]])
            #     print(layer.weight.grad[:, freeze_dim1[0]])
            #     # print(layer.weight.grad[freeze_dim0[:, None], freeze_dim1])
            #     # print(layer.weight.grad[:10])
            #     # print(freeze_dim0.shape, freeze_dim1.shape)
            #     freeze = torch.sum(layer.weight.grad == 0).item()
            #     print(freeze, freeze == (freeze_dim0.shape[0] * freeze_dim1.shape[0]))
            #     plot_tensor(layer.weight.grad, title="Query Layer Gradients", cmap="plasma", figsize=(10, 10))
            #     exit()



        elif isinstance(layer, (nn.LayerNorm, nn.Conv2d)):
            freeze_dim0 = out_indices[layer_name]

            # Freeze non-pruned Gradients
            with torch.no_grad():
                layer.weight.grad[freeze_dim0] = 0
                if layer.bias is not None:
                    layer.bias.grad[freeze_dim0] = 0


def zero_out_gradients_v2(model, optimizer, in_indices, out_indices):

    for layer_name, layer in model.named_modules():
        
        if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            continue
        else:
            if layer_name not in (in_indices.keys() | out_indices.keys()):
                print(f"Warning: Layer '{layer_name}' not found in the Pruning Metadata")
                continue

        if layer.weight.grad is None:
            print(f"Warning: Layer '{layer_name}' does not have gradients.")
            continue

        if 'classifier' in layer_name:
            continue

        elif isinstance(layer, nn.Linear):
            # Get frozen indices
            freeze_dim0 = torch.tensor(out_indices[layer_name], dtype=torch.long, device=layer.weight.grad.device)  # dim 0 indices
            freeze_dim1 = torch.tensor(in_indices[layer_name], dtype=torch.long, device=layer.weight.grad.device)   # dim 1 indices

            # Freeze non-pruned Gradients
            with torch.no_grad():
                layer.weight.grad[freeze_dim0[:, None], freeze_dim1] = 0
                if layer.bias is not None:
                    layer.bias.grad[freeze_dim0] = 0

                # Also clear AdamW momentum buffers for the frozen weights
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p is layer.weight:
                            state = optimizer.state[p]
                            if 'exp_avg' in state:  
                                state['exp_avg'][freeze_dim0[:, None], freeze_dim1] = 0
                            if 'exp_avg_sq' in state:
                                state['exp_avg_sq'][freeze_dim0[:, None], freeze_dim1] = 0
                        elif p is layer.bias and layer.bias is not None:
                            state = optimizer.state[p]
                            if 'exp_avg' in state:
                                state['exp_avg'][freeze_dim0] = 0
                            if 'exp_avg_sq' in state:
                                state['exp_avg_sq'][freeze_dim0] = 0


        elif isinstance(layer, (nn.LayerNorm, nn.Conv2d)):
            freeze_dim0 = out_indices[layer_name]

            # Freeze non-pruned Gradients
            with torch.no_grad():
                layer.weight.grad[freeze_dim0] = 0
                if layer.bias is not None:
                    layer.bias.grad[freeze_dim0] = 0

                # Zero AdamW buffers
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p is layer.weight:
                            state = optimizer.state[p]
                            if 'exp_avg' in state:
                                state['exp_avg'][freeze_dim0] = 0
                            if 'exp_avg_sq' in state:
                                state['exp_avg_sq'][freeze_dim0] = 0
                        elif p is layer.bias and layer.bias is not None:
                            state = optimizer.state[p]
                            if 'exp_avg' in state:
                                state['exp_avg'][freeze_dim0] = 0
                            if 'exp_avg_sq' in state:
                                state['exp_avg_sq'][freeze_dim0] = 0


# More efficient in use of memory
def zero_out_gradients_v3(model, in_indices, out_indices, device='cuda' if torch.cuda.is_available() else 'cpu'):
    for layer_name, layer in model.named_modules():
        
        if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            continue
        else:
            if layer_name not in (in_indices.keys() | out_indices.keys()):
                print(f"Warning: Layer '{layer_name}' not found in the Pruning Metadata")
                continue

        if layer.weight.grad is None:
            print(f"Warning: Layer '{layer_name}' does not have gradients.")
            continue

        if 'classifier' in layer_name:
            continue

        elif isinstance(layer, nn.Linear):
            # Get frozen indices
            freeze_dim0 = torch.tensor(out_indices[layer_name], dtype=torch.long, device=device)  # dim 0 indices
            freeze_dim1 = torch.tensor(in_indices[layer_name], dtype=torch.long, device=device)   # dim 1 indices

            # Ensure indices are within valid range
            freeze_dim0 = freeze_dim0[freeze_dim0 < layer.weight.grad.shape[0]]
            freeze_dim1 = freeze_dim1[freeze_dim1 < layer.weight.grad.shape[1]]

            # Create a grid of (dim0, dim1) combinations
            grid_dim0, grid_dim1 = torch.meshgrid(freeze_dim0, freeze_dim1, indexing='ij')

            # Zero out gradients at specified indices
            with torch.no_grad():
                layer.weight.grad[grid_dim0, grid_dim1] = 0
                if layer.bias is not None:
                        layer.bias.grad[freeze_dim0] = 0

        elif isinstance(layer, (nn.LayerNorm, nn.Conv2d)):
            freeze_dim0 = torch.tensor(out_indices[layer_name], dtype=torch.long, device=device)
            
            freeze_dim0 = freeze_dim0[freeze_dim0 < layer.weight.grad.shape[0]]

            # Freeze non-pruned Gradients
            with torch.no_grad():
                layer.weight.grad[freeze_dim0] = 0
                if layer.bias is not None:
                    layer.bias.grad[freeze_dim0] = 0


def freeze_partial_weights(model, in_indices, out_indices, device='cuda' if torch.cuda.is_available() else 'cpu'):

    for layer_name, layer in model.named_modules():
        
        if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            continue
        else:
            if layer_name not in (in_indices.keys() | out_indices.keys()):
                print(f"Warning: Layer '{layer_name}' not found in the Pruning Metadata")
                continue

        if 'classifier' in layer_name:
            continue

        elif isinstance(layer, nn.Linear):
            # Get frozen indices
            freeze_dim0 = torch.tensor(out_indices[layer_name], dtype=torch.long, device=device)  # dim 0 indices
            freeze_dim1 = torch.tensor(in_indices[layer_name], dtype=torch.long, device=device)   # dim 1 indices
            freeze_linear_params(layer, weight_indices={'dim0': freeze_dim0, 'dim1': freeze_dim1})

        elif isinstance(layer, nn.Conv2d):
            freeze_dim0 = torch.tensor(out_indices[layer_name], dtype=torch.long, device=device)
            freeze_conv2d_params(layer, weight_indices=freeze_dim0)

        elif isinstance(layer, nn.LayerNorm):
            freeze_dim0 = torch.tensor(out_indices[layer_name], dtype=torch.long, device=device)
            freeze_layernorm_params(layer, weight_indices=freeze_dim0)

def change_module_name(pruning_dict):
    new_dict = {}
    for layer in pruning_dict.keys():
        if not layer.startswith("module."):
            new_dict["module." + layer] = pruning_dict[layer]

    return new_dict


def selective_gradient_clipping_norm(model, exclude_indices_dim0, exclude_indices_dim1, max_norm, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Clips gradients selectively while excluding certain indices from clipping.

    Args:
        model (nn.Module): The model whose gradients will be clipped.
        exclude_indices_dim0 (dict): Dictionary mapping layer names to indices in dim 0 to exclude from clipping.
        exclude_indices_dim1 (dict): Dictionary mapping layer names to indices in dim 1 to exclude from clipping.
        max_norm (float): The maximum norm for gradient clipping.
        device (str): The device on which the model is running.
    """

    # Dictionary to store original gradients
    original_weight_grads, original_bias_grads = {}, {}

    for layer_name, layer in model.named_modules():
        if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            continue

        if layer.weight.grad is None:
            print(f"Warning: Layer '{layer_name}' does not have gradients.")
            continue

        if 'classifier' in layer_name:
            continue

        # Check if layer needs selective clipping
        exclude_dim0 = exclude_indices_dim0.get(layer_name, None)
        exclude_dim1 = exclude_indices_dim1.get(layer_name, None)

        if exclude_dim0 is not None:
            exclude_dim0 = torch.tensor(exclude_dim0, dtype=torch.long, device=device)
            exclude_dim0 = exclude_dim0[exclude_dim0 < layer.weight.grad.shape[0]]  # Ensure valid indices

        if exclude_dim1 is not None:
            exclude_dim1 = torch.tensor(exclude_dim1, dtype=torch.long, device=device)
            exclude_dim1 = exclude_dim1[exclude_dim1 < layer.weight.grad.shape[1]]  # Ensure valid indices

        # Store original gradients before clipping
        with torch.no_grad():
            if isinstance(layer, nn.Linear) and exclude_dim0 is not None and exclude_dim1 is not None:
                original_weight_grads[layer_name] = layer.weight.grad[exclude_dim0[:, None], exclude_dim1].clone()
                if layer.bias is not None:
                    original_bias_grads[layer_name] = layer.bias.grad[exclude_dim0].clone()

            elif isinstance(layer, (nn.LayerNorm, nn.Conv2d)) and exclude_dim0 is not None:
                original_weight_grads[layer_name] = layer.weight.grad[exclude_dim0].clone()
                if layer.bias is not None:
                    original_bias_grads[layer_name] = layer.bias.grad[exclude_dim0].clone()

    # sample_layer = 'vit.encoder.layer.11.output.dense'
    # print("Layer Name:", sample_layer)
    # print("Before clipping")
    # for layer_name, layer in model.named_modules():
    #     if layer_name == sample_layer:
    #         print(layer.weight.grad.shape)
    #         exclude_dim0 = torch.tensor(exclude_indices_dim0.get(layer_name, None), dtype=torch.long, device=device)
    #         exclude_dim1 = torch.tensor(exclude_indices_dim1.get(layer_name, None), dtype=torch.long, device=device)
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1].shape)
    #         print(original_weight_grads[layer_name].shape)
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1] == original_weight_grads[layer_name])
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1].sum())
    #         print(original_weight_grads[layer_name].sum())
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1].sum() == original_weight_grads[layer_name].sum())

    # Apply global gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # print("After clipping")
    # for layer_name, layer in model.named_modules():
    #     if layer_name == sample_layer:
    #         print(layer.weight.grad.shape)
    #         exclude_dim0 = torch.tensor(exclude_indices_dim0.get(layer_name, None), dtype=torch.long, device=device)
    #         exclude_dim1 = torch.tensor(exclude_indices_dim1.get(layer_name, None), dtype=torch.long, device=device)
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1].shape)
    #         print(original_weight_grads[layer_name].shape)
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1] == original_weight_grads[layer_name])
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1].sum())
    #         print(original_weight_grads[layer_name].sum())
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1].sum() == original_weight_grads[layer_name].sum())

    # Restore excluded gradients
    with torch.no_grad():
        for layer_name, layer in model.named_modules():
            if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)) or layer.weight.grad is None:
                continue

            exclude_dim0 = exclude_indices_dim0.get(layer_name, None)
            exclude_dim1 = exclude_indices_dim1.get(layer_name, None)

            if exclude_dim0 is not None:
                exclude_dim0 = torch.tensor(exclude_dim0, dtype=torch.long, device=device)

            if exclude_dim1 is not None:
                exclude_dim1 = torch.tensor(exclude_dim1, dtype=torch.long, device=device)

            if layer_name in original_weight_grads:
                if isinstance(layer, nn.Linear) and exclude_dim0 is not None and exclude_dim1 is not None:
                    layer.weight.grad[exclude_dim0[:, None], exclude_dim1] = original_weight_grads[layer_name]

                elif isinstance(layer, (nn.LayerNorm, nn.Conv2d)) and exclude_dim0 is not None:
                    layer.weight.grad[exclude_dim0] = original_weight_grads[layer_name]

            if layer_name in original_bias_grads and layer.bias is not None:
                layer.bias.grad[exclude_dim0] = original_bias_grads[layer_name]

    # print("After restoring")
    # for layer_name, layer in model.named_modules():
    #     if layer_name == sample_layer:
    #         print(layer.weight.grad.shape)
    #         exclude_dim0 = torch.tensor(exclude_indices_dim0.get(layer_name, None), dtype=torch.long, device=device)
    #         exclude_dim1 = torch.tensor(exclude_indices_dim1.get(layer_name, None), dtype=torch.long, device=device)
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1].shape)
    #         print(original_weight_grads[layer_name].shape)
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1] == original_weight_grads[layer_name])
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1].sum())
    #         print(original_weight_grads[layer_name].sum())
    #         print(layer.weight.grad[exclude_dim0[:, None], exclude_dim1].sum() == original_weight_grads[layer_name].sum())



def plot_comparison(accuracy, macs, pruning_ratio, name=None):

    # Data (replace with your values)
    x_labels = ["Original", "Pruned", "Rebuilt"]
    
    x = np.arange(len(x_labels))  # X-axis positions

    fig, ax1 = plt.subplots(figsize=(7,5))

    # Plot Accuracy (Left Y-axis)
    ax1.set_xlabel("Model Stage")
    ax1.set_ylabel("Accuracy (%)", color="tab:blue")
    ax1.plot(x, accuracy, marker="o", linestyle="-", color="tab:blue", label="Accuracy")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Second Y-axis for MACs
    ax2 = ax1.twinx()
    ax2.set_ylabel("MACs (GFLOPs)", color="tab:red")
    ax2.plot(x, macs, marker="s", linestyle="--", color="tab:red", label="MACs")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # X-axis labels
    plt.xticks(x, x_labels)

    # Title and Grid
    plt.title("Accuracy vs MACs Across Model Stages")
    ax1.grid(True, linestyle="--", alpha=0.5)

    if name is not None:
        plt.savefig(f"./saves/plots/comparison_plot_vit_b_16_PR_{pruning_ratio}_{name}.png")
    else:
        plt.savefig(f"./saves/plots/comparison_plot_vit_b_16_PR_{pruning_ratio}.png")

def plot_tensor(tensor, title="Tensor Plot", cmap='viridis', figsize=(6, 4)):
    """
    Plot a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Tensor to plot (1D, 2D, or 3D image-like).
        title (str): Plot title.
        cmap (str): Colormap for 2D/3D tensors.
        figsize (tuple): Size of the plot.
    """
    tensor = tensor.detach().cpu()  # Move to CPU & detach if necessary
    
    plt.figure(figsize=figsize)
    
    if tensor.dim() == 1:
        plt.plot(tensor.numpy())
        plt.xlabel("Index")
        plt.ylabel("Value")
    
    elif tensor.dim() == 2:
        plt.imshow(tensor.numpy(), cmap=cmap, aspect='auto')
        plt.colorbar()
    
    elif tensor.dim() == 3:
        # If image tensor (C, H, W) or (H, W, C), adjust
        if tensor.shape[0] <= 3:  # Assume (C, H, W)
            tensor = tensor.permute(1, 2, 0)  # (H, W, C)
        plt.imshow(tensor.numpy())
    
    else:
        raise ValueError("Tensor has unsupported dimensions for plotting.")
    
    plt.title(title)
    plt.savefig(f"./saves/plots/{title}.png")