import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle
import numpy as np
from models.Resnet import *
from utils.partial_freezing import freeze_conv2d_params, freeze_conv2d_params_v2, freeze_bn_params
from timm.models.layers import DropPath



# Set device for training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)
    
    
def print_pruned_weights(layer, pruned_indices, dim=0):
    """
    Prints the weights of the layer that were pruned based on the pruned indices.

    Parameters:
        layer (torch.nn.Module): The layer from which weights were pruned.
        pruned_indices (torch.Tensor): Indices of the pruned channels.
        dim (int): Dimension along which pruning was applied (0 for output channels, 1 for input channels).
    """
    # Access the weight data
    weights = layer.weight.data

    if dim == 0:
        # Pruned output channels
        pruned_weights = weights[pruned_indices]
    elif dim == 1:
        # Pruned input channels
        pruned_weights = weights[:, pruned_indices]
    else:
        raise ValueError("Dimension for pruning must be 0 or 1.")
    
    print(f"Pruned Weights (dim={dim}) for indices {pruned_indices.tolist()}:\n", pruned_weights)

    
def prune_resnet_torch(model, amount=0.2):
    # Function to apply structural pruning
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0) # Prune along the channel dimension



def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer
        



class ChannelPruning(prune.BasePruningMethod):
    """
    Custom pruning method that prunes entire channels (filters) along a specified dimension.
    """
    PRUNING_TYPE = "structured"
    
    def __init__(self, indices, dim):
        
        self.indices = indices  # Indices of channels to prune
        self.dim = dim          # Dimension along which to prune
        

    def compute_mask(self, t, default_mask):
        """
        Computes a mask that removes the specified channels.
        """
        # mask = torch.ones_like(t, dtype=bool)
        mask = default_mask.clone()
        if self.dim == 0:
            mask[self.indices, :, :, :] = False
        elif self.dim == 1:
            mask[:, self.indices, :, :] = False
        else:
            raise ValueError("Dimension must be 0 or 1 for channel pruning.")
        return mask
    
    @classmethod
    def apply(cls, module, name, indices, dim, importance_scores=None):
        return super().apply(
            module,
            name,
            indices=indices,
            dim=dim,
            importance_scores=importance_scores,
        )


def prune_bias(layer, indices):
    """
    Prunes the bias of a layer by zeroing out values at specified indices.
    """
    if layer.bias is not None:
        mask = torch.ones_like(layer.bias, dtype=bool)
        mask[indices] = False  # Set the mask to zero for the pruned indices
        layer.bias.data[~mask] = 0  # Apply the mask to the bias to prune it
        print(f"Pruned bias for channels {indices.tolist()}.")

def get_pruned_channel_indices(tensor, amount, norm_type="L1", prune_step=0):
    """
    Calculate channel indices to prune based on smallest norms across channels.
    """
    num_channels = tensor.size(0)  # Number of output channels in Conv2D
    num_prune = int(num_channels * amount)

    if norm_type == "L1":
        norms = torch.abs(tensor).view(num_channels, -1).sum(dim=1)
        # norms = torch.sum(torch.abs(tensor), dim=(1, 2, 3))
    elif norm_type == "L2":
        norms = torch.sqrt((tensor ** 2).view(num_channels, -1).sum(dim=1))
    else:
        raise ValueError("norm_type must be 'L1' or 'L2'.")
    
    non_zero_channels = torch.sum(norms != 0).item()  # Number of channels with non-zero L1 norms
    zero_channels = torch.sum(norms == 0).item()      # Number of channels with zero L1 norms
    total_channels = norms.size(0)                   # Total number of channels (out_channels)

    # Output the results
    print(f"Number of channels with non-zero L1 norms: {non_zero_channels}")
    print(f"Number of channels with zero L1 norms: {zero_channels}")
    print(f"Total number of channels: {total_channels}")
    print(total_channels == non_zero_channels + zero_channels)
    
    if prune_step > 0:
        norms[norms == 0] = float('inf') # Set zero norms to infinity to avoid pruning them
        num_prune = int((total_channels - zero_channels) * amount)  # Prune ratio normalized based on non-zero norms

    # Get indices with the smallest norms
    pruned_indices = torch.topk(norms, num_prune, largest=False).indices
    return pruned_indices

def prune_conv_layer_dim0(model, layer_name, amount=0.2, norm_type="L1"):
    """
    Prunes the channels in a convolutional layer based on norms and returns pruned indices.
    """
    layer = dict(model.named_modules())[layer_name]
    pruned_indices = get_pruned_channel_indices(layer.weight, amount, norm_type=norm_type)

    # Apply channel pruning
    pruner = ChannelPruning(pruned_indices, dim=0)  # Pruning output channels
    pruner.apply(layer, name="weight", indices=pruned_indices, dim=0)
    print(f"Pruned channels in {layer_name} with indices {pruned_indices.tolist()} using {norm_type} norm.")
    
    # Prune the bias associated with the pruned channels
    prune_bias(layer, pruned_indices)

    return pruned_indices

def prune_conv_layer_dim1(model, layer_name, pruned_indices):
    """
    Prunes the input channels in the next convolutional layer based on the pruned indices from the previous layer.
    """
    layer = dict(model.named_modules())[layer_name]

    # Apply channel pruning on the input channels using the same indices
    pruner = ChannelPruning(pruned_indices, dim=1)  # Pruning input channels
    pruner.apply(layer, name="weight", indices=pruned_indices, dim=1)
    print(f"Pruned input channels in {layer_name} using the same indices: {pruned_indices.tolist()}.")
    
    # Prune the bias associated with the pruned channels
    prune_bias(layer, pruned_indices)
    

# Example CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def Index_Aware_pruner(model, amount, prune_step=0, res_blocks=3, plot=True, BN=True):
    
    pruning_index = {}
    count = 0
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            if name == "conv1" or name == "features.0":
                print(name,  type(module))
                pruned_indices_0 = get_pruned_channel_indices(module.weight, amount, prune_step=prune_step)
                pruner = ChannelPruning(pruned_indices_0, dim=0)
                pruner.apply(module, name="weight", indices=pruned_indices_0, dim=0) # Prune along the output channel dimension
                prune.remove(module, name="weight")
                pruning_index[name] = {"pruned_dim0": pruned_indices_0.tolist(), "pruned_dim1": []}
 
            else:
                
                print(name,  type(module))
                if 'downsample' in name:
                    
                    pruner_down_1 = ChannelPruning(indices_down_1, dim=1)
                    pruner_down_0 = ChannelPruning(indices_last_layer, dim=0)
                    pruner_down_1.apply(module, name="weight", indices=indices_down_1, dim=1) # Prune along the input channel dimension 
                    pruner_down_0.apply(module, name="weight", indices=indices_last_layer, dim=0) # Prune along the output channel dimension
                    prune.remove(module, name="weight")
                    pruning_index[name] = {"pruned_dim0": indices_last_layer.tolist(), "pruned_dim1": indices_down_1.tolist()}
                    
                else:
  
                    pruned_indices_0 = get_pruned_channel_indices(module.weight, amount, prune_step=prune_step)
                    pruner_dim_1 = ChannelPruning(indices_last_layer, dim=1)
                    pruner_dim_0 = ChannelPruning(pruned_indices_0, dim=0)
                    pruner_dim_1.apply(module, name="weight", indices=indices_last_layer, dim=1) # Prune along the input channel dimension
                    pruner_dim_0.apply(module, name="weight", indices=pruned_indices_0, dim=0) # Prune along the output channel dimension
                    prune.remove(module, name="weight")
                    pruning_index[name] = {"pruned_dim0": pruned_indices_0.tolist(), "pruned_dim1": indices_last_layer.tolist()}

                    count += 1
                
                
            indices_last_layer = pruned_indices_0
            if count != 0 and count % (res_blocks*2) == 0:
                indices_down_1 = indices_last_layer

            print("Pruned at output channel Indices: ", indices_last_layer.tolist())
    
    return model, pruning_index        
        

def prune_batchnorm(model, unpruned_indices):
    """
    Prune the BatchNorm layers based on provided indices.

    Args:
        model (torch.nn.Module): The PyTorch model (ResNet18 or any CNN).
        bn_prune_indices (dict): A dictionary where keys are BN layer names 
                                 and values are lists of channel indices to be restored.

    Returns:
        model (torch.nn.Module): The list with unpruned BN indices and weights.
    """

    bn_indices = {}
    bn_weights = {}

    dict_iter = iter(unpruned_indices.items())

    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            indices_to_keep = next(dict_iter)[1].get("unpruned_dim0", [])
            bn_indices[name] = indices_to_keep

            # Extract and update only the remaining channels
            bn_weights[name+"_weights"] = layer.weight.data[indices_to_keep]
            bn_weights[name+"_bias"] = layer.bias.data[indices_to_keep]
            bn_weights[name+"_runnin_mean"] = layer.running_mean[indices_to_keep]
            bn_weights[name+"_running_var"] = layer.running_var[indices_to_keep]

    return bn_indices, bn_weights
        
def get_nonzero_indices(model):
    non_zero_indices = {}
    new_channels = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            weights = module.weight.data
            indices = []

            # Iterate over both dimensions (dim 0: output channels, dim 1: input channels)
            for out_idx in range(weights.shape[0]):  # dim 0 (output channels)
                for in_idx in range(weights.shape[1]):  # dim 1 (input channels)
                    
                    if weights[out_idx, in_idx].abs().sum() > 0:  # Check if the (out_idx, in_idx) slice is non-zero
                        indices.append((out_idx, in_idx))
        
            non_zero_indices[name] = indices
            out_channels = len(set(t[0] for t in indices))
            in_channels = len(set(t[1] for t in indices))
            new_channels[name] = (out_channels, in_channels)
                            
                        
    return non_zero_indices, new_channels


# Function to copy weights from pruned model to reduced model
def copy_weights(pruned_model, rebuilt_model, nonzero_channels):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            n=0
            pruned_layer = module
            reduced_layer = dict(rebuilt_model.named_modules())[name]
            
            out_channels, in_channels,_,_ = reduced_layer.weight.shape
        
            with torch.no_grad():
                for i in range(out_channels):
                    for j in range(in_channels):
                        dim0, dim1 = nonzero_channels[name][n]
                        reduced_layer.weight.data[i,j] = pruned_layer.weight.data[dim0, dim1]
                        n += 1
     
                
def copy_weights_from_dict(pruned_model, unpruned_weights, BN_params, BN=False):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data = unpruned_weights[name]

    if BN:
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data = BN_params[name+"_weights"]
                module.bias.data = BN_params[name+"_bias"]
                module.running_mean = BN_params[name+"_runnin_mean"]
                module.running_var = BN_params[name+"_running_var"]



def get_pruned_channels_weights(model, pruned_index):
    """
    Analyze a structurally pruned model to determine:
    - The indices of pruned channels (dim 0 and dim 1).
    - The number of pruned channels in each layer.
    
    Parameters:
    - model: The Original PyTorch model.

    Returns:
    - pruned_info: A dictionary containing pruned indices, counts and weights for each layer.
    """
    pruned_info = {}
    pruned_weights = {}
    num_pruned_channels = {}
    
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):  # Check for Conv2d layers
            layer_info = {}
            
            # Get the weight tensor
            weights = layer.weight.detach()
            
            # Identify pruned filters along dim=0 (output channels)
            pruned_dim0 = pruned_index[name]["pruned_dim0"]
            if isinstance(pruned_dim0, int):  # Handle single index case
                pruned_dim0 = [pruned_dim0]
            layer_info['pruned_dim0'] = sorted(pruned_dim0)
            # layer_info['pruned_dim0'] = pruned_dim0
            
            
            # Identify pruned filters along dim=1 (input channels)
            pruned_dim1 = pruned_index[name]["pruned_dim1"]
            if isinstance(pruned_dim1, int):  # Handle single index case
                pruned_dim1 = [pruned_dim1]
            layer_info['pruned_dim1'] = sorted(pruned_dim1)
            # layer_info['pruned_dim1'] = pruned_dim1
            
            # Add layer information to pruned_info
            pruned_info[name] = layer_info
            num_pruned_channels[name] = (len(pruned_dim0), len(pruned_dim1))

            # Save pruned weights
            if pruned_dim0 and pruned_dim1:
                pruned_weights[name] = weights[pruned_dim0, :, :, :][:, pruned_dim1, :, :]
            else:
                pruned_weights[name] = weights[pruned_dim0, :, :, :]
    
    return pruned_info, num_pruned_channels, pruned_weights
            
            
def get_unpruned_indices_and_counts(model):
    non_pruned_info = {}
    num_unpruned_channels = {}
    unpruned_weights = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            layer_info = {}
            
            # Get the weight tensor
            weights = module.weight.detach()
            
            # Identify pruned filters along dim=0 (output channels)
            pruned_dim0 = torch.nonzero(weights.abs().sum(dim=(1, 2, 3)) > 0).squeeze().tolist()
            if isinstance(pruned_dim0, int):  # Handle single index case
                pruned_dim0 = [pruned_dim0]
            layer_info['unpruned_dim0'] = pruned_dim0
            
            
            # Identify pruned filters along dim=1 (input channels)
            pruned_dim1 = torch.nonzero(weights.abs().sum(dim=(0, 2, 3)) > 0).squeeze().tolist()
            if isinstance(pruned_dim1, int):  # Handle single index case
                pruned_dim1 = [pruned_dim1]
            layer_info['unpruned_dim1'] = pruned_dim1
        
            non_pruned_info[name] = layer_info
            num_unpruned_channels[name] = (len(pruned_dim0), len(pruned_dim1))
            unpruned_weights[name] = weights[pruned_dim0][:,pruned_dim1,:,:]
                            
                        
    return non_pruned_info, num_unpruned_channels, unpruned_weights

            

def extend_channels(model, pruned_dict):
    
    new_channel_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            new_in_channel = int(module.weight.data.shape[0] + pruned_dict[name][0])
            new_out_channel = int(module.weight.data.shape[1] + pruned_dict[name][1])
            
            new_channel_dict[name] = (new_in_channel, new_out_channel)
            
    return new_channel_dict


def get_rebuild_channels(unpruned_channels, pruned_channels):
    new_channels_dict = {}
    for name, weight in pruned_channels.items():
        new_in_channels = int(unpruned_channels[name][0] + pruned_channels[name][0])
        new_out_channels = int(unpruned_channels[name][1] + pruned_channels[name][1])
        
        new_channels_dict[name] = (new_in_channels, new_out_channels)
        
    return new_channels_dict


def get_core_weights(pruned_model, unpruned_weights, BN_params, restore_BN=False):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            unpruned_weights[name] = module.weight.data

        if restore_BN:
            if isinstance(module, nn.BatchNorm2d):
                BN_params[name+"_weights"] = module.weight.data
                BN_params[name+"_bias"] = module.bias.data
                BN_params[name+"_runnin_mean"] = module.running_mean
                BN_params[name+"_running_var"] = module.running_var




# Function to replace weights from Lower model to Higher model
def replace_previous_weights(model, pruned_indices, unpruned_indices, unpruned_weights):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            new_device = layer.weight.device

            # Retrieve pruned and unpruned indices
            pruned_dim0, pruned_dim1 = pruned_indices[name].values()
            unpruned_dim0, unpruned_dim1 = unpruned_indices[name].values()

            # Combine and sort
            combined_dim0 = sorted(pruned_dim0 + unpruned_dim0)
            combined_dim1 = sorted(pruned_dim1 + unpruned_dim1)

            # Find new indices for list1 and list2
            new_unpruned_dim0 = [combined_dim0.index(x) for x in unpruned_dim0]
            new_unpruned_dim1 = [combined_dim1.index(x) for x in unpruned_dim1]

            for i in range(len(new_unpruned_dim0)):
                out_idx = new_unpruned_dim0[i]  # Output channel index
                for j in range(len(new_unpruned_dim1)):
                    in_idx = new_unpruned_dim1[j]   # Input channel index
                    layer.weight.data[out_idx, in_idx, :, :] = unpruned_weights[name][i, j].to(new_device)

    return model

def restore_BN_weights(model, BN_params, restore_indices, restore_params, update_weights=False):

    dict_iter = iter(restore_indices.items())
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.weight.data = BN_params[name+"_weights"]
            layer.bias.data = BN_params[name+"_bias"]
            layer.running_mean = BN_params[name+"_runnin_mean"]
            layer.running_var = BN_params[name+"_running_var"]

            # Update the BN params after Fine-Tuning
            if update_weights:
                indices_to_replace = next(dict_iter)[1]

                layer.weight.data[indices_to_replace] = restore_params[name+"_weights"]
                layer.bias.data[indices_to_replace] = restore_params[name+"_bias"]
                layer.running_mean[indices_to_replace] = restore_params[name+"_runnin_mean"]
                layer.running_var[indices_to_replace] = restore_params[name+"_running_var"]
            # check_overlap(layer.weight.data, restore_params[name+"_weights"])

    return model


def reconstruct_Global_weights_from_dicts(model, pruned_indices, pruned_weights, unpruned_indices, unpruned_weights, freezing=False):
    """
    Reconstruct weights for a model using pruned and unpruned indices and tensors.

    Parameters:
    - pruned_indices: dict, mapping layer names to (dim0_indices, dim1_indices) of pruned weights.
    - pruned_weights: dict, mapping layer names to tensors of pruned weights.
    - unpruned_indices: dict, mapping layer names to (dim0_indices, dim1_indices) of unpruned weights.
    - unpruned_weights: dict, mapping layer names to tensors of unpruned weights.
    - model: torch.nn.Module, the model to reconstruct weights for.

    Returns:
    - reconstructed_model: torch.nn.Module, the model with reconstructed weights.
    """
    new_unpruned_dim0_freeze_list = {}
    new_unpruned_dim1_freeze_list = {}
    # Iterate through the model's state_dict to dynamically fetch layer shapes
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            new_device = layer.weight.device

            # Retrieve pruned and unpruned indices
            pruned_dim0, pruned_dim1 = pruned_indices[name].values()
            unpruned_dim0, unpruned_dim1 = unpruned_indices[name].values()

            # Combine and sort
            combined_dim0 = sorted(pruned_dim0 + unpruned_dim0)
            combined_dim1 = sorted(pruned_dim1 + unpruned_dim1)

            # Find new indices for list1 and list2
            new_pruned_dim0 = [combined_dim0.index(x) for x in pruned_dim0]
            new_unpruned_dim0 = [combined_dim0.index(x) for x in unpruned_dim0]

            new_pruned_dim1 = [combined_dim1.index(x) for x in pruned_dim1]
            new_unpruned_dim1 = [combined_dim1.index(x) for x in unpruned_dim1]

            new_unpruned_dim0_freeze_list[name] = new_unpruned_dim0
            new_unpruned_dim1_freeze_list[name] = new_unpruned_dim1

            # Assign pruned weights
            for i in range(len(new_pruned_dim0)):
                    out_idx = new_pruned_dim0[i]  # Output channel index
                    for j in range(len(new_pruned_dim1)):
                        in_idx = new_pruned_dim1[j]   # Input channel index
                        layer.weight.data[out_idx, in_idx, :, :] = pruned_weights[name][i, j].to(new_device)

            # Assign unpruned weights
            for i in range(len(new_unpruned_dim0)):
                    out_idx = new_unpruned_dim0[i]  # Output channel index
                    for j in range(len(new_unpruned_dim1)):
                        in_idx = new_unpruned_dim1[j]   # Input channel index
                        layer.weight.data[out_idx, in_idx, :, :] = unpruned_weights[name][i, j].to(new_device)

            # Channel Freezing --> NOT WORKING
            if freezing:
                for i in range(len(new_unpruned_dim0)):
                        out_idx = new_unpruned_dim0[i]  # Output channel index
                        for j in range(len(new_unpruned_dim1)):
                            in_idx = new_unpruned_dim1[j]   # Input channel index
                            layer.weight.data[out_idx, in_idx, :, :].requires_grad = False

                print(name, layer.weight.requires_grad)
                
    return model, new_unpruned_dim0_freeze_list, new_unpruned_dim1_freeze_list


def reconstruct_weights_from_dicts(model, pruned_indices, pruned_weights, unpruned_indices, unpruned_weights):
    """
    Reconstruct weights for a model using pruned and unpruned indices and tensors.

    Parameters:
    - pruned_indices: dict, mapping layer names to (dim0_indices, dim1_indices) of pruned weights.
    - pruned_weights: dict, mapping layer names to tensors of pruned weights.
    - unpruned_indices: dict, mapping layer names to (dim0_indices, dim1_indices) of unpruned weights.
    - unpruned_weights: dict, mapping layer names to tensors of unpruned weights.
    - model: torch.nn.Module, the model to reconstruct weights for.

    Returns:
    - reconstructed_model: torch.nn.Module, the model with reconstructed weights.
    """
    # Iterate through the model's state_dict to dynamically fetch layer shapes
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            
            new_device = layer.weight.device

            # Retrieve pruned and unpruned indices
            pruned_dim0, pruned_dim1 = pruned_indices[name].values()
            unpruned_dim0, unpruned_dim1 = unpruned_indices[name].values()

            # Assign pruned weights
            if pruned_dim0 and pruned_dim1:
                for i in range(len(pruned_dim0)):
                    out_idx = pruned_dim0[i]  # Output channel index
                    for j in range(len(pruned_dim1)):
                        in_idx = pruned_dim1[j]   # Input channel index
                        layer.weight.data[out_idx, in_idx, :, :] = pruned_weights[name][i, j].to(new_device)
            else:
                for i in range(len(pruned_dim0)):
                    out_idx = pruned_dim0[i]  # Output channel index
                    layer.weight.data[out_idx, :, :, :] = pruned_weights[name][i].to(new_device)
            

            # Assign unpruned weights
            for i in range(len(unpruned_dim0)):
                    out_idx = unpruned_dim0[i]  # Output channel index
                    for j in range(len(unpruned_dim1)):
                        in_idx = unpruned_dim1[j]   # Input channel index
                        layer.weight.data[out_idx, in_idx, :, :] = unpruned_weights[name][i, j].to(new_device)
            
    return model

# More efficient in use of memory
def zero_out_gradients_v2(model, dim0_indices, dim1_indices):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and layer.weight.grad is not None:
            dim0_idx = torch.tensor(dim0_indices[name], dtype=torch.long, device=layer.weight.grad.device)
            dim1_idx = torch.tensor(dim1_indices[name], dtype=torch.long, device=layer.weight.grad.device)

            # Ensure indices are within valid range
            dim0_idx = dim0_idx[dim0_idx < layer.weight.grad.shape[0]]
            dim1_idx = dim1_idx[dim1_idx < layer.weight.grad.shape[1]]

            # Create a grid of (dim0, dim1) combinations
            grid_dim0, grid_dim1 = torch.meshgrid(dim0_idx, dim1_idx, indexing='ij')

            # Zero out gradients at specified indices
            with torch.no_grad():
                layer.weight.grad[grid_dim0, grid_dim1, :, :] = 0


# NOT IN USE
def validate_frozen_channels(model, channel_indices):
    """
    Validate that the specified channels in the model are frozen.

    Args:
        model: The model with frozen channels.
        channel_indices: A dictionary with layer names and indices for `dim0` and `dim1`.

    Returns:
        A boolean indicating whether the specified channels are frozen.
    """

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and name in channel_indices:
            indices = channel_indices[name]
            frozen_dim0 = indices.get('unpruned_dim0', [])
            frozen_dim1 = indices.get('unpruned_dim1', [])

            if layer.weight.grad is None:
                print(name)
            
            if layer.weight.grad is not None:
                grad = layer.weight.grad
                
                # Validate dim0 gradients
                dim0_frozen = grad[frozen_dim0, :, :, :].abs().sum().item() == 0
                if not dim0_frozen:
                    print(f"Dim 0 not frozen for {name}: Indices {frozen_dim0}")
                
                # Validate dim1 gradients
                dim1_frozen = grad[:, frozen_dim1, :, :].abs().sum().item() == 0
                if not dim1_frozen:
                    print(f"Dim 1 not frozen for {name}: Indices {frozen_dim1}")

                # all_frozen = all_frozen and dim0_frozen and dim1_frozen
                all_frozen = dim0_frozen and dim1_frozen

            # Validate bias if it exists
            if layer.bias is not None and layer.bias.grad is not None:
                bias_grad = layer.bias.grad
                dim0_bias_frozen = bias_grad[frozen_dim0].abs().sum().item() == 0
                if not dim0_bias_frozen:
                    print(f"Bias not frozen for {name}: Indices {frozen_dim0}")
                all_frozen = all_frozen and dim0_bias_frozen

    return all_frozen


def check_overlap(big_tensor, small_tensor):
    """
    Check if there's any overlap between two tensors.

    Args:
        big_tensor: The larger tensor.
        small_tensor: The smaller tensor.

    Returns:
        A boolean indicating whether there's overlap.
    """

    print(f"Small Tensor Shape: {small_tensor.shape}")
    print(f"Big Tensor Shape: {big_tensor.shape}")

    overlap = torch.isin(small_tensor, big_tensor)

    print(f"Overlap found: {overlap.any().item()}")  # True if there's at least one overlap
    print(f"Matching values: {small_tensor[overlap].shape}")  # Prints matching values
    print(f"All values Nested: {small_tensor.flatten().shape == small_tensor[overlap].shape}")


def get_resnet_layer_sizes(state_dict):
    resnet_info = {}

    for key, value in state_dict.items():
        if 'weight' in key or 'bias' in key:
            shape = tuple(value.shape)
            resnet_info[key] = shape

    return resnet_info

def extract_weight_subsets(model, out_indices_dict, in_indices_dict):
    """
    Extracts a subset of weights from a model based on specified output and input indices.

    Args:
        model (torch.nn.Module): The model (ViT, ResNet, etc.).
        out_indices_dict (dict): Dictionary mapping layer names to output indices.
        in_indices_dict (dict): Dictionary mapping layer names to input indices.

    Returns:
        dict: Dictionary of selected weights and biases per layer.
    """
    selected_weights = {}

    for name, module in model.named_modules():
        if name not in out_indices_dict and name not in in_indices_dict:
            continue  # Skip if layer is not mentioned

        # Default to full indices if not provided
        out_indices = torch.tensor(out_indices_dict.get(name, []), dtype=torch.long)
        in_indices = torch.tensor(in_indices_dict.get(name, []), dtype=torch.long)

        weight = getattr(module, 'weight', None)
        bias = getattr(module, 'bias', None)

        if weight is None:
            continue

        if isinstance(module, torch.nn.Linear):
            # Shape: (out_dim, in_dim)
            selected_weight = (
                weight[out_indices][:, in_indices]
                if out_indices.numel() > 0 and in_indices.numel() > 0 else weight[out_indices])

            selected_bias = bias[out_indices] if bias is not None and out_indices.numel() > 0 else bias

        elif isinstance(module, torch.nn.Conv2d):
            # Shape: (out_channels, in_channels, kH, kW)
            selected_weight = (
                weight[out_indices][:, in_indices, :, :]
                if out_indices.numel() > 0 and in_indices.numel() > 0 else weight[out_indices])
                
            selected_bias = bias[out_indices] if bias is not None and out_indices.numel() > 0 else bias

        elif isinstance(module, torch.nn.BatchNorm2d):
            # Shape: (features,)
            selected_weight = weight[out_indices] if out_indices.numel() > 0 else weight
            selected_bias = bias[out_indices] if bias is not None and out_indices.numel() > 0 else bias

        else:
            continue  # Skip unsupported types

        selected_weights[name] = {
            "Weight": selected_weight.detach().cpu().numpy(),
            "Bias": selected_bias.detach().cpu().numpy() if selected_bias is not None else None
        }

    print("Weight subsets extracted successfully.")
    return selected_weights


def extract_core_weights(model):
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

        if isinstance(layer, torch.nn.Conv2d):
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

        elif isinstance(layer, torch.nn.BatchNorm2d):
            weight = layer.weight  # Shape: (dim,)
            bias = layer.bias  # Shape: (dim,)

            selected_weights[layer_name] = {
                "Weight": weight.detach().cpu().numpy(),
                "Bias": bias.detach().cpu().numpy() if bias is not None else None
            }

    print("Core weights extracted successfully")

    return selected_weights

def get_model_info_core(pruned_weights=None, non_pruned_weights=None):
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

    for layer_name, layer_weights in non_pruned_weights.items():
        layer_info = {}

        if "weight" in layer_name:
            weight = layer_weights
            if len(weight.shape) > 1: 
                layer_info["out_channels"] = weight.shape[0]
                layer_info["in_channels"] = weight.shape[1]
            else:
                layer_info["num_features"] = weight.shape[0]

        if layer_info:
            model_info[layer_name] = (layer_info["out_channels"], layer_info["in_channels"]) if "out_channels" in layer_info else (layer_info["num_features"], None)

    return model_info

def get_model_info_resnet(pruned_index=None, non_pruned_index=None):
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

    for layer_name, layer_index in non_pruned_index[1].items():
        layer_info = {}

        if "conv" in layer_name:
            out_non_pruned = layer_index
            out_pruned = pruned_index[1].get(layer_name, {})
            in_non_pruned = non_pruned_index[0].get(layer_name, {})
            in_pruned = pruned_index[0].get(layer_name, {})

            layer_info["out_channels"] = len(out_non_pruned) + len(out_pruned)
            layer_info["in_channels"] = len(in_non_pruned) + len(in_pruned)

            # print(layer_name, layer_info["out_channels"], layer_info["in_channels"])

        elif "bn" in layer_name:
            out_non_pruned = layer_index
            out_pruned = pruned_index[1].get(layer_name, {})

            layer_info["num_features"] = len(out_non_pruned) + len(out_pruned)
            # print(layer_name, layer_info["num_features"])

        elif "downsample" in layer_name:
            out_non_pruned = layer_index
            out_pruned = pruned_index[1].get(layer_name, {})
            layer_info["num_features"] = len(out_non_pruned) + len(out_pruned)
            # print(layer_name, layer_info["num_features"])

            if layer_name in non_pruned_index[0]:
                in_non_pruned = non_pruned_index[0].get(layer_name, {})
                in_pruned = pruned_index[0].get(layer_name, {})

                layer_info["out_channels"] = len(out_non_pruned) + len(out_pruned)
                layer_info["in_channels"] = len(in_non_pruned) + len(in_pruned)
                # print(layer_name, layer_info["out_channels"], layer_info["in_channels"])


        if layer_info:
            model_info[layer_name] = (layer_info["out_channels"], layer_info["in_channels"]) if "out_channels" in layer_info else (layer_info["num_features"], None)

    return model_info

def get_model_info_vgg(pruned_index=None, non_pruned_index=None):
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

    for layer_name, layer_index in non_pruned_index[1].items():
        layer_info = {}

        out_non_pruned = layer_index
        out_pruned = pruned_index[1].get(layer_name, {})
        in_non_pruned = non_pruned_index[0].get(layer_name, {})
        in_pruned = pruned_index[0].get(layer_name, {})

        layer_info["out_channels"] = len(out_non_pruned) + len(out_pruned)
        layer_info["in_channels"] = len(in_non_pruned) + len(in_pruned)

        if layer_name == 'features.0':
            layer_info["in_channels"] = 3

        if layer_info:
            model_info[layer_name] = (layer_info["out_channels"], layer_info["in_channels"]) if layer_info["in_channels"] > 0 else (layer_info["out_channels"], None)

    return model_info


# Merge and create new continuous indices / Very Imp for Iterative Growth
def merge_and_remap_indices(pruned, non_pruned):
    merged = sorted(set(pruned + non_pruned))
    remap = {idx: i for i, idx in enumerate(merged)}
    pruned_mapped = [remap[idx] for idx in pruned]
    non_pruned_mapped = [remap[idx] for idx in non_pruned]
    return pruned_mapped, non_pruned_mapped, merged

def update_global_weights(model, pruned_indices_list, non_pruned_indices_list, pruned_weights_dict, non_pruned_weights_dict, include_bias=True, device=None):
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

        if not isinstance(layer, (nn.Linear, nn.BatchNorm2d, nn.Conv2d)):
            continue

        if layer_name not in (pruned_out.keys() | pruned_in.keys()):
            print(f"Warning: Layer '{layer_name}' not found in the Pruning Metadata")
            continue

        if not hasattr(layer, 'weight'):
            continue

        weight = layer.weight.data

        pruned_weights = pruned_weights_dict[layer_name]
        non_pruned_weights = non_pruned_weights_dict[layer_name]


        if isinstance(layer, nn.Conv2d):
            if layer_name == "conv1" or layer_name == 'features.0':	
                
                pruned_dim0, non_pruned_dim0, _ = merge_and_remap_indices(pruned_out[layer_name], non_pruned_out[layer_name])

                # Replace pruned weights
                rows_pruned = torch.tensor(pruned_dim0, device=device)
                if rows_pruned.numel() > 0:
                    weight[rows_pruned] = torch.tensor(pruned_weights["Weight"], device=device)
                
                # Replace non-pruned weights
                rows_non_pruned = torch.tensor(non_pruned_dim0, device=device)
                if rows_non_pruned.numel() > 0:
                    weight[rows_non_pruned] = torch.tensor(non_pruned_weights["Weight"], device=device)

                pruned_out_mapped[layer_name] = pruned_dim0
                non_pruned_out_mapped[layer_name] = non_pruned_dim0

            else:
                pruned_dim0, non_pruned_dim0, _ = merge_and_remap_indices(pruned_out[layer_name], non_pruned_out[layer_name])
                pruned_dim1, non_pruned_dim1, _ = merge_and_remap_indices(pruned_in[layer_name], non_pruned_in[layer_name])

                # print("----------------------------------")
                # print(weight.shape)
                # print(layer_name)
                # print(pruned_dim0, len(pruned_dim0))
                # print(pruned_dim1, len(pruned_dim1))
                # print(non_pruned_dim0, len(non_pruned_dim0))    
                # print(non_pruned_dim1, len(non_pruned_dim1))
                # print(pruned_weights["Weight"].shape)
                # print(non_pruned_weights["Weight"].shape)
                
                # try:
                rows_pruned = torch.tensor(pruned_dim0, device=device)
                cols_pruned = torch.tensor(pruned_dim1, device=device)

                # Replace pruned weights
                if rows_pruned.numel() > 0 and cols_pruned.numel() > 0:
                    row_idx, col_idx = torch.meshgrid(rows_pruned, cols_pruned, indexing="ij")
                    # print(weight[row_idx, col_idx, :, :].shape)
                    # print("Pruned Weights Shape:", pruned_weights["Weight"].shape)
                    weight[row_idx, col_idx, :, :] = torch.tensor(pruned_weights["Weight"], device=device)

                rows_non_pruned = torch.tensor(non_pruned_dim0, device=device)
                cols_non_pruned = torch.tensor(non_pruned_dim1, device=device)
                
                # Replace non-pruned weights
                if rows_non_pruned.numel() > 0 and cols_non_pruned.numel() > 0:
                    row_idx, col_idx = torch.meshgrid(rows_non_pruned, cols_non_pruned, indexing="ij")
                    # print(weight[row_idx, col_idx, :, :].shape)
                    # print("Non Pruned Weights Shape:", non_pruned_weights["Weight"].shape)
                    weight[row_idx, col_idx, :, :] = torch.tensor(non_pruned_weights["Weight"], device=device)
                
                # except Exception as e:
                #     print(f"Failed at: {layer_name}")
                #     print("Error:", e)

                pruned_out_mapped[layer_name] = pruned_dim0
                non_pruned_out_mapped[layer_name] = non_pruned_dim0
                pruned_in_mapped[layer_name] = pruned_dim1
                non_pruned_in_mapped[layer_name] = non_pruned_dim1

        elif isinstance(layer, nn.BatchNorm2d):
            pruned_dim0, non_pruned_dim0, _ = merge_and_remap_indices(pruned_out[layer_name], non_pruned_out[layer_name])
            # print("----------------------------------")
            # print(layer)
            # print(weight.shape)
            # print(layer_name)
            # print(pruned_dim0, len(pruned_dim0))
            # print(non_pruned_dim0, len(non_pruned_dim0))    
            # print(pruned_weights["Weight"].shape)
            # print(non_pruned_weights["Weight"].shape)

            rows_pruned = torch.tensor(pruned_dim0, device=device)
            rows_non_pruned = torch.tensor(non_pruned_dim0, device=device)

            weight[rows_pruned] = torch.tensor(pruned_weights["Weight"], device=device)
            weight[rows_non_pruned] = torch.tensor(non_pruned_weights["Weight"], device=device)

            pruned_out_mapped[layer_name] = pruned_dim0
            non_pruned_out_mapped[layer_name] = non_pruned_dim0

        elif isinstance(layer, nn.Linear):
            continue


        if include_bias and layer.bias is not None:
            bias = layer.bias.data
            # print("------------Bias------------------")
            # print(bias.shape)
            # print(layer_name)
            # print(pruned_dim0, len(pruned_dim0))
            # print(non_pruned_dim0, len(non_pruned_dim0))    
            # print(pruned_weights["Bias"].shape)
            # print(non_pruned_weights["Bias"].shape)

            if len(pruned_dim0) > 0:
                rows_pruned = torch.tensor(pruned_dim0, device=device)
                bias[rows_pruned] = torch.tensor(pruned_weights["Bias"], device=device)
            if len(non_pruned_dim0) > 0:
                rows_non_pruned = torch.tensor(non_pruned_dim0, device=device)
                bias[rows_non_pruned] = torch.tensor(non_pruned_weights["Bias"], device=device)

    return model, [pruned_in_mapped, pruned_out_mapped], [non_pruned_in_mapped, non_pruned_out_mapped]


def freeze_partial_weights_cnn(model, in_indices, out_indices, device='cuda' if torch.cuda.is_available() else 'cpu'):

    for layer_name, layer in model.named_modules():
        
        if not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            continue
        else:
            if layer_name not in (in_indices.keys() | out_indices.keys()):
                print(f"Warning: Layer '{layer_name}' not found in the Pruning Metadata")
                continue


        if isinstance(layer, nn.Linear):
            continue

        elif isinstance(layer, nn.Conv2d):
            if layer_name == "conv1" or layer_name == 'features.0':
                freeze_dim0 = torch.tensor(out_indices[layer_name], dtype=torch.long, device=device)
                freeze_conv2d_params(layer, weight_indices=freeze_dim0)
            else:
                freeze_dim0 = torch.tensor(out_indices[layer_name], dtype=torch.long, device=device)  # dim 0 indices
                freeze_dim1 = torch.tensor(in_indices[layer_name], dtype=torch.long, device=device)   # dim 1 indices

                # Create a grid of (dim0, dim1) combinations
                grid_dim0, grid_dim1 = torch.meshgrid(freeze_dim0, freeze_dim1, indexing='ij')
                freeze_conv2d_params_v2(layer, weight_indices={'dim0': grid_dim0, 'dim1': grid_dim1})
            

        elif isinstance(layer, nn.BatchNorm2d):
            freeze_dim0 = torch.tensor(out_indices[layer_name], dtype=torch.long, device=device)
            freeze_bn_params(layer, weight_indices=freeze_dim0)



def selective_gradient_clipping_norm_cnn(model, exclude_indices_dim0, exclude_indices_dim1, max_norm, device='cuda' if torch.cuda.is_available() else 'cpu'):
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

        if isinstance(layer, nn.Linear):
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
            if isinstance(layer, nn.Conv2d) and exclude_dim0 is not None and exclude_dim1 is not None:
                grid_dim0, grid_dim1 = torch.meshgrid(exclude_dim0, exclude_dim1, indexing='ij')
                original_weight_grads[layer_name] = layer.weight.grad[grid_dim0, grid_dim1, :,:].clone()
                if layer.bias is not None:
                    original_bias_grads[layer_name] = layer.bias.grad[exclude_dim0].clone()

            elif isinstance(layer, nn.BatchNorm2d) and exclude_dim0 is not None:
                original_weight_grads[layer_name] = layer.weight.grad[exclude_dim0].clone()
                if layer.bias is not None:
                    original_bias_grads[layer_name] = layer.bias.grad[exclude_dim0].clone()

    # Apply global gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)

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
                if isinstance(layer, nn.Conv2d) and exclude_dim0 is not None and exclude_dim1 is not None:
                    grid_dim0, grid_dim1 = torch.meshgrid(exclude_dim0, exclude_dim1, indexing='ij')
                    layer.weight.grad[grid_dim0, grid_dim1, :,:] = original_weight_grads[layer_name]

                elif isinstance(layer, nn.BatchNorm2d) and exclude_dim0 is not None:
                    layer.weight.grad[exclude_dim0] = original_weight_grads[layer_name]

            if layer_name in original_bias_grads and layer.bias is not None:
                layer.bias.grad[exclude_dim0] = original_bias_grads[layer_name]


def inject_stochastic_depth_resnet(model, max_drop_path=0.1):
    """
    Injects DropPath (stochastic depth) into each residual block of a ResNet.

    Args:
        model (nn.Module): ResNet model with layers like model.layer1, model.layer2, etc.
        max_drop_path (float): Maximum drop probability linearly scaled across depth.
    """
    # Collect all residual blocks (assumes model.layer1, layer2, layer3, layer4)
    all_blocks = []
    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, name)
        for block in layer:
            all_blocks.append(block)

    num_blocks = len(all_blocks)
    drop_rates = torch.linspace(0, max_drop_path, steps=num_blocks).tolist()

    for i, block in enumerate(all_blocks):
        drop_prob = drop_rates[i]
        if not hasattr(block, 'drop_path'):
            print("Adding Drop Path to the ResNet Block")
            block.drop_path = DropPath(drop_prob) if drop_prob > 0 else nn.Identity()
        else:
            block.drop_path.drop_prob = drop_prob  # Update if already exists

        # Modify the forward method of each block (monkey patching)
        def modified_forward(self, x):
            identity = x

            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)  # Apply stochastic depth
            out += identity
            return self.relu(out)

        block.forward = modified_forward.__get__(block, block.__class__)
