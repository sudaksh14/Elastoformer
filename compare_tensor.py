import torch
import torch.nn as nn
from ViT_iterative_adaptivity import create_vit_general 
from prune_utils import get_vit_info

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



     
if __name__ == '__main__':
    paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_ViT_Iter_Adaptivity_test_SD.pth"] + \
            [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_ViT_Iter_Adaptivity_test_SD.pth" for i in range(2,5)]
    compare_weights_multilevel(paths, prune_steps=3)
    
    # compare_weights()
    
    # cls_token =  nn.Parameter(torch.randn(1, 1, 768))
    # pos_embed = nn.Parameter(torch.randn(1, 196 + 1, 768))

    # print(cls_token.shape)
    # print(pos_embed.shape)


    # FOR DEBUGGING
    # for name, layer in model.named_modules():
    #     if name == "vit.encoder.layer.0.attention.attention.key":
    #         freeze_dim0 = torch.tensor(non_pruned_index_out[name], dtype=torch.long, device=device)
    #         freeze_dim1 = torch.tensor(non_pruned_index_in[name], dtype=torch.long, device=device)
    #         t1 = layer.weight[freeze_dim0[:, None], freeze_dim1]
    #         print(t1.shape)
    #         # print(t1)

    #         t2 = torch.tensor(non_pruned_weights[name]["Weight"], device=device)
    #         print(t2.shape)
    #         # print(t2)
            
    #         print(t1==t2)
    #         overlap = torch.isin(t1, t2)
    #         print(t1[overlap].shape)