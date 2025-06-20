import torch
import time
from ViT_iterative_adaptivity import create_vit_general 
from prune_utils import get_vit_info
from transformers.models.vit.modeling_vit_pruned import PrunedViTSelfAttention, ViTSelfOutput, ViTLayer, ViTForImageClassification, ViTModel, ViTConfig

def load_vit_model(state_dict_path=None, device='cuda'):
    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location=device)
        
    model_info = get_vit_info(non_pruned_weights=state_dict, core_model=True)
    model = create_vit_general(dim_dict=model_info)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def measure_latency(model, input_size=(1, 3, 224, 224), warmup=10, trials=100):
    dummy_input = torch.randn(*input_size).to(next(model.parameters()).device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Measure latency
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(trials):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_latency = (end_time - start_time) / trials * 1000  # in ms
    return avg_latency





if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViTForImageClassification.from_pretrained("facebook/deit-base-patch16-224").to(device)
    model.eval()
    latency_ms = measure_latency(model)
    print(f"🕒 Average Latency (BS=128, 224x224): {latency_ms:.2f} ms")
    exit()

    paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_deit_Iter_Adaptivity_lowlr.pth"] + \
            [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_deit_Iter_Adaptivity_lowlr.pth" for i in range(2,7)]
    
    for state_dict_path in paths:
        print(f"Loading model from: {state_dict_path}")
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_vit_model(state_dict_path, device=device)
        latency_ms = measure_latency(model)
        print(f"🕒 Average Latency (BS=128, 224x224): {latency_ms:.2f} ms")



