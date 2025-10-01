import torch
from torch import nn
import timm
from datasets import load_imagenet
import torch_pruning as tp
import tqdm
from sklearn.metrics import accuracy_score


def evaluate_model(model, dataloader, device):
    all_preds = []
    all_labels = []
    # Move model to the correct device and ensure correct data type
    model = model.to(device).to(torch.float32)
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader):
            # Ensure images are on the same device and data type
            images = images.to(device).to(torch.float32)
            labels = labels.to(device)

            outputs = model(images)  # Perform forward pass
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print("Accuracy (%):", accuracy*100)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    example_inputs = torch.randn(1,3,224,224).to(device)
    # model_original = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
    # flops, params = tp.utils.count_ops_and_params(model_original, example_inputs)
    # print(f"Original FLOPs: {flops/1e9:.2f} G, Original Params: {params/1e6:.2f} M")
    
    # matvit = timm.create_model(
    #     "vit_base_patch16_224",
    #     pretrained=False,     # must be False if you change dimensions
    #     mlp_ratio=2.0,
    #     num_classes=1000).to(device)
    # flops, params = tp.utils.count_ops_and_params(matvit, example_inputs)
    # print(f"FLOPs: {flops/1e9:.2f} G, Params: {params/1e6:.2f} M")
    # print(matvit)
    
    # cfg = timm.get_model_cfg("vit_base_patch16_224")
    # print(cfg)
    
    for ratio in [0.5,1,2,4]:
        matvit = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,     # must be False if you change dimensions
        mlp_ratio=ratio,
        num_classes=1000).to(device)
        flops, params = tp.utils.count_ops_and_params(matvit, example_inputs)
        print(f"FLOPs: {flops/1e9:.2f} G, Params: {params/1e6:.2f} M")
    