import torch
from torch import nn
import timm, tome
from datasets import load_imagenet
import torch_pruning as tp


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
    return accuracy * 100

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    _,_, test_loader = load_imagenet(datapath="/var/scratch/dchabal/quokka/data/imagenet", batch_size=512, distributed=False, ra_sampler=False, debug=True)
    example_inputs = torch.randn(1,3,224,224).to(device)
    
    
    
    model_timm = timm.create_model("vit_base_patch16_224", pretrained=True)
    flops, params = tp.utils.count_ops_and_params(model_timm, example_inputs)
    
    # Load a pretrained model, can be any vit / deit model.
    model_tome = timm.create_model("vit_base_patch16_224", pretrained=True)
    # Patch the model with ToMe.
    tome.patch.timm(model_tome)
    # Set the number of tokens reduced per layer. See paper for details.
    model_tome.r = 13
    
    flops, params = tp.utils.count_ops_and_params(model_tome, example_inputs)
    exit()
    
    evaluate_model(model_timm, test_loader, device)
    evaluate_model(model_tome, test_loader, device)