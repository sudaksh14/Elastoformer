import torch
import time
from models.elastoformer import ElasticViTForImageClassification
from utils.prune_utils import get_vit_info, create_vit_general 
from utils.cnn_prune_utils import resnet_generator
import torchvision
import onnx
import onnxruntime as ort
import os


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

def measure_latency(model, input_size=(32, 3, 224, 224), warmup=10, trials=100):
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

def export_onnx(model, path, dummy_input=torch.randn((1, 3, 224, 224), dtype=torch.float32), device='cuda'):
    print(f"\nExporting PyTorch model to ONNX: {path}...")
    torch.onnx.export(
        model,                                                                    # The PyTorch model to export
        dummy_input.to(device),                        # A dummy input to trace the model
        path,                                                               # Path where the ONNX model will be saved
        export_params=True,                                                       # Export model parameters (weights)
        opset_version=11,                                          # ONNX opset version (e.g., 11)
        do_constant_folding=True,           # Apply constant folding for optimization
        input_names=['input'],              # Name for the input node in the ONNX graph
        output_names=['output'],            # Name for the output node in the ONNX graph
        dynamic_axes={                      # Define dynamic axes for flexible input sizes
            'input': {0: 'batch_size'},     # Allow variable batch size
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to {path}")


def get_model_info_vgg(state_dict=None):
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
            else:
                layer_info["num_features"] = weight.shape[0]

        
        if layer_info:
            model_info[layer_name.replace('.weight', '').replace('.bias', '')] = (layer_info["out_channels"], layer_info["in_channels"]) if "out_channels" in layer_info else (layer_info["num_features"], None)

    return model_info


class VGG_AnyDepth(torch.nn.Module):
    def __init__(self, channel_dict, num_classes=1000, bias=False):
        super(VGG_AnyDepth, self).__init__()
        self.features, last_channels = self._make_features(channel_dict)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.bias = bias

        # FOR IMAGENET
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(last_channels * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes)
        )

    def _make_features(self, channel_dict):
        layers = []
        last_channels = None

        sorted_keys = sorted(channel_dict.keys(), key=lambda x: int(x.split('.')[1]))
        print(sorted_keys)

        for key in sorted_keys:
            if "classifier" in key:
                continue
            out_ch, in_ch = channel_dict[key]
            idx = int(key.split('.')[1])

            # Special handling for the first layer: always 3 input channels
            if key == 'features.0':
                in_ch = 3

            if in_ch is not None and out_ch is not None:
                layers.append(torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=self.bias))
                layers.append(torch.nn.BatchNorm2d(out_ch))
                layers.append(torch.nn.ReLU(inplace=True))
                last_channels = out_ch

            # Optional: Add pooling at typical VGG locations
            if str(idx) in {'4', '11', '21', '31', '41'}:
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        return torch.nn.Sequential(*layers), last_channels

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_vgg_model(state_dict_path=None, device='cuda'):
    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location=device)

    model_info = get_model_info_vgg(state_dict)
    print(model_info)
    model = VGG_AnyDepth(model_info, num_classes=100)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------------------------------------VIT-----------------------------------------------------
    # model = ElasticViTForImageClassification.from_pretrained("facebook/deit-base-patch16-224").to(device)
    # model.eval()
    # export_onnx(model, path=os.path.join("./saves/onnx/DeIT_Original.onnx"), device=device)
    # latency_ms = measure_latency(model)
    # print(f"🕒 Average Latency (BS=32, 224x224): {latency_ms:.2f} ms")
    # exit()

    # paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_deit_Iter_Adaptivity_lowlr.pth"] + \
    #         [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_deit_Iter_Adaptivity_lowlr.pth" for i in range(2,7)]
    
    # paths = ["./saves/state_dicts/Vit_b_16_Rebuilt_Level_5_state_dict_deit_Iter_Adaptivity_lowlr.pth"]
    
    # for state_dict_path in paths:
    #     print(f"Loading model from: {state_dict_path}")
    #     model = load_vit_model(state_dict_path, device=device)
    #     latency_ms = measure_latency(model)
    #     print(f"🕒 Average Latency (BS=32, 224x224): {latency_ms:.2f} ms")

    # for state_dict_path in paths:
    #     print(f"Loading model from: {state_dict_path}")
    #     model = load_vit_model(state_dict_path, device=device)
    #     save_path = os.path.splitext(os.path.basename(state_dict_path))[0]
    #     export_onnx(model, path=os.path.join("./saves/onnx", f"{save_path}.onnx"), device=device)

    # exit()


    # ----------------------------------------------------RESNET-----------------------------------------------------
    # model = torchvision.models.resnet50().to(device)
    # model.eval()
    # latency_ms = measure_latency(model)
    # print(f"🕒 Average Latency (BS=32, 224x224): {latency_ms:.2f} ms")
    # exit()

        
    # paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_Resnet50_Iter_Adaptivity_noSD.pth"] + \
    #         [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_Resnet50_Iter_Adaptivity_noSD.pth" for i in range(2,7)]

    # for state_dict_path in paths:
    #     print(f"Loading model from: {state_dict_path}")
    #     model = load_resnet_model(state_dict_path, device=device)
    #     save_path = os.path.splitext(os.path.basename(state_dict_path))[0]
    #     export_onnx(model, path=os.path.join("./saves/onnx", f"{save_path}.onnx"), device=device)

    # exit()


    # input_batch_size = 1 # Or 128 if you meant a batch of 128 images
    # input_resolution = 224
    # input_shape = (input_batch_size, 3, input_resolution, input_resolution)

    # Optional: Load an image processor if you were actually processing real images
    # processor = AutoImageProcessor.from_pretrained("facebook/deit-base-patch16-224")
    # print(f"🕒 Average Latency (BS={input_batch_size}, {input_resolution}x{input_resolution}): {latency_ms:.2f} ms")
    
    # for state_dict_path in paths:
    #     print(f"Loading model from: {state_dict_path}")
    #     model = load_resnet_model(state_dict_path, device=device)
    #     latency_ms = measure_latency(model)
    #     print(f"🕒 Average Latency (BS=32, 224x224): {latency_ms:.2f} ms")

    # ----------------------------------------------------VGG16-----------------------------------------------------


    paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_VGG_Iter_Adaptivity_cifar100_testSD.pth"] + \
            [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_VGG_Iter_Adaptivity_cifar100_testSD.pth" for i in range(2,7)]
    
    for state_dict_path in paths:
        print(f"Loading model from: {state_dict_path}")
        model = load_vgg_model(state_dict_path, device=device)
        latency_ms = measure_latency(model)
        print(f"🕒 Average Latency (BS=32, 224x224): {latency_ms:.2f} ms")


