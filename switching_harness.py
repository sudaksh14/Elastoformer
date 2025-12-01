import time
import torch
import torch.nn as nn
import torch.cuda as cuda

# -----------------------------
# Masked Linear Layer
# -----------------------------
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.active_in = torch.zeros(in_features, dtype=torch.bool)
        self.active_out = torch.zeros(out_features, dtype=torch.bool)

    def forward(self, x):
        w = self.weight[:, self.active_in]
        w = w[self.active_out, :]
        b = self.bias[self.active_out] if self.bias is not None else None
        return nn.functional.linear(x[:, self.active_in], w, b)

    def expand(self, added_in, added_out):
        self.active_in[added_in] = True
        self.active_out[added_out] = True


# -----------------------------
# Transition Functions
# -----------------------------
def transition_masked(model, cur_ckpt, prev_ckpt):
    for name, module in model.named_modules():
        if isinstance(module, MaskedLinear):
            cur_in, cur_out = cur_ckpt["non_pruned_index"]
            prev_in, prev_out = prev_ckpt["non_pruned_index"]

            added_in = list(set(prev_in) - set(cur_in))
            added_out = list(set(prev_out) - set(cur_out))
            module.expand(added_in, added_out)


def transition_struct(model, prev_ckpt, name_map):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in prev_ckpt["weights"]:
            W = prev_ckpt["weights"][f"{name}.weight"]
            B = prev_ckpt["weights"].get(f"{name}.bias", None)

            new_layer = nn.Linear(W.shape[1], W.shape[0], bias=B is not None)
            new_layer.weight.data.copy_(W)
            if B is not None:
                new_layer.bias.data.copy_(B)

            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_layer)


# -----------------------------
# Benchmark Utilities
# -----------------------------
def measure_transition(method, model, cur_ckpt, prev_ckpt, val_loader, name_map=None):
    start_mem = cuda.memory_allocated()
    start_time = time.time()

    if method == "mask":
        transition_masked(model, cur_ckpt, prev_ckpt)
    elif method == "struct":
        transition_struct(model, prev_ckpt, name_map)
    elif method == "swap":
        model.load_state_dict(prev_ckpt["weights"])

    elapsed = time.time() - start_time
    end_mem = cuda.memory_allocated()

    acc, loss = evaluate(model, val_loader)

    return {"method": method, "time": elapsed, "mem": end_mem - start_mem, "acc": acc, "loss": loss}


# -----------------------------
# Example Evaluation Function
# -----------------------------
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in dataloader:
        imgs, labels = imgs.cuda(), labels.cuda()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
    return correct / total * 100, total_loss / total


# -----------------------------
# Experiment Loop
# -----------------------------
def run_experiment(levels, base_model, val_loader, name_map=None):
    results = []

    # Start from smallest level (L-0)
    model = base_model.cuda()

    for i in reversed(range(len(levels) - 1)):
        cur_ckpt, prev_ckpt = levels[i+1], levels[i]

        for method in ["swap", "mask", "struct"]:
            res = measure_transition(method, model, cur_ckpt, prev_ckpt, val_loader, name_map)
            res["level_from"], res["level_to"] = i+1, i
            results.append(res)
            print(f"Transition {method}: L{i+1}→L{i}, acc={res['acc']:.2f}%, "
                  f"time={res['time']:.3f}s, mem={res['mem']/1e6:.2f}MB")

    return results


if __name__ == "__main__":
    run_experiment()
