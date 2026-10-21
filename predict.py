#!/usr/bin/env python
"""Knee osteoarthritis KL-grade predictor (from-scratch ensemble, no pretrained weights).

Usage:   python predict.py knee_xray.png [more.png ...] [--bundle knee_oa_model.pt] [--out-dir predictions]
Needs:   pip install torch pillow numpy matplotlib      (CPU is fine; a GPU is used automatically if present)
Output:  grade 0-4 + probability of every grade printed in the terminal, and a PNG per image with the X-ray,
         a Grad-CAM heat-map (where the model looked) and the probability bars.
Research prototype for a course project. NOT a medical device.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

NC = 5
CLASS_NAMES = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]

def make_head(in_dim):
    return nn.Sequential(nn.Linear(in_dim, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
                         nn.Dropout(0.4), nn.Linear(256, NC))


class MixedKernelNet(nn.Module):
    FEAT_DIM = 256
    def __init__(self):
        super().__init__()
        def blk(ci, co, k):
            return nn.Sequential(nn.Conv2d(ci, co, k, padding=k // 2, bias=False), nn.BatchNorm2d(co),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.block1, self.block2 = blk(3, 64, 3), blk(64, 128, 5)
        self.block3, self.block4 = blk(128, 128, 3), blk(128, 256, 5)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = make_head(256)
    def extract_features(self, x):
        return self.pool(self.block4(self.block3(self.block2(self.block1(x)))))
    def forward(self, x):
        return self.classifier(self.extract_features(x))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(c)
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)                                 # add THEN activate (standard residual block)


def down(ci, co):                                            # stride-2 3x3 conv-BN-ReLU (downsampling between stages)
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(co), nn.ReLU())


class ResNet(nn.Module):
    """v5: four stages (64/128/256/256 channels, overall /16, ~5 M params). v4 had two stages and only /4."""
    FEAT_DIM = 256
    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.stage1 = nn.Sequential(ResBlock(64), ResBlock(64))                              # 112x112
        self.stage2 = nn.Sequential(down(64, 128), ResBlock(128), ResBlock(128))             # 56x56
        self.stage3 = nn.Sequential(down(128, 256), ResBlock(256), ResBlock(256))           # 28x28
        self.stage4 = nn.Sequential(down(256, 256), ResBlock(256))                           # 14x14
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = make_head(256)
    def extract_features(self, x):
        return self.pool(self.stage4(self.stage3(self.stage2(self.stage1(self.entry(x))))))
    def forward(self, x):
        return self.classifier(self.extract_features(x))


class DenseBlock(nn.Module):
    def __init__(self, in_c, growth=16, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(nn.BatchNorm2d(in_c), nn.ReLU(), nn.Conv2d(in_c, growth, 3, padding=1, bias=False)))
            in_c += growth
        self.out_c = in_c
    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x


def transition(cin, cout):
    return nn.Sequential(nn.BatchNorm2d(cin), nn.ReLU(), nn.Conv2d(cin, cout, 1, bias=False), nn.AvgPool2d(2))


class DenseNet(nn.Module):
    FEAT_DIM = 192
    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.db1 = DenseBlock(32); self.t1 = transition(self.db1.out_c, 64)      # 112 -> 56
        self.db2 = DenseBlock(64); self.t2 = transition(self.db2.out_c, 128)     # 56 -> 28
        self.db3 = DenseBlock(128); self.t3 = transition(self.db3.out_c, 192)    # 28 -> 14   (v5: third dense block)
        self.final = nn.Sequential(nn.BatchNorm2d(192), nn.ReLU())
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = make_head(192)
    def extract_features(self, x):
        x = self.t3(self.db3(self.t2(self.db2(self.t1(self.db1(self.entry(x)))))))
        return self.pool(self.final(x))
    def forward(self, x):
        return self.classifier(self.extract_features(x))


MODELS = {"c": MixedKernelNet, "d": ResNet, "e": DenseNet}
ARCH_NAMES = {"c": "MixedKernelNet", "d": "ResNet", "e": "DenseNet"}



def load_bundle(path, device):
    b = torch.load(path, map_location="cpu", weights_only=True)
    models = []
    for m in b["models"]:
        net = MODELS[m["arch"]]()
        net.load_state_dict(m["state"])
        models.append((net.eval().to(device), float(m["weight"]), m["tag"]))
    tot = sum(w for _, w, _ in models)
    b["_models"] = [(n, w / tot, t) for n, w, t in models]
    b["_device"] = device
    return b


def preprocess(path, size):
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8)
    x = (torch.from_numpy(arr.astype(np.float32)) / 255.0 - 0.5) / 0.5
    return x.view(1, 1, size, size).expand(1, 3, -1, -1).contiguous(), arr


@torch.no_grad()
def ensemble_probs(b, x):
    x = x.to(b["_device"])
    P = 0
    for net, w, _ in b["_models"]:
        p = (net(x).softmax(1) + net(x.flip(-1)).softmax(1)) / 2          # horizontal-flip TTA
        P = P + w * p
    return P[0].cpu().numpy()


def final_grade(b, P):
    raw = int(P.argmax())
    adj = int((np.log(P + 1e-9) + np.array(b["bias"])).argmax())          # per-class bias tuned on validation
    return raw, adj


def grad_cam(b, x, cls):
    """Grad-CAM of the highest-weight model for class `cls`. Returns a (H,W) map in [0,1]."""
    net = max(b["_models"], key=lambda t: t[1])[0]
    layer = getattr(net, {"MixedKernelNet": "block4", "ResNet": "stage4", "DenseNet": "final"}[type(net).__name__])
    store = {}
    def fwd(m, i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))
    h = layer.register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            out = net(x.to(b["_device"]))
            net.zero_grad()
            out[0, cls].backward()
    finally:
        h.remove()
    a, g = store["a"].detach(), store["g"]
    raw = (g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True)
    cam = F.relu(raw)
    if float(cam.max()) <= 1e-12:                                          # degenerate map: fall back to the signed map
        cam = raw
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    cam = cam - cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def predict_image(b, path, out_png=None, cam=True, size=None, true=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = size or b["img"]
    x, arr = preprocess(path, size)
    P = ensemble_probs(b, x)
    raw, grade = final_grade(b, P)
    heat = grad_cam(b, x, grade) if cam else None
    if out_png and true is not None:                                        # comparison layout: input | model output | ground truth
        t = int(true); ok = (t == grade)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.6))
        ax[0].imshow(arr, cmap="gray"); ax[0].set_title("Input X-ray", fontsize=12); ax[0].axis("off")
        ax[1].imshow(arr, cmap="gray")
        if cam:
            ax[1].imshow(heat, cmap="jet", alpha=0.40)
        ax[1].set_title(f"Model output: G{grade} {CLASS_NAMES[grade]}\nconfidence {P[grade]:.0%}" + ("  (Grad-CAM)" if cam else ""),
                        fontsize=12, color="#2E7D32" if ok else "#C62828"); ax[1].axis("off")
        ax[1].text(0.5, -0.04, "  ".join(f"G{i} {P[i]:.0%}" for i in range(NC)), transform=ax[1].transAxes,
                   ha="center", va="top", fontsize=9, color="#444")
        ax[2].imshow(arr, cmap="gray"); ax[2].set_title(f"Ground truth: G{t} {CLASS_NAMES[t]}", fontsize=12); ax[2].axis("off")
        fig.suptitle(("CORRECT" if ok else "WRONG") + f"   (true G{t}, predicted G{grade})", fontsize=13, fontweight="bold",
                     color="#2E7D32" if ok else "#C62828")
        plt.tight_layout(rect=(0, 0.02, 1, 0.94)); plt.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    elif out_png:
        fig, ax = plt.subplots(1, 3 if cam else 2, figsize=(13 if cam else 9, 4.2))
        ax[0].imshow(arr, cmap="gray"); ax[0].set_title("input (224x224 grayscale)"); ax[0].axis("off")
        k = 1
        if cam:
            ax[1].imshow(arr, cmap="gray"); ax[1].imshow(heat, cmap="jet", alpha=0.40)
            ax[1].set_title(f"Grad-CAM (class G{grade})"); ax[1].axis("off"); k = 2
        cols = ["#2E7D32" if i == grade else "#9AA5B1" for i in range(NC)]
        ax[k].barh(range(NC), P, color=cols); ax[k].set_yticks(range(NC))
        ax[k].set_yticklabels([f"G{i} {CLASS_NAMES[i]}" for i in range(NC)]); ax[k].invert_yaxis(); ax[k].set_xlim(0, 1)
        ax[k].set_title(f"prediction: G{grade} {CLASS_NAMES[grade]}  ({P[grade]:.0%})")
        plt.tight_layout(); plt.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return {"grade": grade, "raw_grade": raw, "probs": P, "confidence": float(P[grade]), "cam": heat, "img": arr}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--bundle", default=str(Path(__file__).with_name("knee_oa_model.pt")))
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--no-cam", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = load_bundle(a.bundle, dev)
    print(f"loaded {len(b['_models'])} models on {dev}")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for p in a.images:
        r = predict_image(b, p, out / (Path(p).stem + "_pred.png"), cam=not a.no_cam)
        print(f"\n{p}\n  predicted KL grade: G{r['grade']} ({CLASS_NAMES[r['grade']]})  confidence {r['confidence']:.1%}"
              f"   [plain argmax without bias: G{r['raw_grade']}]")
        for i in range(NC):
            print(f"    G{i} {CLASS_NAMES[i]:9s} {r['probs'][i]:6.1%} " + "#" * int(round(r["probs"][i] * 30)))
        print(f"  saved {out / (Path(p).stem + '_pred.png')}")
    print("\nResearch prototype for a course project - not a medical device.")


if __name__ == "__main__":
    main()
