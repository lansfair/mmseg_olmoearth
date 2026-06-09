import sys
import torch

# 修改为你的 DINOv3 源码目录
REPO_DIR = "./dinov3-main"

# 修改为你的权重路径
WEIGHT_PATH = "./dinov3-vitl16-pretrain-sat493m.pth"

# 输出 txt 文件
OUT_TXT = "./dinov3_vitl16_model_info.txt"

sys.path.insert(0, REPO_DIR)

from dinov3.hub.backbones import dinov3_vitl16, Weights


def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "teacher", "student"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    new_ckpt = {}
    for k, v in ckpt.items():
        for prefix in ["module.", "backbone.", "model.", "teacher.", "student."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        new_ckpt[k] = v

    return new_ckpt


def main():
    model = dinov3_vitl16(pretrained=False, weights=Weights.SAT493M)

    ckpt = torch.load(WEIGHT_PATH, map_location="cpu")
    state_dict = get_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("DINOv3 ViT-L/16 SAT493M Model Information\n")
        f.write("=" * 80 + "\n\n")

        f.write("Model Structure:\n")
        f.write("-" * 80 + "\n")
        f.write(str(model))
        f.write("\n\n")

        f.write("Parameter Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Missing keys: {len(missing)}\n")
        f.write(f"Unexpected keys: {len(unexpected)}\n\n")

        if missing:
            f.write("Missing Keys:\n")
            for k in missing:
                f.write(f"{k}\n")
            f.write("\n")

        if unexpected:
            f.write("Unexpected Keys:\n")
            for k in unexpected:
                f.write(f"{k}\n")
            f.write("\n")

        f.write("Parameter Details:\n")
        f.write("-" * 80 + "\n")
        for name, param in model.named_parameters():
            f.write(
                f"{name:80s} "
                f"shape={list(param.shape)} "
                f"numel={param.numel():,} "
                f"requires_grad={param.requires_grad} "
                f"dtype={param.dtype}\n"
            )

    print(f"Model information saved to: {OUT_TXT}")


if __name__ == "__main__":
    main()
