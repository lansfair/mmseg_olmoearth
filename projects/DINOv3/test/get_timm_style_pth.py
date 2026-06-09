#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_dinov3_timm_original_to_txt.py

功能：
    加载 Meta/timm 原始 DINOv3 权重，然后将模型结构、参数信息、buffer 信息、
    state_dict 信息、missing/unexpected keys 等保存到 txt 文件。

适用场景：
    checkpoint key 类似：
        blocks.0.attn.qkv.weight
        blocks.0.attn.proj.weight
        blocks.0.mlp.fc1.weight
        blocks.0.mlp.fc2.weight
        patch_embed.proj.weight
        cls_token
        mask_token
        storage_tokens

推荐模型：
    DINOv3 ViT-L/16 SAT493M:
        vit_large_patch16_dinov3_qkvb.sat493m
    或：
        vit_large_patch16_dinov3.sat493m

安装依赖：
    pip install torch timm

如果读取 safetensors：
    pip install safetensors

示例：
    python dump_dinov3_timm_original_to_txt.py \
        --checkpoint /path/to/dinov3_vitl16_sat493m.pth \
        --output dinov3_timm_model_info.txt

如果你确认权重里没有 attn.qkv.bias，也可以手动指定：
    python dump_dinov3_timm_original_to_txt.py \
        --checkpoint /path/to/dinov3_vitl16_sat493m.pth \
        --model-name vit_large_patch16_dinov3.sat493m \
        --output dinov3_timm_model_info.txt
"""

import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn


# -------------------------
# 基础工具函数
# -------------------------
def fmt_num(n: int) -> str:
    return f"{int(n):,}"


def tensor_meta(t: torch.Tensor) -> str:
    return (
        f"shape={tuple(t.shape)}, "
        f"dtype={t.dtype}, "
        f"numel={fmt_num(t.numel())}"
    )


def unwrap_checkpoint(obj):
    """
    从 checkpoint 中提取 state_dict。

    常见 checkpoint 结构：
        checkpoint
        checkpoint["state_dict"]
        checkpoint["model"]
        checkpoint["teacher"]
        checkpoint["student"]
        checkpoint["module"]
    """
    if isinstance(obj, OrderedDict):
        return obj

    if isinstance(obj, dict):
        candidate_keys = [
            "state_dict",
            "model",
            "teacher",
            "student",
            "module",
            "backbone",
            "encoder",
            "net",
        ]

        for key in candidate_keys:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]

        # 如果 dict 本身就是 state_dict
        tensor_like = {
            k: v for k, v in obj.items()
            if torch.is_tensor(v)
        }
        if len(tensor_like) > 0:
            return obj

    raise ValueError(
        "无法从 checkpoint 中提取 state_dict。"
        "请检查 checkpoint 是否是 PyTorch 权重文件。"
    )


def load_checkpoint(path: str):
    """
    读取 .pth/.pt/.bin/.safetensors 权重。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "当前 checkpoint 是 safetensors 格式，请先安装：pip install safetensors"
            ) from e
        ckpt = load_file(path, device="cpu")
    else:
        ckpt = torch.load(path, map_location="cpu")

    state_dict = unwrap_checkpoint(ckpt)

    # 只保留 tensor 项
    clean = OrderedDict()
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            clean[k] = v

    if len(clean) == 0:
        raise ValueError("state_dict 中没有 tensor 参数。")

    return clean


def strip_prefix_once(state_dict, prefix: str):
    """
    如果大部分 key 都带有某个 prefix，则去掉。
    """
    new_sd = OrderedDict()
    plen = len(prefix)
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_sd[k[plen:]] = v
        else:
            new_sd[k] = v
    return new_sd


def choose_best_state_dict_for_model(raw_sd, model_sd_keys):
    """
    针对不同 checkpoint wrapper 自动尝试去前缀，选择与 timm 模型 key 命中最多的一版。

    注意：
        这里不会做 Hugging Face <-> timm 的 key 转换。
        它只是处理 module. / model. / teacher. / backbone. 等外层前缀。
    """
    prefixes = [
        "",
        "module.",
        "model.",
        "teacher.",
        "student.",
        "backbone.",
        "encoder.",
        "net.",
        "teacher.backbone.",
        "student.backbone.",
        "model.backbone.",
    ]

    model_keys = set(model_sd_keys)

    best_sd = raw_sd
    best_prefix = ""
    best_hits = -1

    for prefix in prefixes:
        if prefix:
            cand = strip_prefix_once(raw_sd, prefix)
        else:
            cand = raw_sd

        hits = len(set(cand.keys()) & model_keys)
        if hits > best_hits:
            best_hits = hits
            best_sd = cand
            best_prefix = prefix

    return best_sd, best_prefix, best_hits


def detect_qkv_bias(state_dict):
    """
    检测 checkpoint 中是否存在 blocks.*.attn.qkv.bias。
    如果存在，优先使用 timm 的 qkvb 模型名。
    """
    for k in state_dict.keys():
        if k.endswith("attn.qkv.bias"):
            return True
    return False


def create_timm_model(model_name: str):
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "未安装 timm。请先执行：pip install timm"
        ) from e

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
    )
    model.eval()
    return model


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


# -------------------------
# 写入 txt
# -------------------------
def write_model_info_txt(
    output_path,
    model,
    model_name,
    checkpoint_path,
    used_prefix,
    matched_key_count,
    load_result,
):
    total, trainable, frozen = count_params(model)

    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("DINOv3 Meta/timm Model Dump\n")
        f.write("=" * 120 + "\n")
        f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Model name:   {model_name}\n")
        f.write(f"Checkpoint:   {checkpoint_path}\n")
        f.write(f"Used prefix stripped: {repr(used_prefix)}\n")
        f.write(f"Matched state_dict keys before loading: {matched_key_count}\n")
        f.write("\n")

        f.write("=" * 120 + "\n")
        f.write("Load Result\n")
        f.write("=" * 120 + "\n")
        f.write(f"Missing keys:    {len(missing_keys)}\n")
        f.write(f"Unexpected keys: {len(unexpected_keys)}\n\n")

        if missing_keys:
            f.write("[Missing Keys]\n")
            for k in missing_keys:
                f.write(f"  {k}\n")
            f.write("\n")

        if unexpected_keys:
            f.write("[Unexpected Keys]\n")
            for k in unexpected_keys:
                f.write(f"  {k}\n")
            f.write("\n")

        f.write("=" * 120 + "\n")
        f.write("Basic Parameter Summary\n")
        f.write("=" * 120 + "\n")
        f.write(f"Total parameters:        {fmt_num(total)}\n")
        f.write(f"Trainable parameters:    {fmt_num(trainable)}\n")
        f.write(f"Non-trainable parameters:{fmt_num(frozen)}\n")
        f.write(f"Total buffers:           {fmt_num(sum(b.numel() for b in model.buffers()))}\n")
        f.write("\n")

        # 常见 timm ViT 属性
        f.write("=" * 120 + "\n")
        f.write("Common Model Attributes\n")
        f.write("=" * 120 + "\n")
        attrs = [
            "num_classes",
            "num_features",
            "embed_dim",
            "num_prefix_tokens",
            "num_reg_tokens",
            "num_tokens",
        ]
        for attr in attrs:
            if hasattr(model, attr):
                try:
                    f.write(f"{attr}: {getattr(model, attr)}\n")
                except Exception:
                    pass

        if hasattr(model, "patch_embed"):
            f.write("\n[patch_embed]\n")
            patch_embed = model.patch_embed
            for attr in ["img_size", "patch_size", "grid_size", "num_patches"]:
                if hasattr(patch_embed, attr):
                    try:
                        f.write(f"{attr}: {getattr(patch_embed, attr)}\n")
                    except Exception:
                        pass
        f.write("\n")

        f.write("=" * 120 + "\n")
        f.write("Model Structure\n")
        f.write("=" * 120 + "\n")
        f.write(str(model))
        f.write("\n\n")

        f.write("=" * 120 + "\n")
        f.write("Named Parameters\n")
        f.write("=" * 120 + "\n")
        for name, p in model.named_parameters():
            f.write(
                f"{name:90s} | "
                f"shape={str(tuple(p.shape)):30s} | "
                f"dtype={str(p.dtype):15s} | "
                f"numel={fmt_num(p.numel()):15s} | "
                f"requires_grad={p.requires_grad}\n"
            )

        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("Named Buffers\n")
        f.write("=" * 120 + "\n")
        for name, b in model.named_buffers():
            f.write(
                f"{name:90s} | "
                f"shape={str(tuple(b.shape)):30s} | "
                f"dtype={str(b.dtype):15s} | "
                f"numel={fmt_num(b.numel())}\n"
            )

        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("Module Tree\n")
        f.write("=" * 120 + "\n")
        for name, module in model.named_modules():
            module_name = name if name else "<root>"
            direct_params = sum(
                p.numel() for p in module.parameters(recurse=False)
            )
            direct_buffers = sum(
                b.numel() for b in module.buffers(recurse=False)
            )
            f.write(
                f"{module_name:90s} | "
                f"type={module.__class__.__name__:35s} | "
                f"direct_params={fmt_num(direct_params):15s} | "
                f"direct_buffers={fmt_num(direct_buffers)}\n"
            )

        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("State Dict After Loading\n")
        f.write("=" * 120 + "\n")
        for name, t in model.state_dict().items():
            if torch.is_tensor(t):
                f.write(f"{name:90s} | {tensor_meta(t)}\n")
            else:
                f.write(f"{name:90s} | type={type(t)}\n")


def write_checkpoint_key_preview(output_path, state_dict, title="Checkpoint Key Preview"):
    """
    在 txt 末尾补充原始 checkpoint 的 key 预览，方便排查。
    """
    with open(output_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write(title + "\n")
        f.write("=" * 120 + "\n")
        f.write(f"Total tensor keys in checkpoint: {len(state_dict)}\n\n")

        f.write("[First 100 checkpoint keys]\n")
        for i, (k, v) in enumerate(state_dict.items()):
            if i >= 100:
                break
            f.write(f"{i:04d}: {k:90s} | {tensor_meta(v)}\n")


# -------------------------
# 主函数
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Load Meta/timm original DINOv3 checkpoint and dump model structure/parameters to txt."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Meta/timm 原始 DINOv3 checkpoint 路径，例如 .pth/.pt/.bin/.safetensors",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="auto",
        help=(
            "timm 模型名。默认 auto。"
            "auto 会根据 checkpoint 是否含 attn.qkv.bias 选择 "
            "vit_large_patch16_dinov3_qkvb.sat493m 或 vit_large_patch16_dinov3.sat493m。"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dinov3_timm_model_info.txt",
        help="输出 txt 文件路径。",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="使用 strict=True 加载。默认 False，更利于诊断 missing/unexpected keys。",
    )

    parser.add_argument(
        "--freeze",
        action="store_true",
        help="加载后冻结模型参数。只影响 txt 中 requires_grad 显示，不影响权重加载。",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Loading checkpoint")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")

    raw_sd = load_checkpoint(args.checkpoint)

    has_qkv_bias = detect_qkv_bias(raw_sd)

    if args.model_name == "auto":
        if has_qkv_bias:
            model_name = "vit_large_patch16_dinov3_qkvb.sat493m"
        else:
            model_name = "vit_large_patch16_dinov3.sat493m"
    else:
        model_name = args.model_name

    print(f"Detected attn.qkv.bias: {has_qkv_bias}")
    print(f"Using timm model: {model_name}")

    print("\n" + "=" * 80)
    print("Creating timm model")
    print("=" * 80)
    model = create_timm_model(model_name)

    model_sd_keys = list(model.state_dict().keys())

    best_sd, used_prefix, matched = choose_best_state_dict_for_model(
        raw_sd,
        model_sd_keys,
    )

    print(f"Best stripped prefix: {repr(used_prefix)}")
    print(f"Matched keys before loading: {matched} / {len(model_sd_keys)}")

    print("\n" + "=" * 80)
    print("Loading weights")
    print("=" * 80)

    load_result = model.load_state_dict(
        best_sd,
        strict=args.strict,
    )

    if args.freeze:
        for p in model.parameters():
            p.requires_grad = False

    model.eval()

    print(f"Missing keys:    {len(load_result.missing_keys)}")
    print(f"Unexpected keys: {len(load_result.unexpected_keys)}")

    if len(load_result.missing_keys) > 0:
        print("\nFirst 20 missing keys:")
        for k in load_result.missing_keys[:20]:
            print("  ", k)

    if len(load_result.unexpected_keys) > 0:
        print("\nFirst 20 unexpected keys:")
        for k in load_result.unexpected_keys[:20]:
            print("  ", k)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Writing txt")
    print("=" * 80)
    print(f"Output: {args.output}")

    write_model_info_txt(
        output_path=args.output,
        model=model,
        model_name=model_name,
        checkpoint_path=args.checkpoint,
        used_prefix=used_prefix,
        matched_key_count=matched,
        load_result=load_result,
    )

    write_checkpoint_key_preview(
        output_path=args.output,
        state_dict=best_sd,
        title="Checkpoint Key Preview After Prefix Processing",
    )

    print("\nDone.")
    print(f"Saved to: {args.output}")

    if len(load_result.missing_keys) == 0 and len(load_result.unexpected_keys) == 0:
        print("Load status: OK, no missing/unexpected keys.")
    else:
        print("Load status: WARNING, please check missing/unexpected keys in txt.")


if __name__ == "__main__":
    main()
