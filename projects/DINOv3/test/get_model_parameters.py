#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_dinov3_vitl16_sat493m_to_py.py

读取 DINOv3 ViT-L/16 SAT493M 权重，提取模型结构和参数信息，
并将信息保存为一个 Python 文件。

默认使用 timm:
    vit_large_patch16_dinov3.sat493m

也支持 Hugging Face:
    facebook/dinov3-vitl16-pretrain-sat493m

用法示例：

1) 使用 timm 加载：
    python dump_dinov3_vitl16_sat493m_to_py.py \
        --backend timm \
        --output dinov3_vitl16_sat493m_info.py

2) 使用 Hugging Face 加载：
    python dump_dinov3_vitl16_sat493m_to_py.py \
        --backend hf \
        --output dinov3_vitl16_sat493m_info.py

3) 使用本地 Hugging Face 模型目录：
    python dump_dinov3_vitl16_sat493m_to_py.py \
        --backend hf \
        --hf-model /path/to/dinov3-vitl16-pretrain-sat493m \
        --output dinov3_vitl16_sat493m_info.py

注意：
本脚本默认不会把真实权重数值写进 Python 文件，
只写模型结构和参数元信息。
如果把 300M 级别权重数值直接写入 .py 文件，文件会非常巨大，不推荐。
"""

import argparse
import datetime
import os
import pprint
import sys
from typing import Any, Dict, List

import torch


def count_numel(items):
    return sum(x.numel() for x in items)


def safe_repr(obj: Any) -> str:
    """
    生成适合写入 Python 文件的 repr 字符串。
    """
    return pprint.pformat(obj, width=120, sort_dicts=False)


def load_with_timm(model_name: str):
    """
    使用 timm 加载 DINOv3 SAT493M 模型。
    默认模型名:
        vit_large_patch16_dinov3.sat493m
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "没有安装 timm。请先执行：pip install timm"
        ) from e

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
    )
    model.eval()
    return model


def load_with_huggingface(hf_model: str):
    """
    使用 Hugging Face Transformers 加载模型。
    默认模型名:
        facebook/dinov3-vitl16-pretrain-sat493m
    """
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "没有安装 transformers。请先执行：pip install transformers"
        ) from e

    model = AutoModel.from_pretrained(hf_model)
    model.eval()
    return model


def get_model_basic_info(model: torch.nn.Module, backend: str, model_id: str) -> Dict[str, Any]:
    """
    获取模型基本信息。
    """
    total_params = count_numel(model.parameters())
    trainable_params = count_numel(p for p in model.parameters() if p.requires_grad)
    buffer_count = count_numel(model.buffers())

    info = {
        "backend": backend,
        "model_id": model_id,
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "total_buffers": buffer_count,
    }

    # timm ViT 常见属性
    for attr in [
        "num_classes",
        "num_features",
        "embed_dim",
        "num_prefix_tokens",
        "patch_size",
    ]:
        if hasattr(model, attr):
            try:
                value = getattr(model, attr)
                info[attr] = value
            except Exception:
                pass

    # timm patch_embed 常见属性
    if hasattr(model, "patch_embed"):
        patch_embed = model.patch_embed
        patch_info = {}
        for attr in ["img_size", "patch_size", "grid_size", "num_patches"]:
            if hasattr(patch_embed, attr):
                try:
                    patch_info[attr] = getattr(patch_embed, attr)
                except Exception:
                    pass
        info["patch_embed"] = patch_info

    # Hugging Face config
    if hasattr(model, "config"):
        try:
            cfg = model.config
            info["hf_config_class"] = cfg.__class__.__name__
            info["hf_config"] = cfg.to_dict()
        except Exception:
            info["hf_config"] = str(model.config)

    return info


def get_named_parameters_info(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """
    获取每个参数的元信息，不保存真实权重值。
    """
    params_info = []

    for name, p in model.named_parameters():
        params_info.append(
            {
                "name": name,
                "shape": tuple(p.shape),
                "dtype": str(p.dtype),
                "numel": p.numel(),
                "requires_grad": bool(p.requires_grad),
            }
        )

    return params_info


def get_named_buffers_info(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """
    获取每个 buffer 的元信息。
    """
    buffers_info = []

    for name, b in model.named_buffers():
        buffers_info.append(
            {
                "name": name,
                "shape": tuple(b.shape),
                "dtype": str(b.dtype),
                "numel": b.numel(),
            }
        )

    return buffers_info


def get_modules_info(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """
    获取模块树信息。
    direct_parameters/direct_buffers 只统计当前模块直接拥有的参数/缓冲区，
    不递归统计子模块。
    """
    modules_info = []

    for name, module in model.named_modules():
        direct_parameters = sum(p.numel() for p in module.parameters(recurse=False))
        direct_buffers = sum(b.numel() for b in module.buffers(recurse=False))

        modules_info.append(
            {
                "name": name if name else "<root>",
                "type": module.__class__.__name__,
                "direct_parameters": direct_parameters,
                "direct_buffers": direct_buffers,
            }
        )

    return modules_info


def get_state_dict_info(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """
    获取 state_dict 中每一项的 key、shape、dtype、numel。
    不保存真实 tensor 数值。
    """
    state_info = []

    state_dict = model.state_dict()

    for name, tensor in state_dict.items():
        if torch.is_tensor(tensor):
            state_info.append(
                {
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                }
            )
        else:
            state_info.append(
                {
                    "name": name,
                    "type": str(type(tensor)),
                }
            )

    return state_info


def write_python_info_file(
    output_path: str,
    model_structure: str,
    basic_info: Dict[str, Any],
    params_info: List[Dict[str, Any]],
    buffers_info: List[Dict[str, Any]],
    modules_info: List[Dict[str, Any]],
    state_dict_info: List[Dict[str, Any]],
):
    """
    将所有信息写入一个 Python 文件。
    生成的文件可以被 import，例如：

        import dinov3_vitl16_sat493m_info as info
        print(info.BASIC_INFO)
        print(info.PARAMETERS[0])
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n")
        f.write('"""\n')
        f.write("Auto-generated DINOv3 ViT-L/16 SAT493M model information file.\n")
        f.write("This file contains model structure and parameter metadata only.\n")
        f.write("It does NOT contain raw weight tensor values.\n")
        f.write('"""\n\n')

        f.write("GENERATED_AT = ")
        f.write(repr(datetime.datetime.now().isoformat(timespec="seconds")))
        f.write("\n\n")

        f.write("BASIC_INFO = ")
        f.write(safe_repr(basic_info))
        f.write("\n\n")

        f.write("MODEL_STRUCTURE = r'''\n")
        f.write(model_structure)
        f.write("\n'''\n\n")

        f.write("PARAMETERS = ")
        f.write(safe_repr(params_info))
        f.write("\n\n")

        f.write("BUFFERS = ")
        f.write(safe_repr(buffers_info))
        f.write("\n\n")

        f.write("MODULES = ")
        f.write(safe_repr(modules_info))
        f.write("\n\n")

        f.write("STATE_DICT = ")
        f.write(safe_repr(state_dict_info))
        f.write("\n\n")

        f.write(
            """
def print_summary():
    print("Model ID:", BASIC_INFO.get("model_id"))
    print("Backend:", BASIC_INFO.get("backend"))
    print("Model class:", BASIC_INFO.get("model_class"))
    print("Total parameters:", f'{BASIC_INFO.get("total_parameters", 0):,}')
    print("Trainable parameters:", f'{BASIC_INFO.get("trainable_parameters", 0):,}')
    print("Non-trainable parameters:", f'{BASIC_INFO.get("non_trainable_parameters", 0):,}')
    print("Number of parameter tensors:", len(PARAMETERS))
    print("Number of buffers:", len(BUFFERS))
    print("Number of modules:", len(MODULES))
    print("Number of state_dict entries:", len(STATE_DICT))


if __name__ == "__main__":
    print_summary()
"""
        )


def main():
    parser = argparse.ArgumentParser(
        description="Dump DINOv3 ViT-L/16 SAT493M model structure and parameter metadata to a Python file."
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="timm",
        choices=["timm", "hf"],
        help="模型加载后端：timm 或 hf。默认 timm。",
    )

    parser.add_argument(
        "--timm-model",
        type=str,
        default="vit_large_patch16_dinov3.sat493m",
        help="timm 模型名。默认 vit_large_patch16_dinov3.sat493m。",
    )

    parser.add_argument(
        "--hf-model",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-sat493m",
        help="Hugging Face 模型名或本地目录。",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dinov3_vitl16_sat493m_info.py",
        help="输出 Python 文件路径。",
    )

    args = parser.parse_args()

    if args.backend == "timm":
        model_id = args.timm_model
        print(f"Loading model with timm: {model_id}")
        model = load_with_timm(model_id)
    else:
        model_id = args.hf_model
        print(f"Loading model with Hugging Face: {model_id}")
        model = load_with_huggingface(model_id)

    print("Collecting model structure...")
    model_structure = str(model)

    print("Collecting basic info...")
    basic_info = get_model_basic_info(model, args.backend, model_id)

    print("Collecting named parameters...")
    params_info = get_named_parameters_info(model)

    print("Collecting named buffers...")
    buffers_info = get_named_buffers_info(model)

    print("Collecting modules...")
    modules_info = get_modules_info(model)

    print("Collecting state_dict...")
    state_dict_info = get_state_dict_info(model)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Writing Python info file: {args.output}")
    write_python_info_file(
        output_path=args.output,
        model_structure=model_structure,
        basic_info=basic_info,
        params_info=params_info,
        buffers_info=buffers_info,
        modules_info=modules_info,
        state_dict_info=state_dict_info,
    )

    print("Done.")
    print(f"Output file: {args.output}")
    print()
    print("You can test it with:")
    print(f"    python {args.output}")


if __name__ == "__main__":
    main()
