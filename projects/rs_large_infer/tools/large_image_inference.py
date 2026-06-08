import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from projects.rs_large_infer.large_image_inference import main


DEFAULT_ARGS = [
    # Fill this list to run without command-line arguments.
    # Example:
    # "/data/large.tif",
    # "projects/dinov3/configs/potsdam/dinov3-vitl16_4xb4-50e_potsdam-rgb.py",
    # "/checkpoints/model.pth",
    # "/outputs/pred_label.tif",
    # "--input-mode", "standard",
    # "--batch-size", "1",
    # "--device", "cuda:0",
]


if __name__ == "__main__":
    main(DEFAULT_ARGS if len(sys.argv) == 1 and DEFAULT_ARGS else None)
