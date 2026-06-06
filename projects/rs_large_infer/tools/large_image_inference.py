import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from projects.rs_large_infer.large_image_inference import main


if __name__ == "__main__":
    main()
