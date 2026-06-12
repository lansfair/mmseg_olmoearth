#! /usr/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

# 自动读取归档目录绝对路径，非必要不修改。  
export MM_ARCHIVE_HOME=$(echo "$PWD" | cut -d '/' -f 1-5)

# 自动读取归档数据目录路径，非必要不修改。
export MM_ARCHIVE_DATA_HOME="$MM_ARCHIVE_HOME/dat"

# 自动读取归档预训练骨干网络权重目录路径，非必要不修改。
export MM_ARCHIVE_CKPT_HOME="$MM_ARCHIVE_HOME/pretrained"

# Miniconda3 安装路径，非必要不修改。
export CONDA_HOME="$HOME/miniconda3"

# Miniconda3 虚拟环境名称，确保此环境可以正常运行你的项目。
export CONDA_ENV_NAME='mmdofa2'

# 指定训练使用的 GPU 索引，格式：0, 1, 2..., N。
export CUDA_VISIBLE_DEVICES='5'

# 训练启动配置名称，权重文件要求相同命名。
export CONFIG_NAME='cashew-plant_copernicus-fm-base_lp_wft_lr-5e-4_e50'


SRCDIR=$PWD
cd "$PWD/../../../"
export PYTHONPATH=".":"$PYTHONPATH"
PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"

WORK_DIR="/mnt/ht2-nas2/EO_test/zhc/mmsegmentation/work_dirs/"$(basename $SRCDIR)"/$CONFIG_NAME"
mkdir -p "$WORK_DIR"

$PYINTERPRETER "$PWD/tools/test.py" "$SRCDIR/configs/$CONFIG_NAME.py" "$SRCDIR/checkpoints/$CONFIG_NAME.pth" --work-dir "$WORK_DIR" --show-dir "$WORK_DIR"