#! /usr/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

export CONDA_HOME="$HOME/miniconda3"                                        # miniconda3 安装路径，非必要不修改。
export CONDA_ENV_NAME='sspp_tune'                                           # miniconda 虚拟环境名称，确保此环境可以正常运行你的项目。
export CUDA_VISIBLE_DEVICES='5'                                             # 指定训练使用的 GPU 索引，格式：0, 1, 2..., N。
export CONFIG_NAME='linear_probe_freeze10_ft40_50e'    # 训练启动配置名称，权重文件要求相同命名。

SRCDIR=$PWD
DATASET=$(basename $SRCDIR)

cd "$PWD/../../../"
export PYTHONPATH="/mnt/ht2-nas2/EO_test/openmmlab-archive/src/v1/mmseg/projects/copernicus/pastis":"$PYTHONPATH"
PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"

WORK_DIR="/tmp/work_dirs/$DATASET"
mkdir -p "$WORK_DIR"

$PYINTERPRETER "$PWD/tools/train.py" "$SRCDIR/configs/$CONFIG_NAME.py" --work-dir "$WORK_DIR"