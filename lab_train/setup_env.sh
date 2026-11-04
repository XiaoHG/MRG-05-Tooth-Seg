#!/usr/bin/env bash
# 一键安装 lab_train 训练环境（Miniconda + conda env + PyTorch(CUDA) + 项目依赖）
#
# 用法：
#   bash setup_env.sh                 # 默认官方源（快）
#   bash setup_env.sh --mirror cn     # 启用国内镜像（清华 TUNA + 阿里云）
#   bash setup_env.sh --mirror tuna   # 仅清华 TUNA（conda/pip/miniconda），PyTorch 官方
#   bash setup_env.sh --mirror aliyun # 仅阿里云（pip + PyTorch wheels）
#   bash setup_env.sh --external      # 额外装 RT-DETRv2-L / RF-DETR-M 后端
#   bash setup_env.sh --pretrain      # 额外装 DINOv3 域内预训练依赖（transformers 等）
#   CUDA=cu128 bash setup_env.sh      # 用 cu128 wheel（较旧稳定版）
#   ENV_NAME=toothseg bash setup_env.sh
#
# 目标：RTX PRO 6000 Blackwell (sm_120)，需 PyTorch >= 2.7 且 CUDA >= 12.8。

set -euo pipefail

# ------------------------- 可配置项 -------------------------
ENV_NAME="${ENV_NAME:-toothseg}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA="${CUDA:-cu130}"                              # cu130 或 cu128
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
MIRROR="${MIRROR:-none}"                           # none | cn | tuna | aliyun
INSTALL_EXTERNAL=0
INSTALL_PRETRAIN=0

# ------------------------- 解析参数 -------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --external) INSTALL_EXTERNAL=1 ;;
    --pretrain) INSTALL_PRETRAIN=1 ;;
    --mirror) MIRROR="$2"; shift ;;
    --mirror=*) MIRROR="${1#*=}" ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
  shift
done

case "$MIRROR" in
  none|cn|tuna|aliyun) ;;
  *) echo "无效 --mirror 值: $MIRROR（可选 none/cn/tuna/aliyun）" >&2; exit 1 ;;
esac

# 本脚本所在目录（即 lab_train 根目录）
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------- 镜像源映射 -------------------------
case "$MIRROR" in
  none)
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    CONDA_CHANNEL=""
    PIP_INDEX="https://pypi.org/simple"
    PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA}"
    ;;
  cn)
    MINICONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    CONDA_CHANNEL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
    PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
    PYTORCH_INDEX="https://mirrors.aliyun.com/pytorch-wheels/${CUDA}"
    ;;
  tuna)
    MINICONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    CONDA_CHANNEL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
    PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
    PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA}"
    ;;
  aliyun)
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    CONDA_CHANNEL=""
    PIP_INDEX="https://mirrors.aliyun.com/pypi/simple"
    PYTORCH_INDEX="https://mirrors.aliyun.com/pytorch-wheels/${CUDA}"
    ;;
esac

echo "======================================================"
echo " lab_train 环境安装"
echo "  环境名       : ${ENV_NAME}"
echo "  Python       : ${PYTHON_VERSION}"
echo "  CUDA wheel   : ${CUDA}"
echo "  镜像         : ${MIRROR}（--mirror 可切换 none/cn/tuna/aliyun）"
echo "  PyTorch 源   : ${PYTORCH_INDEX}"
echo "  项目目录     : ${PACKAGE_ROOT}"
echo "======================================================"

# ------------------------- 1. Miniconda -------------------------
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [ -x "${MINICONDA_DIR}/bin/conda" ]; then
  CONDA_BIN="${MINICONDA_DIR}/bin/conda"
else
  echo "[1/5] 未找到 conda，开始安装 Miniconda 到 ${MINICONDA_DIR} ..."
  echo "      下载: ${MINICONDA_URL}"
  INSTALLER="/tmp/miniconda-$(date +%s).sh"
  wget -q "$MINICONDA_URL" -O "$INSTALLER"
  bash "$INSTALLER" -b -p "$MINICONDA_DIR"
  rm -f "$INSTALLER"
  CONDA_BIN="${MINICONDA_DIR}/bin/conda"
fi
echo "[1/5] conda: ${CONDA_BIN}"
# shellcheck disable=SC1090
source "$(dirname "$CONDA_BIN")/../etc/profile.d/conda.sh"

# ------------------------- 2. 创建/复用环境 -------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[2/5] 环境 ${ENV_NAME} 已存在，复用"
else
  echo "[2/5] 创建环境 ${ENV_NAME} (python=${PYTHON_VERSION}) ..."
  if [ -n "$CONDA_CHANNEL" ]; then
    conda create -c "$CONDA_CHANNEL" -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
  else
    conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
  fi
fi
conda activate "$ENV_NAME"

# ------------------------- 3. PyTorch (Blackwell / sm_120) -------------------------
echo "[3/5] 安装 PyTorch + torchvision（index: ${PYTORCH_INDEX}）..."
python -m pip install --upgrade pip -i "$PIP_INDEX" -q || python -m pip install --upgrade pip -q
if ! python -m pip install torch torchvision --index-url "$PYTORCH_INDEX"; then
  if [ "$MIRROR" != "none" ]; then
    echo "      镜像源安装 PyTorch 失败，回退官方源 https://download.pytorch.org/whl/${CUDA} ..."
    python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA}"
  else
    exit 1
  fi
fi

# ------------------------- 4. 项目依赖 -------------------------
echo "[4/5] 安装项目训练依赖 (pip install -e .[train]) ..."
python -m pip install -e ".[train]" -i "$PIP_INDEX"

if [ "$INSTALL_EXTERNAL" -eq 1 ]; then
  echo "[4/5] 额外安装 Transformers / RF-DETR 后端 (.[external]) ..."
  python -m pip install -e ".[external]" -i "$PIP_INDEX"
  python -m pip install rfdetr -i "$PIP_INDEX" || echo "警告: rfdetr 单独安装失败，请检查其官方安装方式" >&2
fi

if [ "$INSTALL_PRETRAIN" -eq 1 ]; then
  echo "[4/5] 额外安装 DINOv3 域内预训练依赖 (.[pretrain]) ..."
  python -m pip install -e ".[pretrain]" -i "$PIP_INDEX"
fi

# ------------------------- 5. 验证 -------------------------
echo "[5/5] 验证 GPU 与 sm_120 支持 ..."
python - <<'PY'
import torch
print("torch version :", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count  :", torch.cuda.device_count())
    print("device name   :", torch.cuda.get_device_name(0))
    print("capability    :", torch.cuda.get_device_capability(0))
    print("arch list     :", torch.cuda.get_arch_list())
    if "sm_120" not in torch.cuda.get_arch_list():
        raise SystemExit("错误: arch_list 不含 sm_120，PyTorch 缺 Blackwell 内核，请改用 cu130/cu128 重装。")
print("OK")
PY

echo ""
echo "======================================================"
echo " 安装完成。激活环境后即可训练："
echo "   conda activate ${ENV_NAME}"
echo "   python cli/main.py models"
echo " 训练示例（先冒烟 1 epoch）："
echo "   python cli/main.py train --data dataset/OMNI_COCO/after --model yolo11n.pt \\"
echo "     --epochs 1 --imgsz 1024 --batch 1 --device 0 --project output/smoke --name smoke-yolo11n"
echo " DINOv3 域内预训练（需先 --pretrain 装依赖，并准备数据清单）："
echo "   python pretrain/prepare_pretrain_data.py"
echo "   HF_TOKEN=... python pretrain/pretrain.py --config pretrain/config.example.json"
echo "======================================================"
