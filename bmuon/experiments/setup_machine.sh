#!/usr/bin/env bash
# Install everything needed to run BMuon on this machine, in /tmp/$USER (local disk):
# the Python environment, the OpenEquivariance kernels and the MPtrj data.
#   bash bmuon/experiments/setup_machine.sh            # everything (~10 min)
#   bash bmuon/experiments/setup_machine.sh --no-data  # environment only
# Idempotent: what is already there is skipped; rerun it after a reboot that wiped /tmp.
#
# The lab machines have no full CUDA toolkit, which the OpenEquivariance JAX extension needs
# to compile. We assemble a minimal CUDA 12 root (headers + libcudart + libnvrtc) from the
# NVIDIA pip wheels already installed with JAX, plus the driver's libcuda.
set -euo pipefail
USER="${USER:-$(id -un)}"
cd "$(dirname "$0")/../.."

export PATH="$HOME/.local/bin:$PATH"
# Lab homes are quota-capped; keep compiler/uv caches on the local disk.
export CCACHE_DISABLE=1
export CCACHE_DIR="${CCACHE_DIR:-/tmp/$USER/ccache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/$USER/xdg-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/tmp/$USER/uv-python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/nequix-venv}"
CUDA_ROOT="/tmp/$USER/cuda12"
DATA="${BMUON_DATA:-/tmp/$USER/bmuon-data}"
RAW="/tmp/$USER/raw"
ZIP="2023-11-22-mp-trj-extxyz-by-yuan.zip"
WITH_DATA=1
[ "${1:-}" = "--no-data" ] && WITH_DATA=0
mkdir -p "/tmp/$USER"

if [ -f "$UV_PROJECT_ENVIRONMENT/.bmuon_ready" ]; then
  echo "== python environment already installed ($UV_PROJECT_ENVIRONMENT)"
  # shellcheck disable=SC1091
  source "$UV_PROJECT_ENVIRONMENT/bin/activate"
else

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "== python environment ($UV_PROJECT_ENVIRONMENT)"
uv sync --extra oeq
# shellcheck disable=SC1091
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
uv pip install "nvidia-cuda-nvrtc-cu12>=12.8,<13" "nvidia-cuda-nvcc-cu12>=12.8,<13"

echo "== minimal CUDA 12 root ($CUDA_ROOT)"
SITE="$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__path__[0]))')/nvidia"
rm -rf "$CUDA_ROOT"
mkdir -p "$CUDA_ROOT/include" "$CUDA_ROOT/lib64" "$CUDA_ROOT/bin"
for d in cuda_runtime cuda_nvrtc cuda_nvcc; do
  if [ -d "$SITE/$d/include" ]; then
    for f in "$SITE/$d/include"/*; do ln -sfn "$f" "$CUDA_ROOT/include/"; done
  fi
done
ln -sf "$SITE/cuda_runtime/lib/libcudart.so.12" "$CUDA_ROOT/lib64/libcudart.so"
ln -sf "$SITE/cuda_runtime/lib/libcudart.so.12" "$CUDA_ROOT/lib64/libcudart.so.12"
ln -sf "$SITE/cuda_nvrtc/lib/libnvrtc.so.12" "$CUDA_ROOT/lib64/libnvrtc.so"
ln -sf "$SITE/cuda_nvrtc/lib/libnvrtc.so.12" "$CUDA_ROOT/lib64/libnvrtc.so.12"
for f in "$SITE"/cuda_nvrtc/lib/libnvrtc-builtins*; do ln -sf "$f" "$CUDA_ROOT/lib64/"; done
DRV="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so(\.1)? /{print $NF; exit}')"
[ -z "$DRV" ] && DRV="$(ls /usr/lib64/libcuda.so* /usr/lib/x86_64-linux-gnu/libcuda.so* 2>/dev/null | head -1 || true)"
if [ -z "$DRV" ]; then echo "NVIDIA driver library libcuda not found" >&2; exit 1; fi
ln -sf "$DRV" "$CUDA_ROOT/lib64/libcuda.so"
echo '{"cuda": {"version": "12.8.0"}}' > "$CUDA_ROOT/version.json"
# Stand-in nvcc: CMake's FindCUDAToolkit locates the toolkit from nvcc and would otherwise pick
# the incomplete system CUDA 13 (/usr/local/bin/nvcc). Nothing is compiled with it.
cat > "$CUDA_ROOT/bin/nvcc" <<'NVCC'
#!/bin/sh
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release 12.8, V12.8.93"
NVCC
chmod +x "$CUDA_ROOT/bin/nvcc"

echo "== OpenEquivariance JAX extension"
CUDAToolkit_ROOT="$CUDA_ROOT" \
CMAKE_ARGS="-DCUDAToolkit_ROOT=$CUDA_ROOT -DCUDAToolkit_NVCC_EXECUTABLE=$CUDA_ROOT/bin/nvcc" \
  uv pip install --no-cache --no-build-isolation --reinstall "openequivariance_extjax==0.7.0"

echo "== checks"
python -c "import jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'JAX does not see the GPU'"
# openequivariance imports torch unless OEQ_NOTORCH is set (Nequix sets it the same way)
OEQ_NOTORCH=1 python -c "import openequivariance, openequivariance_extjax; print('OpenEquivariance OK')"
touch "$UV_PROJECT_ENVIRONMENT/.bmuon_ready"
fi

if [ "$WITH_DATA" = 1 ]; then
  if [ -f "$DATA/mptrj-aselmdb/.complete" ]; then
    echo "== MPtrj already prepared ($DATA/mptrj-aselmdb)"
  else
    echo "== MPtrj: download, unzip, preprocess into $DATA/mptrj-aselmdb (~5-10 min)"
    mkdir -p "$RAW"
    if [ ! -f "$RAW/$ZIP" ]; then
      wget -q --show-progress -O "$RAW/$ZIP.part" https://ndownloader.figshare.com/files/43302033
      mv "$RAW/$ZIP.part" "$RAW/$ZIP"
    fi
    rm -rf "$RAW/mptrj-gga-ggapu"
    unzip -q "$RAW/$ZIP" -d "$RAW" -x "__MACOSX/*"
    rm -rf "$DATA/mptrj-aselmdb"
    python scripts/preprocess_data.py "$RAW/mptrj-gga-ggapu" "$DATA/mptrj-aselmdb"
    n=$(ls "$DATA"/mptrj-aselmdb/*.aselmdb 2>/dev/null | wc -l)
    if [ "$n" -ne 16 ]; then echo "preprocessing produced $n files, expected 16" >&2; exit 1; fi
    touch "$DATA/mptrj-aselmdb/.complete"
    rm -rf "$RAW/mptrj-gga-ggapu"   # keep only the zip (1.6 GB)
  fi
  du -sh "$DATA/mptrj-aselmdb"
fi
echo "== ready: source $UV_PROJECT_ENVIRONMENT/bin/activate"
