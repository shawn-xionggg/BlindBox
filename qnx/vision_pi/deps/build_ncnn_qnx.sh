#!/bin/sh
# Cross-compile ncnn (Tencent, tag 20261020) for QNX 8.0 aarch64 (Pi 5) from Windows, static, OpenMP, ARMv8.2 fp16.
set -e
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
export QNX_HOST=C:/Users/junio/qnx800/host/win64/x86_64
export QNX_TARGET=C:/Users/junio/qnx800/target/qnx
export PATH="/c/Users/junio/qnx800/host/win64/x86_64/usr/bin:/c/Users/junio/AppData/Roaming/Python/Python313/Scripts:$PATH"
cmake -S "$HERE/ncnn" -B "$HERE/ncnn-qnx-build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$HERE/qnx-win.toolchain.cmake" -DCMAKE_MAKE_PROGRAM=ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HERE/ncnn-qnx-install" \
  -DNCNN_SHARED_LIB=OFF -DNCNN_OPENMP=ON -DNCNN_THREADS=ON -DNCNN_VULKAN=OFF -DNCNN_SIMPLEOCV=OFF \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_PYTHON=OFF -DNCNN_ARM82=ON -DNCNN_ARM82DOT=ON -DNCNN_RUNTIME_CPU=ON \
  -DOpenMP_C_FLAGS=-fopenmp -DOpenMP_CXX_FLAGS=-fopenmp -DOpenMP_C_LIB_NAMES=gomp -DOpenMP_CXX_LIB_NAMES=gomp \
  -DOpenMP_gomp_LIBRARY=C:/Users/junio/qnx800/target/qnx/aarch64le/lib/gcc/12.2.0/libgomp.a
cmake --build "$HERE/ncnn-qnx-build" -j ${JOBS:-10}
cmake --install "$HERE/ncnn-qnx-build"
echo "ncnn for QNX installed in $HERE/ncnn-qnx-install"
