#!/bin/sh
# Cross-compile a minimal static OpenCV (core, imgproc, video, dnn) for QNX 8.0 aarch64 (Pi 5)
# from Windows, using QNX's own OpenCV port (qnx-ports/opencv, branch qnx-4.12.0) and their
# toolchain file (qnx-ports/build-files/ports/opencv). Output: ./opencv-qnx-install
set -e
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
export QNX_HOST=C:/Users/junio/qnx800/host/win64/x86_64
export QNX_TARGET=C:/Users/junio/qnx800/target/qnx
export PATH="/c/Users/junio/qnx800/host/win64/x86_64/usr/bin:/c/Users/junio/AppData/Roaming/Python/Python313/Scripts:$PATH"
JOBS=${JOBS:-10}

cmake -S "$HERE/opencv" -B "$HERE/opencv-qnx-build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$HERE/qnx-win.toolchain.cmake" \
  -DCMAKE_MAKE_PROGRAM=ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HERE/opencv-qnx-install" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_LIST=core,imgproc,video,dnn \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_JAVA=OFF \
  -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_EIGEN=OFF \
  -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF \
  -DWITH_OPENEXR=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_ADE=OFF -DWITH_LAPACK=OFF \
  -DWITH_PROTOBUF=ON -DBUILD_PROTOBUF=ON -DOPENCV_DNN_OPENCL=OFF \
  -DCV_ENABLE_INTRINSICS=ON -DENABLE_NEON=ON \
  -DWITH_PTHREADS_PF=ON

cmake --build "$HERE/opencv-qnx-build" -j "$JOBS"
cmake --install "$HERE/opencv-qnx-build"
echo "OpenCV for QNX installed in $HERE/opencv-qnx-install"
