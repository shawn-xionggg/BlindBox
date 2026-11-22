#!/bin/sh
# Builds qnx/build_aarch64/vision (QNX 8.0, aarch64, Raspberry Pi 5) from Windows Git Bash.
# Needs (scripts in qnx/vision_pi/deps, sources/outputs in ../qnx-build next to the repo):
#   - static OpenCV from QNX's port (qnx-ports/opencv qnx-4.12.0): build_opencv_qnx.sh
#   - static ncnn (Tencent 20261020, patched for QNX CPU detection): cd ncnn && git apply deps/ncnn-qnx.patch, then deps/build_ncnn_qnx.sh
#   - libcamapi.so copied from the Pi (/usr/lib/libcamapi.so.1) into ../qnx-build/pi-libs
#   - QNX SDP 8.0 Sensor Framework Base Support (camera headers) from the QNX Software Center
set -e
cd "$(dirname "$0")/.."
export QNX_HOST=C:/Users/junio/qnx800/host/win64/x86_64
export QNX_TARGET=C:/Users/junio/qnx800/target/qnx
export PATH="/c/Users/junio/qnx800/host/win64/x86_64/usr/bin:$PATH"
CV=${OPENCV_QNX:-../../qnx-build/opencv-qnx-install}
NC=${NCNN_QNX:-../../qnx-build/ncnn-qnx-install}
PILIBS=${PI_LIBS:-../../qnx-build/pi-libs}
CVLIB="$CV/lib"
[ -d "$CV/lib/opencv4/3rdparty" ] && CV3="$CV/lib/opencv4/3rdparty" || CV3="$CVLIB"

qcc -Vgcc_ntoaarch64le -O2 -Wall -std=c99 -c pose/pose_decode.c -o build_aarch64/pose_decode.o
qcc -Vgcc_ntoaarch64le -O2 -Wall -std=c99 -c model/early_model.c -o build_aarch64/early_model.o
qcc -Vgcc_ntoaarch64le -O2 -Wall -Wno-unused-function -DHAT_LIB -c hat_usb/hat_usb.c -o build_aarch64/hat_lib.o
q++ -Vgcc_ntoaarch64le_cxx -O2 -Wall -std=c++17 -D_QNX_SOURCE -DHAVE_NCNN -DHAVE_CAMAPI -fopenmp \
    -I"$CV/include/opencv4" -I"$NC/include" -c vision_pi/vision.cpp -o build_aarch64/vision.o
q++ -Vgcc_ntoaarch64le_cxx -fopenmp -o build_aarch64/vision \
    build_aarch64/vision.o build_aarch64/pose_decode.o build_aarch64/early_model.o build_aarch64/hat_lib.o \
    -L"$CVLIB" -L"$CV3" -L"$NC/lib" -L"$PILIBS" \
    -lncnn -lopencv_dnn -lopencv_video -lopencv_imgproc -lopencv_core -llibprotobuf -ltegra_hal \
    -lcamapi -lgomp -lusbdi -lz -lm -Wl,--allow-shlib-undefined
rm -f build_aarch64/*.o
ls -l build_aarch64/vision
