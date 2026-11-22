# QNX 8.0 aarch64 toolchain for building on a WINDOWS host.
# Same as qnx-ports/build-files/ports/opencv/qnx.nto.toolchain.cmake, minus Python, with .exe tools.
set(QNX_HOST "$ENV{QNX_HOST}")
set(QNX_TARGET "$ENV{QNX_TARGET}")
set(CMAKE_SYSTEM_NAME QNX)
set(CMAKE_SYSTEM_PROCESSOR aarch64le)

set(CMAKE_C_COMPILER "${QNX_HOST}/usr/bin/qcc.exe")
set(CMAKE_CXX_COMPILER "${QNX_HOST}/usr/bin/q++.exe")
set(CMAKE_AR "${QNX_HOST}/usr/bin/ntoaarch64-ar.exe" CACHE FILEPATH "archiver")
set(CMAKE_RANLIB "${QNX_HOST}/usr/bin/ntoaarch64-ranlib.exe" CACHE FILEPATH "ranlib")
set(CMAKE_STRIP "${QNX_HOST}/usr/bin/ntoaarch64-strip.exe" CACHE FILEPATH "strip")

set(CMAKE_C_FLAGS_INIT "-Vgcc_ntoaarch64le -D_QNX_SOURCE")
set(CMAKE_CXX_FLAGS_INIT "-Vgcc_ntoaarch64le_cxx -D_QNX_SOURCE")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_FIND_ROOT_PATH "${QNX_TARGET};${QNX_TARGET}/aarch64le")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
