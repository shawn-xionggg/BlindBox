# Native MSYS2 UCRT: mingw32-make (or MSYS make if runnable on this host).
# QNX: make TARGET=qnx CC=qcc
# Explicit SDP 8 aarch64 variant:
# make TARGET=qnx CC="qcc -Vgcc_ntoaarch64le"
# Cross builds from Windows: add EXE= (target files are not Windows .exe).
CC = gcc
TARGET ?= portable
CFLAGS ?= -O2 -g
CFLAGS += -std=c99 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wpedantic -Werror
CPPFLAGS += -Icommon
EXE :=
ifeq ($(TARGET),qnx)
IPC = common/ipc_qnx.c
CPPFLAGS += -D_QNX_SOURCE
# QNX provides pthread APIs in libc; no separate pthread library needed.
else
IPC = common/ipc_posix.c
ifeq ($(OS),Windows_NT)
EXE := .exe
LDLIBS += -lws2_32
endif
endif

PROGRAMS = bin/cue_driver$(EXE) bin/vision_stub$(EXE) bin/supervisor$(EXE)
HEADERS = common/protocol.h common/ipc.h common/runtime.h

.PHONY: all clean test
all: $(PROGRAMS)

# bin/ is checked in through .gitkeep; no dependency on a particular shell.
bin/cue_driver$(EXE): cue_driver/cue_driver.c $(IPC) $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) cue_driver/cue_driver.c $(IPC) -o $@ $(LDFLAGS) $(LDLIBS)

bin/vision_stub$(EXE): vision_stub/vision_stub.c $(IPC) $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) vision_stub/vision_stub.c $(IPC) -o $@ $(LDFLAGS) $(LDLIBS)

bin/supervisor$(EXE): supervisor/supervisor.c common/runtime.h
	$(CC) $(CPPFLAGS) $(CFLAGS) supervisor/supervisor.c -o $@ $(LDFLAGS) $(LDLIBS)

test:
	python test_qnx_skeleton.py

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('bin').iterdir() if p.is_file() and p.name != '.gitkeep']"
