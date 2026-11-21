
#include <stdio.h>
#include <stdlib.h>
#include "pose_decode.h"
int main(int argc, char **argv) {
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb"); if (!f) return 2;
    int frames, n, w, h; static float out[56 * 2100]; static int scratch[2 * 2100];
    if (fread(&frames, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) return 3;
    for (int i = 0; i < frames; i++) {
        if (fread(&w, 4, 1, f) != 1 || fread(&h, 4, 1, f) != 1) return 4;
        if (fread(out, 4, (size_t)(56 * n), f) != (size_t)(56 * n)) return 5;
        pose_person_t p;
        letterbox_t lb = letterbox_params(w, h, 320);
        if (!pose_decode_largest(out, n, lb, 0.25f, 0.7f, scratch, &p)) { printf("none\n"); continue; }
        for (int k = 0; k < POSE_KPTS; k++) printf("%.4f %.4f %.4f ", p.kpt[k][0], p.kpt[k][1], p.kconf[k]);
        printf("\n");
    }
    fclose(f);
    return 0;
}
