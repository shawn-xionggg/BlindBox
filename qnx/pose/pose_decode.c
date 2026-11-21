#include "pose_decode.h"
#include <math.h>
#include <stdlib.h>

letterbox_t letterbox_params(int w, int h, int size)
{
    letterbox_t lb;
    float r = fminf((float)size / (float)h, (float)size / (float)w);
    int new_w = (int)lroundf((float)w * r), new_h = (int)lroundf((float)h * r);
    float dw = (float)(size - new_w) / 2.0f, dh = (float)(size - new_h) / 2.0f;
    lb.gain = r;
    /* ultralytics: left = round(dw - 0.1), top = round(dh - 0.1) */
    lb.pad_x = (float)lroundf(dw - 0.1f);
    lb.pad_y = (float)lroundf(dh - 0.1f);
    lb.w = w;
    lb.h = h;
    return lb;
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* sort indices by score, descending (insertion sort: candidates are few) */
static void sort_by_score(const float *score, int *idx, int m)
{
    for (int i = 1; i < m; i++) {
        int v = idx[i], j = i - 1;
        while (j >= 0 && score[idx[j]] < score[v]) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = v;
    }
}

static void xyxy(const float *out, int n, int a, float b[4])
{
    float cx = out[0 * n + a], cy = out[1 * n + a], w = out[2 * n + a], h = out[3 * n + a];
    b[0] = cx - w / 2; b[1] = cy - h / 2; b[2] = cx + w / 2; b[3] = cy + h / 2;
}

static float iou(const float a[4], const float b[4])
{
    float x1 = fmaxf(a[0], b[0]), y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[2], b[2]), y2 = fminf(a[3], b[3]);
    float inter = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return ua > 0 ? inter / ua : 0.0f;
}

int pose_decode_largest(const float *out, int n, letterbox_t lb,
                        float conf_thres, float iou_thres,
                        int *scratch, pose_person_t *best)
{
    const float *score = out + 4 * n;
    int *cand = scratch, *keep = scratch + n;
    int m = 0, k = 0;
    for (int a = 0; a < n; a++)
        if (score[a] > conf_thres) cand[m++] = a;
    if (m == 0) return 0;
    sort_by_score(score, cand, m);
    for (int i = 0; i < m && k < 300; i++) {          /* greedy NMS, max_det 300 */
        float bi[4], bj[4];
        int ok = 1;
        xyxy(out, n, cand[i], bi);
        for (int j = 0; j < k; j++) {
            xyxy(out, n, keep[j], bj);
            if (iou(bi, bj) > iou_thres) { ok = 0; break; }
        }
        if (ok) keep[k++] = cand[i];
    }
    int pick = -1;
    float best_area = -1.0f;
    for (int j = 0; j < k; j++) {                     /* largest box = the attacker */
        float b[4];
        xyxy(out, n, keep[j], b);
        float area = (b[2] - b[0]) * (b[3] - b[1]);
        if (area > best_area) { best_area = area; pick = keep[j]; }
    }
    float b[4];
    xyxy(out, n, pick, b);
    /* undo letterbox, then clip to the image like ultralytics' scale_boxes/scale_coords */
    best->box[0] = clampf((b[0] - lb.pad_x) / lb.gain, 0, (float)lb.w);
    best->box[1] = clampf((b[1] - lb.pad_y) / lb.gain, 0, (float)lb.h);
    best->box[2] = clampf((b[2] - lb.pad_x) / lb.gain, 0, (float)lb.w);
    best->box[3] = clampf((b[3] - lb.pad_y) / lb.gain, 0, (float)lb.h);
    best->score = score[pick];
    for (int p = 0; p < POSE_KPTS; p++) {
        best->kpt[p][0] = clampf((out[(5 + 3 * p) * n + pick] - lb.pad_x) / lb.gain, 0, (float)lb.w);
        best->kpt[p][1] = clampf((out[(6 + 3 * p) * n + pick] - lb.pad_y) / lb.gain, 0, (float)lb.h);
        best->kconf[p] = out[(7 + 3 * p) * n + pick];
    }
    return 1;
}
