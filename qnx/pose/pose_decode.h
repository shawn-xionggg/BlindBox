/* Decode yolo11n-pose ONNX output into the largest person's 17 keypoints.
 *
 * Model output (exported by ultralytics, imgsz 320): float32 [1, 56, N],
 * N = 2100 anchors, channel-major. Per anchor: cx, cy, w, h (input pixels),
 * person score (already sigmoid), then 17 x (x, y, visibility(sigmoid)).
 *
 * Steps (same as ultralytics predict): keep score > conf_thres, sort by score,
 * greedy NMS at IoU > iou_thres, then pick the LARGEST box (the attacker, as
 * in pose_features.PoseRunner) and undo the letterbox: x = (x - pad_x) / gain.
 *
 * Plain C99, no allocation beyond the caller's scratch buffer; builds on QNX.
 */
#ifndef POSE_DECODE_H
#define POSE_DECODE_H

#define POSE_KPTS 17
#define POSE_CH (5 + 3 * POSE_KPTS)   /* 56 */

typedef struct {
    float box[4];                  /* x1, y1, x2, y2 in original-image pixels */
    float score;
    float kpt[POSE_KPTS][2];       /* original-image pixels */
    float kconf[POSE_KPTS];        /* 0..1 */
} pose_person_t;

/* Letterbox geometry used to build the network input. */
typedef struct { float gain, pad_x, pad_y; int w, h; } letterbox_t;  /* w,h: original image */

/* Letterbox for an image of w x h into a square of side `size` (ultralytics
 * LetterBox with auto=False, center=True). */
letterbox_t letterbox_params(int w, int h, int size);

/* out: model output [POSE_CH * n]; scratch: at least 2*n ints.
 * Returns 1 and fills *best if a person was found, else 0. */
int pose_decode_largest(const float *out, int n, letterbox_t lb,
                        float conf_thres, float iou_thres,
                        int *scratch, pose_person_t *best);

#endif
