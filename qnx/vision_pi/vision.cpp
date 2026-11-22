// BlindBox vision on QNX (Raspberry Pi 5): camera -> pose -> features -> learned early detector -> hat.
//
// A line-for-line C++ port of early_live.py (_FeatureState, feature_matrix, EarlyDetector.update)
// and launch_detector.py (LaunchDetector.last_disp), using:
//   - OpenCV (QNX's port, qnx-ports/opencv qnx-4.12.0): Farneback flow, blur, resize, DNN for pose
//   - qnx/pose/pose_decode.c: yolo11n-pose output -> largest person's keypoints (verified vs ultralytics)
//   - qnx/model/early_model.c: the trained detector exported to C (verified vs scikit-learn)
//   - qnx/hat_usb/hat_usb.c: our libusbdi driver for the ESP32 buzzer hat
//
// Frame sources:
//   --frames FILE   recorded test frames (make_test_frames.py); --file-pose uses the keypoints stored
//                   in the file instead of running pose, to check features/model against Python exactly
//   --camera N      live camera unit N through QNX's camera API (needs HAVE_CAMAPI build)
// Output: --csv FILE writes per-frame features/scores (compare_csv.py checks them against Python).
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <string>
#include <vector>
#include <time.h>
#include <pthread.h>
#include <sched.h>

extern "C" {
#include "../pose/pose_decode.h"
#include "../model/early_model.h"
#include "../hat_usb/hat.h"
}

static const double NaN = std::numeric_limits<double>::quiet_NaN();
static const int L_SH = 5, R_SH = 6, WR[2] = {9, 10};
static const double KC = 0.35, SCALE = EARLY_FLOW_SCALE, COOLDOWN = 0.6, UPPER_DROP = 0.10;
static const int WIN = 12;   // early_live: max(3, round(0.6 / dt_est)), dt_est = 0.048 -> round(12.5) = 12

static double now_s() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

struct V2 { double x, y; };
static inline V2 sub(V2 a, V2 b) { return {a.x - b.x, a.y - b.y}; }
static inline V2 scl(V2 a, double s) { return {a.x * s, a.y * s}; }
static inline double norm(V2 a) { return std::hypot(a.x, a.y); }
static inline bool fin(V2 a) { return std::isfinite(a.x) && std::isfinite(a.y); }

static double median(std::vector<double> v) {   // np.median (even count -> mean of the middle two)
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static V2 median2(const std::vector<V2> &p) {    // np.median(..., axis=0)
    std::vector<double> xs, ys;
    for (auto &q : p) { xs.push_back(q.x); ys.push_back(q.y); }
    return {median(xs), median(ys)};
}

// ---------------- launch_detector.LaunchDetector (thresh = 1e9: only last_disp is used) ----------------
struct LaunchWrist {
    struct H { double t; V2 p; double speed; };
    std::deque<H> hist;
    bool has_prev = false; double prev_t = 0; V2 prev_p{0, 0};
    int run = 0; double run_start = 0;
    bool armed = false; V2 armed_rest{0, 0}; double armed_t = 0;
    bool has_leave = false; double leave_t = 0;
};
struct Launch {
    const double rest_r = 0.15, calm = 1.5, hold = 0.15, after = 0.30, window = 0.6, kconf = 0.35;
    LaunchWrist w[2];
    double last_disp[2] = {0, 0};
    void update(double t, const V2 *kp, const double *conf) {
        last_disp[0] = last_disp[1] = 0.0;
        if (conf[L_SH] < kconf || conf[R_SH] < kconf) return;
        double width = norm(sub(kp[R_SH], kp[L_SH]));
        if (!std::isfinite(width) || width <= 10) return;
        V2 mid = scl({kp[L_SH].x + kp[R_SH].x, kp[L_SH].y + kp[R_SH].y}, 0.5);
        for (int k = 0; k < 2; k++) {
            LaunchWrist &s = w[k];
            int wi = WR[k];
            if (conf[wi] < kconf || !fin(kp[wi])) { s.run = 0; continue; }
            V2 p = scl(sub(kp[wi], mid), 1.0 / width);
            double speed = 99.0, frame_dt = 0.0;
            if (s.has_prev && t > s.prev_t && t - s.prev_t < 0.25) {
                frame_dt = t - s.prev_t;
                speed = norm(sub(p, s.prev_p)) / frame_dt;
            }
            s.has_prev = true; s.prev_t = t; s.prev_p = p;
            while (!s.hist.empty() && t - s.hist.front().t > window) s.hist.pop_front();
            std::vector<V2> calm_pts;
            for (auto &h : s.hist) if (h.speed < calm) calm_pts.push_back(h.p);
            bool has_rest = calm_pts.size() >= 3;
            V2 rest = has_rest ? median2(calm_pts) : V2{0, 0};
            s.hist.push_back({t, p, speed});
            double disp = has_rest ? norm(sub(p, rest)) : INFINITY;
            bool at_rest = disp < rest_r && speed < calm * 1.5;
            if (at_rest) {
                if (s.run == 0) s.run_start = t;
                s.run++;
                if ((t - s.run_start) + frame_dt >= hold - 1e-9 && s.run >= 2) {
                    s.armed = true; s.armed_rest = rest; s.armed_t = t; s.has_leave = false;
                }
                continue;
            }
            s.run = 0;
            if (!s.armed || t - s.armed_t > after) continue;
            if (!s.has_leave) { s.has_leave = true; s.leave_t = t; }
            last_disp[k] = norm(sub(p, s.armed_rest));
        }
    }
};

// ---------------- early_live._FeatureState ----------------
struct Row {
    double disp[2], launch[2], speed[2], relx[2], rely[2], energy[2], div_rel[2], mag_rel[2], out_rel[2];
    double torso_div, torso_mag, width_rate;
    bool sh;
    bool has_rel[2];
    V2 rel[2];
};

struct FeatureState {
    struct RelH { bool has; V2 rel; bool valid; double speed; };
    bool has_prev_t = false; double prev_t = 0;
    bool has_prev_rel[2] = {false, false}; V2 prev_rel[2];
    std::deque<RelH> rel_hist[2];
    std::deque<double> e_hist[2];
    cv::Mat prev_gray_full, prev_gray_half;
    bool has_last_pix[2] = {false, false}, has_last_pix_half[2] = {false, false};
    V2 last_pix[2], last_pix_half[2];
    std::vector<double> widths_ok;
    struct WH { double t, width; bool sh; };
    std::deque<WH> width_hist;
    std::deque<Row> rows;
    Launch launch;

    // mean of a[y0:y1, x0:x1] for a box of half-size r around (cx, cy), like early_live's bm()
    static bool box_rect(int W, int H, double cx, double cy, double r, cv::Rect &out) {
        int x0 = (int)std::max(0.0, cx - r), x1 = (int)std::min((double)W, cx + r + 1);
        int y0 = (int)std::max(0.0, cy - r), y1 = (int)std::min((double)H, cy + r + 1);
        if (x1 - x0 < 3 || y1 - y0 < 3) return false;
        out = cv::Rect(x0, y0, x1 - x0, y1 - y0);
        return true;
    }

    const Row &base(double t, const cv::Mat &frame, const V2 *kp, const double *conf) {
        Row f;
        double width = (fin(kp[L_SH]) && fin(kp[R_SH])) ? norm(sub(kp[R_SH], kp[L_SH])) : NaN;
        bool sh = conf[L_SH] > KC && conf[R_SH] > KC && std::isfinite(width) && width > 10;
        V2 mid = scl({kp[L_SH].x + kp[R_SH].x, kp[L_SH].y + kp[R_SH].y}, 0.5);
        double dt = has_prev_t ? t - prev_t : NaN;
        if (sh) {
            widths_ok.push_back(width);
            if (widths_ok.size() > 2000) widths_ok.erase(widths_ok.begin(), widths_ok.end() - 2000);
        }
        V2 rel[2] = {{0, 0}, {0, 0}};
        bool wok[2];
        double speed[2] = {NaN, NaN};
        for (int k = 0; k < 2; k++) {
            wok[k] = sh && conf[WR[k]] > KC;
            if (wok[k]) rel[k] = scl(sub(kp[WR[k]], mid), 1.0 / width);
            if (wok[k] && has_prev_rel[k] && std::isfinite(dt)) speed[k] = norm(sub(rel[k], prev_rel[k])) / dt;
        }
        // disp from rest: median of calm valid frames among the previous WIN frames
        for (int k = 0; k < 2; k++) {
            f.disp[k] = NaN;
            if (!wok[k]) continue;
            std::vector<V2> calm;
            for (auto &h : rel_hist[k])
                if (h.valid && (std::isfinite(h.speed) ? h.speed : 99.0) < 1.5) calm.push_back(h.rel);
            if (calm.size() >= 3) f.disp[k] = norm(sub(rel[k], median2(calm)));
        }
        launch.update(t, kp, conf);
        f.launch[0] = launch.last_disp[0]; f.launch[1] = launch.last_disp[1];

        // motion energy: full-res blurred gray diff around the last known wrist pixel
        cv::Mat g0, gray;
        cv::cvtColor(frame, g0, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(g0, gray, cv::Size(5, 5), 0);
        for (int k = 0; k < 2; k++)
            if (conf[WR[k]] > KC && fin(kp[WR[k]])) { has_last_pix[k] = true; last_pix[k] = kp[WR[k]]; }
        double box = (sh ? width : (widths_ok.empty() ? NaN : median(widths_ok))) * 0.5;
        double raw_e[2] = {NaN, NaN};
        if (!prev_gray_full.empty() && std::isfinite(box)) {
            cv::Mat diff;
            cv::absdiff(gray, prev_gray_full, diff);
            int H = gray.rows, W = gray.cols;
            for (int k = 0; k < 2; k++) {
                if (!sh || !has_last_pix[k]) continue;
                double x = last_pix[k].x, y = last_pix[k].y;
                int r = (int)std::max(20.0, box / 2);
                int x0 = (int)std::max(0.0, x - r), x1 = (int)std::min((double)W, x + r);
                int y0 = (int)std::max(0.0, y - r), y1 = (int)std::min((double)H, y + r);
                if (x1 > x0 && y1 > y0) raw_e[k] = cv::mean(diff(cv::Rect(x0, y0, x1 - x0, y1 - y0)))[0];
            }
        }
        for (int k = 0; k < 2; k++) {
            f.energy[k] = NaN;
            std::vector<double> past;
            for (double v : e_hist[k]) if (std::isfinite(v)) past.push_back(v);
            if (past.size() >= 3 && std::isfinite(raw_e[k])) f.energy[k] = raw_e[k] / (median(past) + 1e-3);
        }
        prev_gray_full = gray;

        // optical flow at SCALE (Farneback), same parameters as training
        cv::Mat small, g;
        cv::resize(frame, small, cv::Size(), SCALE, SCALE, cv::INTER_AREA);
        cv::cvtColor(small, g, cv::COLOR_BGR2GRAY);
        for (int k = 0; k < 2; k++)
            if (conf[WR[k]] > KC && fin(kp[WR[k]])) { has_last_pix_half[k] = true; last_pix_half[k] = scl(kp[WR[k]], SCALE); }
        for (int k = 0; k < 2; k++) f.div_rel[k] = f.mag_rel[k] = f.out_rel[k] = NaN;
        f.torso_div = f.torso_mag = NaN;
        bool sh_conf = conf[L_SH] > KC && conf[R_SH] > KC;
        if (!prev_gray_half.empty() && sh_conf && std::isfinite(dt) && dt > 0 && std::isfinite(width)) {
            cv::Mat flow;
            cv::calcOpticalFlowFarneback(prev_gray_half, g, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            std::vector<cv::Mat> uv;
            cv::split(flow, uv);
            const cv::Mat &u = uv[0], &v = uv[1];
            int H = g.rows, W = g.cols;
            // dv = np.gradient(u, axis=1) + np.gradient(v, axis=0)  (central inside, one-sided at edges)
            cv::Mat dvm(H, W, CV_32F);
            for (int y = 0; y < H; y++) {
                const float *ur = u.ptr<float>(y);
                const float *vp = v.ptr<float>(y > 0 ? y - 1 : 0), *vc = v.ptr<float>(y), *vn = v.ptr<float>(y < H - 1 ? y + 1 : H - 1);
                float *o = dvm.ptr<float>(y);
                for (int x = 0; x < W; x++) {
                    float gx = (x == 0) ? ur[1] - ur[0] : (x == W - 1) ? ur[W - 1] - ur[W - 2] : 0.5f * (ur[x + 1] - ur[x - 1]);
                    float gy = (y == 0) ? vn[x] - vc[x] : (y == H - 1) ? vc[x] - vp[x] : 0.5f * (vn[x] - vp[x]);
                    o[x] = gx + gy;
                }
            }
            double wh = width * SCALE;
            if (wh > 5) {
                V2 midh = scl(mid, SCALE);
                double tr = 0.35 * wh, tcy = midh.y + 0.6 * wh, tfu = 0, tfv = 0;
                cv::Rect tb;
                if (box_rect(W, H, midh.x, tcy, tr, tb)) {
                    tfu = cv::mean(u(tb))[0];
                    tfv = cv::mean(v(tb))[0];
                    f.torso_div = cv::mean(dvm(tb))[0] / dt;
                    f.torso_mag = std::hypot(tfu, tfv) / wh / dt;
                } else {
                    f.torso_div = 0.0;
                }
                double r = 0.35 * wh;
                for (int k = 0; k < 2; k++) {
                    if (!has_last_pix_half[k]) continue;
                    double cx = last_pix_half[k].x, cy = last_pix_half[k].y;
                    cv::Rect b;
                    if (!box_rect(W, H, cx, cy, r, b)) continue;
                    V2 d = sub({cx, cy}, midh);
                    double dn = norm(d) + 1e-6;
                    d = scl(d, 1.0 / dn);
                    double sum_mag = 0, sum_out = 0;
                    for (int y = b.y; y < b.y + b.height; y++) {
                        const float *ur = u.ptr<float>(y), *vr = v.ptr<float>(y);
                        for (int x = b.x; x < b.x + b.width; x++) {
                            double du = ur[x] - tfu, dvv = vr[x] - tfv;
                            sum_mag += std::hypot(du, dvv);
                            sum_out += du * d.x + dvv * d.y;
                        }
                    }
                    double area = (double)b.area();
                    f.div_rel[k] = cv::mean(dvm(b))[0] / dt - f.torso_div;
                    f.mag_rel[k] = sum_mag / area / wh / dt;
                    f.out_rel[k] = sum_out / area / wh / dt;
                }
            }
        }
        prev_gray_half = g;

        // width rate over 2 frames (raw widths)
        width_hist.push_back({t, width, sh});
        if (width_hist.size() > 3) width_hist.pop_front();
        f.width_rate = NaN;
        if (width_hist.size() == 3 && sh) {
            double t2 = width_hist.front().t, w2 = width_hist.front().width;
            // Python: max(nan, 1) is nan, so a missing old width gives nan (fmax would give 1)
            if (t > t2 && std::isfinite(w2)) f.width_rate = (width - w2) / std::max(w2, 1.0) / (t - t2);
        }

        for (int k = 0; k < 2; k++) {
            rel_hist[k].push_back({wok[k], rel[k], wok[k], speed[k]});
            if ((int)rel_hist[k].size() > WIN) rel_hist[k].pop_front();
            e_hist[k].push_back(raw_e[k]);
            if ((int)e_hist[k].size() > WIN) e_hist[k].pop_front();
            has_prev_rel[k] = wok[k];
            if (wok[k]) prev_rel[k] = rel[k];
        }
        has_prev_t = true; prev_t = t;

        for (int k = 0; k < 2; k++) {
            f.speed[k] = speed[k];
            f.relx[k] = wok[k] ? rel[k].x : NaN;
            f.rely[k] = wok[k] ? rel[k].y : NaN;
            f.has_rel[k] = wok[k];
            f.rel[k] = rel[k];
        }
        f.sh = sh;
        rows.push_back(f);
        if (rows.size() > 3) rows.pop_front();
        return rows.back();
    }

    // early_live.feature_matrix, in EARLY_FEATURE_NAMES order
    void matrix(double X[2][EARLY_N_FEATURES]) const {
        auto lag = [&](int L) -> const Row * { return (int)rows.size() > L ? &rows[rows.size() - 1 - L] : nullptr; };
        for (int k = 0; k < 2; k++) {
            int j = 0;
            for (int L = 0; L < 3; L++) { auto r = lag(L); X[k][j++] = r ? r->disp[k] : NaN; }
            for (int L = 0; L < 3; L++) { auto r = lag(L); X[k][j++] = r ? r->launch[k] : NaN; }
            for (int L = 0; L < 2; L++) { auto r = lag(L); X[k][j++] = r ? r->speed[k] : NaN; }
            X[k][j++] = rows.back().relx[k];
            X[k][j++] = rows.back().rely[k];
            for (int L = 0; L < 2; L++) { auto r = lag(L); X[k][j++] = r ? r->energy[k] : NaN; }
            for (int L = 0; L < 3; L++) { auto r = lag(L); X[k][j++] = r ? r->div_rel[k] : NaN; }
            for (int L = 0; L < 2; L++) { auto r = lag(L); X[k][j++] = r ? r->mag_rel[k] : NaN; }
            for (int L = 0; L < 2; L++) { auto r = lag(L); X[k][j++] = r ? r->out_rel[k] : NaN; }
            X[k][j++] = rows.back().torso_div;
            X[k][j++] = rows.back().torso_mag;
            X[k][j++] = rows.back().width_rate;
            X[k][j++] = rows.back().speed[1 - k];   // other_speed
            X[k][j++] = (double)k;                  // wrist_id
        }
    }
};

// ---------------- early_live.EarlyDetector.update ----------------
struct Detector {
    FeatureState st;
    double threshold = EARLY_THRESHOLD_BUDGET_2;
    double last_event = -INFINITY;
    bool has_rest[2] = {false, false};
    V2 rest_pos[2];

    // returns the side ("LEFT", "RIGHT", "CENTER", "BACK") or nullptr; fills scores and X
    const char *update(double t, const cv::Mat &frame, const V2 *kp, const double *conf,
                       double scores[2], double X[2][EARLY_N_FEATURES]) {
        const Row &row = st.base(t, frame, kp, conf);
        st.matrix(X);
        scores[0] = scores[1] = 0.0;
        if (row.sh) { scores[0] = early_score(X[0]); scores[1] = early_score(X[1]); }
        for (int k = 0; k < 2; k++)
            if (row.has_rel[k] && std::isfinite(row.disp[k]) && row.disp[k] < 0.15) { has_rest[k] = true; rest_pos[k] = row.rel[k]; }
        int k = scores[1] > scores[0] ? 1 : 0;
        if (!(scores[k] > threshold && t - last_event >= COOLDOWN)) return nullptr;
        last_event = t;
        const char *side = "CENTER";
        for (int j = 0; j < 2; j++)
            if (has_rest[j] && row.has_rel[j]) {
                V2 d = sub(row.rel[j], rest_pos[j]);   // d.y > 0 = down
                if (d.y > UPPER_DROP && d.y > std::fabs(d.x)) side = "BACK";
            }
        if (std::strcmp(side, "BACK") != 0 && has_rest[k] && row.has_rel[k]) {
            V2 m = sub(row.rel[k], rest_pos[k]);
            if (std::fabs(m.x) > 1.3 * std::fabs(m.y) && norm(m) > 0.15) side = m.x > 0 ? "LEFT" : "RIGHT";
        }
        return side;
    }
};

static char cue_byte(const char *side) {   // dodge_live DIRECTIONAL -> esp32_cue CODES
    if (!std::strcmp(side, "LEFT")) return 'R';    // punch from the left  -> DODGE RIGHT
    if (!std::strcmp(side, "RIGHT")) return 'L';   // punch from the right -> DODGE LEFT
    if (!std::strcmp(side, "BACK")) return 'B';    // uppercut -> STEP BACK
    return 'D';                                    // straight -> DUCK
}

// ---------------- pose: yolo11n-pose (OpenCV DNN or ncnn) ----------------
// ultralytics LetterBox to 320x320 (gray 114 border), BGR order; both backends start from this.
static cv::Mat letterbox_img(const cv::Mat &bgr, letterbox_t &lb) {
    lb = letterbox_params(bgr.cols, bgr.rows, 320);
    int nw = (int)std::lround(bgr.cols * lb.gain), nh = (int)std::lround(bgr.rows * lb.gain);
    cv::Mat rs, img;
    cv::resize(bgr, rs, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    int top = (int)lb.pad_y, left = (int)lb.pad_x;
    cv::copyMakeBorder(rs, img, top, 320 - nh - top, left, 320 - nw - left, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return img;
}

static bool decode_person(const float *out, int n, letterbox_t lb, std::vector<int> &scratch, V2 kp[17], double conf[17]) {
    if ((int)scratch.size() < 2 * n) scratch.resize(2 * n);
    pose_person_t p;
    if (!pose_decode_largest(out, n, lb, 0.25f, 0.7f, scratch.data(), &p)) {
        for (int i = 0; i < 17; i++) { kp[i] = {NaN, NaN}; conf[i] = 0; }
        return false;
    }
    for (int i = 0; i < 17; i++) { kp[i] = {p.kpt[i][0], p.kpt[i][1]}; conf[i] = p.kconf[i]; }
    return true;
}

struct Pose {
    virtual ~Pose() {}
    virtual bool run(const cv::Mat &bgr, V2 kp[17], double conf[17]) = 0;
};

struct PoseOpenCV : Pose {
    cv::dnn::Net net;
    std::vector<int> scratch;
    bool load(const std::string &path) {
        net = cv::dnn::readNetFromONNX(path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        return !net.empty();
    }
    bool run(const cv::Mat &bgr, V2 kp[17], double conf[17]) override {
        letterbox_t lb;
        cv::Mat img = letterbox_img(bgr, lb);
        net.setInput(cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false));
        cv::Mat out = net.forward();                 // [1, 56, N]
        return decode_person(out.ptr<float>(), out.size[2], lb, scratch, kp, conf);
    }
};

#ifdef HAVE_NCNN
#include <ncnn/net.h>
struct PoseNcnn : Pose {   // same weights, exported by ultralytics (format=ncnn, imgsz=320)
    ncnn::Net net;
    std::vector<int> scratch;
    std::vector<float> buf;
    bool load(const std::string &dir, int threads, bool fp16) {
        net.opt.num_threads = threads;
        net.opt.use_packing_layout = true;
        net.opt.use_fp16_packed = fp16;
        net.opt.use_fp16_storage = fp16;
        net.opt.use_fp16_arithmetic = fp16;
        net.opt.use_bf16_storage = false;
        return net.load_param((dir + "/model.ncnn.param").c_str()) == 0 && net.load_model((dir + "/model.ncnn.bin").c_str()) == 0;
    }
    bool run(const cv::Mat &bgr, V2 kp[17], double conf[17]) override {
        letterbox_t lb;
        cv::Mat img = letterbox_img(bgr, lb);
        ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
        const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in.substract_mean_normalize(nullptr, norm);
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);                     // w = N anchors, h = 56 channels
        int n = out.w;
        buf.resize((size_t)out.h * n);
        for (int c = 0; c < out.h; c++) std::memcpy(&buf[(size_t)c * n], out.row(c), n * sizeof(float));
        return decode_person(buf.data(), n, lb, scratch, kp, conf);
    }
};
#endif

// ---------------- recorded test frames (make_test_frames.py) ----------------
struct FrameFile {
    FILE *f = nullptr; int w = 0, h = 0, n = 0;
    bool open(const char *path) {
        char magic[4];
        int32_t hdr[3];
        if (!(f = std::fopen(path, "rb")) || std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "BBXF", 4) ||
            std::fread(hdr, 4, 3, f) != 3) return false;
        w = hdr[0]; h = hdr[1]; n = hdr[2];
        return true;
    }
    bool next(double &t, cv::Mat &bgr, V2 kp[17], double conf[17]) {
        float k[17][2], c[17];
        if (std::fread(&t, 8, 1, f) != 1 || std::fread(k, 4, 34, f) != 34 || std::fread(c, 4, 17, f) != 17) return false;
        bgr.create(h, w, CV_8UC3);
        if (std::fread(bgr.data, 1, (size_t)w * h * 3, f) != (size_t)w * h * 3) return false;
        for (int i = 0; i < 17; i++) { kp[i] = {k[i][0], k[i][1]}; conf[i] = c[i]; }
        return true;
    }
};

// ---------------- live camera through QNX's camera API (Sensor Framework, camera unit N) ----------------
#ifdef HAVE_CAMAPI
#include <camera/camera_api.h>
#include <pthread.h>
#include <signal.h>

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

struct CameraSource {
    camera_handle_t h = CAMERA_HANDLE_INVALID;
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
    cv::Mat latest;              // newest frame (BGR), replaced on every callback
    double latest_t = 0;         // capture time (CLOCK_MONOTONIC, s)
    long seq = 0, taken = 0, dropped = 0;
    int bad_type = -1;

    static void on_frame(camera_handle_t, camera_buffer_t *buf, void *arg) {
        CameraSource *s = (CameraSource *)arg;
        double t = now_s();
        cv::Mat bgr;
        if (buf->frametype == CAMERA_FRAMETYPE_YCBYCR) {
            const camera_frame_ycbycr_t &d = buf->framedesc.ycbycr;
            cv::Mat yuyv((int)d.height, (int)d.width, CV_8UC2, buf->framebuf, d.stride);
            cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUYV);
        } else if (buf->frametype == CAMERA_FRAMETYPE_NV12) {
            const camera_frame_nv12_t &d = buf->framedesc.nv12;
            cv::Mat y((int)d.height, (int)d.width, CV_8UC1, buf->framebuf, d.stride);
            cv::Mat uv((int)d.height / 2, (int)d.width / 2, CV_8UC2, buf->framebuf + d.uv_offset, (size_t)d.uv_stride);
            cv::cvtColorTwoPlane(y, uv, bgr, cv::COLOR_YUV2BGR_NV12);
        } else {
            s->bad_type = (int)buf->frametype;
            return;
        }
        pthread_mutex_lock(&s->mu);
        if (s->seq > s->taken) s->dropped++;   // the detector was still busy with an older frame
        s->latest = bgr;
        s->latest_t = t;
        s->seq++;
        pthread_cond_signal(&s->cv);
        pthread_mutex_unlock(&s->mu);
    }

    bool open(int unit) {
        camera_error_t e = camera_open((camera_unit_t)unit, CAMERA_MODE_RW | CAMERA_MODE_ROLL, &h);
        if (e != CAMERA_EOK) e = camera_open((camera_unit_t)unit, CAMERA_MODE_RO | CAMERA_MODE_ROLL, &h);
        if (e != CAMERA_EOK) { std::fprintf(stderr, "vision: camera_open(%d) failed: %d\n", unit, (int)e); return false; }
        camera_set_vf_property(h, CAMERA_IMGPROP_CREATEWINDOW, 0);   // no on-screen window; ignore if not allowed
        e = camera_start_viewfinder(h, on_frame, nullptr, this);
        if (e != CAMERA_EOK) { std::fprintf(stderr, "vision: camera_start_viewfinder failed: %d\n", (int)e); return false; }
        return true;
    }

    // waits for a frame newer than the last one taken; false on stop or timeout
    bool next(double &t, cv::Mat &bgr) {
        pthread_mutex_lock(&mu);
        while (seq == taken && !g_stop) {
            struct timespec until;
            clock_gettime(CLOCK_REALTIME, &until);
            until.tv_sec += 2;
            if (pthread_cond_timedwait(&cv, &mu, &until) != 0 && seq == taken) {
                pthread_mutex_unlock(&mu);
                if (bad_type >= 0) std::fprintf(stderr, "vision: unsupported camera frame type %d\n", bad_type);
                else std::fprintf(stderr, "vision: no camera frames for 2 s\n");
                return false;
            }
        }
        bool ok = !g_stop;
        if (ok) { bgr = latest; t = latest_t; taken = seq; }
        pthread_mutex_unlock(&mu);
        return ok;
    }

    void close() {
        if (h != CAMERA_HANDLE_INVALID) { camera_stop_viewfinder(h); camera_close(h); h = CAMERA_HANDLE_INVALID; }
    }
};
#endif

static void usage() {
    std::fprintf(stderr,
        "usage: vision (--frames FILE [--file-pose] | --camera UNIT [--seconds S])\n"
        "              [--ncnn DIR [--fp16] | --model yolo11n-pose.onnx] [--csv OUT] [--hat] [--threads N] [--prio P]\n");
}

// Saves the frame as a binary PPM with the keypoints drawn (green = confident, red = not), to see what the Pi sees.
static void save_snapshot(const char *path, const cv::Mat &bgr, const V2 *kp, const double *conf) {
    cv::Mat img = bgr.clone();
    for (int i = 0; i < 17; i++) {
        if (!fin(kp[i])) continue;
        cv::Scalar col = conf[i] > KC ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        int r = (i == 9 || i == 10) ? 9 : (i == 5 || i == 6) ? 7 : 4;   // wrists and shoulders bigger
        cv::circle(img, cv::Point((int)kp[i].x, (int)kp[i].y), r, col, 2);
    }
    std::string tmp = std::string(path) + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%d %d\n255\n", img.cols, img.rows);
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    for (int y = 0; y < rgb.rows; y++) std::fwrite(rgb.ptr(y), 1, (size_t)rgb.cols * 3, f);
    std::fclose(f);
    std::rename(tmp.c_str(), path);
}

static const char *cue_name(const char *side) {
    if (!std::strcmp(side, "LEFT")) return "DODGE RIGHT";
    if (!std::strcmp(side, "RIGHT")) return "DODGE LEFT";
    if (!std::strcmp(side, "BACK")) return "STEP BACK";
    return "DUCK";
}

int main(int argc, char **argv) {
    const char *frames_path = nullptr, *csv_path = nullptr, *model_path = "yolo11n-pose.onnx", *ncnn_dir = nullptr;
    const char *snap_path = nullptr;
    bool file_pose = false, use_hat = false, fp16 = false;
    int threads = 4, camera_unit = -1, prio = 0;
    double seconds = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--frames" && i + 1 < argc) frames_path = argv[++i];
        else if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--ncnn" && i + 1 < argc) ncnn_dir = argv[++i];
        else if (a == "--fp16") fp16 = true;
        else if (a == "--file-pose") file_pose = true;
        else if (a == "--hat") use_hat = true;
        else if (a == "--camera" && i + 1 < argc) camera_unit = std::atoi(argv[++i]);
        else if (a == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
        else if (a == "--prio" && i + 1 < argc) prio = std::atoi(argv[++i]);
        else if (a == "--snapshot" && i + 1 < argc) snap_path = argv[++i];
        else { usage(); return 2; }
    }
    if (!frames_path == (camera_unit < 0) || (file_pose && camera_unit >= 0)) { usage(); return 2; }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    cv::setNumThreads(threads);
    if (prio > 0) {   // before any worker thread exists, so pose/OpenMP threads inherit it
        struct sched_param sp;
        sp.sched_priority = prio;
        int e = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
        std::printf("vision: SCHED_FIFO priority %d %s\n", prio, e ? std::strerror(e) : "set");
    }

    FrameFile src;
    if (frames_path && !src.open(frames_path)) { std::fprintf(stderr, "vision: can't read %s\n", frames_path); return 1; }
#ifdef HAVE_CAMAPI
    CameraSource cam;
    if (camera_unit >= 0) {
        signal(SIGINT, on_sigint);
        signal(SIGTERM, on_sigint);
        if (!cam.open(camera_unit)) return 1;
    }
#else
    if (camera_unit >= 0) { std::fprintf(stderr, "vision: built without the camera API\n"); return 1; }
#endif
    Pose *pose = nullptr;
    const char *backend = "from file";
    if (!file_pose && ncnn_dir) {
#ifdef HAVE_NCNN
        PoseNcnn *p = new PoseNcnn;
        if (!p->load(ncnn_dir, threads, fp16)) { std::fprintf(stderr, "vision: can't load ncnn model in %s\n", ncnn_dir); return 1; }
        pose = p;
        backend = fp16 ? "ncnn fp16" : "ncnn fp32";
#else
        std::fprintf(stderr, "vision: built without ncnn\n");
        return 1;
#endif
    } else if (!file_pose) {
        PoseOpenCV *p = new PoseOpenCV;
        if (!p->load(model_path)) { std::fprintf(stderr, "vision: can't load %s\n", model_path); return 1; }
        pose = p;
        backend = "OpenCV DNN";
    }
    if (use_hat && hat_open() != 0) return 1;
    FILE *csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (csv) {
        std::fprintf(csv, "i,t,found");
        for (int k = 0; k < 2; k++) for (int j = 0; j < EARLY_N_FEATURES; j++) std::fprintf(csv, ",w%d_%s", k, EARLY_FEATURE_NAMES[j]);
        std::fprintf(csv, ",score0,score1,event\n");
    }
    if (camera_unit >= 0)
        std::printf("vision: camera unit %d, pose=%s, threads=%d, threshold %.3f, hat %s (Ctrl-C to stop)\n", camera_unit,
                    backend, threads, EARLY_THRESHOLD_BUDGET_2, use_hat ? "on" : "off");
    else
        std::printf("vision: %d frames %dx%d, pose=%s, threads=%d, threshold %.3f\n", src.n, src.w, src.h,
                    backend, threads, EARLY_THRESHOLD_BUDGET_2);

    Detector det;
    std::vector<double> t_pose, t_feat, t_total, t_capture;
    double t, scores[2], X[2][EARLY_N_FEATURES];
    V2 kp[17], fkp[17];
    double conf[17], fconf[17];
    cv::Mat frame;
    int i = 0, events = 0, found_n = 0;
    double t_start = now_s(), t_status = t_start;
    auto next_frame = [&]() -> bool {
#ifdef HAVE_CAMAPI
        if (camera_unit >= 0) {
            if (seconds > 0 && now_s() - t_start > seconds) return false;
            return cam.next(t, frame);
        }
#endif
        return src.next(t, frame, fkp, fconf);
    };
    while (next_frame()) {
        if (i == 0) std::printf("vision: first frame %dx%d\n", frame.cols, frame.rows);
        double a = now_s();
        bool found;
        if (file_pose) {
            std::memcpy(kp, fkp, sizeof kp); std::memcpy(conf, fconf, sizeof conf);
            found = fin(kp[L_SH]);
        } else {
            found = pose->run(frame, kp, conf);
        }
        double b = now_s();
        const char *side = det.update(t, frame, kp, conf, scores, X);
        double c = now_s();
        if (use_hat) hat_send(side ? cue_byte(side) : 'H');
        double d = now_s();
        t_pose.push_back((b - a) * 1e3); t_feat.push_back((c - b) * 1e3); t_total.push_back((d - a) * 1e3);
        if (camera_unit >= 0) t_capture.push_back((d - t) * 1e3);   // camera callback -> cue sent to the hat
        static double last_snap = 0;
        if (snap_path && d - last_snap >= 2.0) { save_snapshot(snap_path, frame, kp, conf); last_snap = d; }
        found_n += found;
        if (side) {
            events++;
            std::printf("  %7.2fs  %-11s (%-6s) p=%.3f/%.3f  frame %d, %.1f ms after capture\n", t - (camera_unit >= 0 ? t_start : 0),
                        cue_name(side), side, scores[0], scores[1], i, camera_unit >= 0 ? (d - t) * 1e3 : (d - a) * 1e3);
        }
        if (camera_unit >= 0 && d - t_status >= 2.0) {
            size_t n = t_pose.size();
            double fps = 0, lat = 0;
            int m = 0;
            for (size_t q = n; q-- > 0 && m < 60; m++) lat += t_capture[q];
            fps = std::min<size_t>(n, 60) / std::max(1e-3, (d - t_status));
            (void)fps;
            std::printf("  [%5.1fs] %d frames, person %d%%, capture->cue avg %.1f ms (last %d)%s\n", d - t_start, i + 1,
                        (int)(100.0 * found_n / (i + 1)), lat / std::max(1, m), m,
#ifdef HAVE_CAMAPI
                        cam.dropped ? " (older frames skipped while busy)" : ""
#else
                        ""
#endif
            );
            t_status = d;
        }
        if (csv) {
            std::fprintf(csv, "%d,%.6f,%d", i, t, found ? 1 : 0);
            for (int k = 0; k < 2; k++) for (int j = 0; j < EARLY_N_FEATURES; j++) std::fprintf(csv, ",%.9g", X[k][j]);
            std::fprintf(csv, ",%.9g,%.9g,%s\n", scores[0], scores[1], side ? side : "");
        }
        i++;
    }
    if (use_hat) hat_send('X');
#ifdef HAVE_CAMAPI
    if (camera_unit >= 0) cam.close();
#endif
    if (csv) std::fclose(csv);
    double wall = now_s() - t_start;
    auto stats = [](const char *name, std::vector<double> v) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end());
        double s = 0; for (double x : v) s += x;
        std::printf("  %-22s avg %6.2f  p99 %6.2f  max %6.2f ms\n", name, s / v.size(), v[(size_t)(0.99 * (v.size() - 1))], v.back());
    };
    std::printf("vision: %d frames in %.1f s (%.1f fps), %d cues\n", i, wall, i / std::max(wall, 1e-3), events);
    stats("pose", t_pose);
    stats("features + model", t_feat);
    stats("frame -> cue total", t_total);
    stats("capture -> cue (live)", t_capture);
    return 0;
}
