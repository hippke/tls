/* TLS period search kernel in C (same statistic as core_fused.py).
 *
 * tls_search(): for every trial period (OpenMP-parallel over periods) fold,
 * sort, and scan all trial durations and phase shifts. See core_fused.py for
 * the derivation: chi2 = sum r^2 w + k^2 A2 - 2 k AR.
 *
 * Build: cc -O3 -march=native -ffast-math -fopenmp -shared -fPIC tls_kernel.c
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define G 6.673e-11
#define R_SUN 695508000.0
#define M_SUN 1.989e30
#define R_JUP 69911000.0
#define SECONDS_PER_DAY 86400.0

static double T14(double R_s, double M_s, double P, double upper, int small) {
    P = P * SECONDS_PER_DAY;
    R_s = R_SUN * R_s;
    M_s = M_SUN * M_s;
    double base = pow((4 * P) / (M_PI * G * M_s), 1.0 / 3.0);
    double T = small ? R_s * base : (R_s + 2 * R_JUP) * base;
    double r = T / P;
    return r > upper ? upper : r;
}

/* ---- sorting: stable argsort of phases (natural merge sort on runs) ----
 * The folded phases of time-sorted data consist of increasing runs (one per
 * orbit), so a natural merge sort costs O(N log K) for K runs. */
static void argsort_phases(const double *ph, int64_t n, int64_t *idx, int64_t *tmp,
                           int64_t *runs, int64_t *runs_tmp) {
    for (int64_t i = 0; i < n; i++) idx[i] = i;
    /* run boundaries: runs[0..nr] */
    int64_t nr = 0;
    runs[nr++] = 0;
    for (int64_t i = 1; i < n; i++)
        if (ph[i] < ph[i - 1]) runs[nr++] = i;
    runs[nr] = n;
    int64_t *src = idx, *dst = tmp;
    while (nr > 1) {
        int64_t nr_new = 0;
        for (int64_t r = 0; r < nr; r += 2) {
            int64_t lo = runs[r], mid = runs[r + 1];
            int64_t hi = (r + 2 <= nr) ? runs[r + 2] : runs[r + 1];
            runs_tmp[nr_new++] = lo;
            if (r + 1 >= nr) { /* odd run out: copy */
                memcpy(dst + lo, src + lo, (size_t)(mid - lo) * sizeof(int64_t));
                continue;
            }
            int64_t a = lo, b = mid, k = lo;
            while (a < mid && b < hi) {
                /* stable: take from the left run on ties */
                if (ph[src[b]] < ph[src[a]]) dst[k++] = src[b++];
                else dst[k++] = src[a++];
            }
            while (a < mid) dst[k++] = src[a++];
            while (b < hi) dst[k++] = src[b++];
        }
        runs_tmp[nr_new] = n;
        memcpy(runs, runs_tmp, (size_t)(nr_new + 1) * sizeof(int64_t));
        nr = nr_new;
        int64_t *sw = src; src = dst; dst = sw;
    }
    if (src != idx) memcpy(idx, src, (size_t)n * sizeof(int64_t));
}

static inline double dot1(const double *a, const double *r, int64_t len) {
    double acc = 0.0;
    for (int64_t j = 0; j < len; j++) acc += a[j] * r[j];
    return acc;
}

static inline void dot2(const double *a, const double *r, const double *w, int64_t len,
                        double *ar_out, double *a2_out) {
    double ar = 0.0, a2 = 0.0;
    for (int64_t j = 0; j < len; j++) {
        double aj = a[j];
        ar += aj * r[j];
        a2 += aj * aj * w[j];
    }
    *ar_out = ar;
    *a2_out = a2;
}

static void search_one(double period, const double *t, const double *y,
                       const double *inv_dy2, int64_t n, int uniform, double time_span,
                       double depth_min, double R_min, double R_max, double M_min,
                       double M_max, double upper, const int64_t *widths,
                       const int64_t *rows, const int64_t *offsets,
                       const int64_t *lengths, int64_t n_widths, const double *profile,
                       const double *overshoot, const double *sum_a2, double margin,
                       double signal_depth, int prune, double *work, int64_t *iwork,
                       double *out_chi2, int64_t *out_row, double *out_depth) {
    int64_t maxw = widths[n_widths - 1];
    if (maxw % 2) maxw++;
    int64_t m = n + maxw;
    double *ph = work;               /* n */
    double *rw = ph + n;             /* m */
    double *w = rw + m;              /* m */
    double *cum = w + m;             /* m+1 */
    double *cum_r2 = cum + m + 1;    /* m+1 */
    double *cum_w = cum_r2 + m + 1;  /* m+1 */
    int64_t *order = iwork;          /* n */
    int64_t *tmp = order + n;        /* n */
    int64_t *runs = tmp + n;         /* n+1 */
    int64_t *runs_tmp = runs + n + 1;/* n+1 */

    for (int64_t i = 0; i < n; i++) {
        double x = t[i] / period;
        ph[i] = x - floor(x);
    }
    argsort_phases(ph, n, order, tmp, runs, runs_tmp);

    double total = 0.0;
    cum[0] = cum_r2[0] = cum_w[0] = 0.0;
    for (int64_t k = 0; k < m; k++) {
        int64_t src = k < n ? order[k] : order[k - n];
        double f = y[src], wk = inv_dy2[src], r = 1.0 - f;
        w[k] = wk;
        rw[k] = r * wk;
        cum[k + 1] = cum[k] + f;
        cum_r2[k + 1] = cum_r2[k] + r * r * wk;
        cum_w[k + 1] = cum_w[k] + wk;
        if (k < n) total += r * r * wk;
    }

    /* M_min / M_max: masses paired with R_min (shortest duration) and R_max
       (longest), see grid.duration_limit_masses (BUGS.md F1) */
    double dmax = T14(R_max, M_max, period, upper, 0);
    double dmin = T14(R_min, M_min, period, upper, 1);
    double tn = time_span / period;
    double cf = (tn + 1) / tn;
    int64_t wmin = (int64_t)floor(dmin * n);
    int64_t wmax = (int64_t)ceil(dmax * n * cf);
    double w0 = inv_dy2[0];

    /* seed for pruning */
    double seed = 0.0;
    if (prune) {
        for (int64_t u = 0; u < n_widths; u++) {
            int64_t d = widths[u];
            if (d < wmin || d > wmax) continue;
            int64_t xth = 1;
            if (margin > 0 && d > margin) {
                xth = (int64_t)(d / (1 / margin));
                if (xth < 1) xth = 1;
            }
            int64_t ns = m - d + 1;
            if (xth == 1 && ns > n) ns = n;
            int64_t bi = -1;
            double bm = depth_min;
            for (int64_t i = 0; i < ns; i += xth) {
                double mean = 1 - (cum[i + d] - cum[i]) / d;
                if (mean > bm) { bm = mean; bi = i; }
            }
            if (bi < 0) continue;
            double k = 1 / (signal_depth / (bm * overshoot[u]));
            double ar, a2;
            dot2(profile + offsets[u], rw + bi, w + bi, lengths[u], &ar, &a2);
            double gain = 2 * k * ar - k * k * a2;
            if (gain > seed) seed = gain;
        }
    }
    double threshold = seed * (1 - 1e-9);

    double best_gain = 0.0, best_depth = 0.0;
    int64_t best_row = 0;
    for (int64_t u = 0; u < n_widths; u++) {
        int64_t d = widths[u];
        if (d < wmin || d > wmax) continue;
        const double *prof = profile + offsets[u];
        int64_t len = lengths[u];
        double ov = overshoot[u];
        double a2c = sum_a2[u] * w0;
        double amax2 = 0.0;
        for (int64_t j = 0; j < len; j++)
            if (prof[j] * prof[j] > amax2) amax2 = prof[j] * prof[j];
        int64_t xth = 1;
        if (margin > 0 && d > margin) {
            xth = (int64_t)(d / (1 / margin));
            if (xth < 1) xth = 1;
        }
        int64_t ns = m - d + 1;
        if (xth == 1 && ns > n) ns = n;
        for (int64_t i = 0; i < ns; i += xth) {
            double mean = 1 - (cum[i + d] - cum[i]) / d;
            if (!(mean > depth_min)) continue;
            double target = mean * ov;
            double k = 1 / (signal_depth / target);
            if (prune) {
                double r2 = cum_r2[i + len] - cum_r2[i], bound;
                if (uniform) {
                    bound = 2 * k * sqrt(a2c * r2) - k * k * a2c;
                } else {
                    double a2b = amax2 * (cum_w[i + len] - cum_w[i]);
                    bound = (a2b * k * k >= r2) ? r2 : 2 * k * sqrt(a2b * r2) - k * k * a2b;
                }
                if (bound * (1 + 1e-9) <= best_gain || bound < threshold) continue;
            }
            double ar, a2;
            if (uniform) {
                ar = dot1(prof, rw + i, len);
                a2 = a2c;
            } else {
                dot2(prof, rw + i, w + i, len, &ar, &a2);
            }
            double gain = 2 * k * ar - k * k * a2;
            if (gain > best_gain) {
                best_gain = gain;
                best_row = rows[u];
                best_depth = 1 - target;
            }
        }
    }
    *out_chi2 = total - best_gain;
    *out_row = best_row;
    *out_depth = best_depth;
}

/* Search all periods. Returns 0 on success, -1 on allocation failure. */
int tls_search(const double *periods, int64_t n_periods, const double *t, const double *y,
               const double *inv_dy2, int64_t n, int uniform, double time_span,
               double depth_min, double R_min, double R_max, double M_min, double M_max,
               double upper, const int64_t *widths, const int64_t *rows,
               const int64_t *offsets, const int64_t *lengths, int64_t n_widths,
               const double *profile, const double *overshoot, const double *sum_a2,
               double margin, double signal_depth, int prune, int n_threads,
               double *out_chi2, int64_t *out_row, double *out_depth) {
    int64_t maxw = widths[n_widths - 1] + 1;
    int64_t m = n + maxw;
    int failed = 0;
#pragma omp parallel num_threads(n_threads)
    {
        double *work = malloc(sizeof(double) * (size_t)(n + 2 * m + 3 * (m + 1)));
        int64_t *iwork = malloc(sizeof(int64_t) * (size_t)(4 * n + 2));
        int ok = work && iwork;
        if (!ok) {
#pragma omp atomic write
            failed = 1;
        }
#pragma omp for schedule(dynamic, 4)
        for (int64_t p = 0; p < n_periods; p++) {
            if (!ok) continue;
            search_one(periods[p], t, y, inv_dy2, n, uniform, time_span, depth_min,
                       R_min, R_max, M_min, M_max, upper, widths, rows, offsets,
                       lengths, n_widths, profile, overshoot, sum_a2, margin,
                       signal_depth, prune, work, iwork, out_chi2 + p, out_row + p,
                       out_depth + p);
        }
        free(work);
        free(iwork);
    }
    return failed ? -1 : 0;
}
