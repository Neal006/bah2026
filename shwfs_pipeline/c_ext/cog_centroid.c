/*
 * Fast CoG centroiding for SH-WFS.
 * 10x10 lenslet, 16px/subaperture, 160x160 frame.
 * No VLAs: patch buffer is fixed at 256 (16*16).
 */
#include <stdint.h>
#include <string.h>

#define PPS_MAX 16
#define PATCH_MAX (PPS_MAX * PPS_MAX)   /* 256 */

void cog_frame(
    const uint8_t  *frame,
    const uint8_t  *bg,
    const int32_t  *active,   /* NY*NX ints, row-major */
    float          *cx_out,   /* NY*NX floats */
    float          *cy_out,
    int NY, int NX, int pps, int frame_w)
{
    float patch[PATCH_MAX];
    float bg_sum;

    for (int i = 0; i < NY; i++) {
        for (int j = 0; j < NX; j++) {
            int idx = i * NX + j;
            if (!active[idx]) {
                cx_out[idx] = (j + 0.5f) * pps;
                cy_out[idx] = (i + 0.5f) * pps;
                continue;
            }

            int y0 = i * pps, x0 = j * pps;

            /* copy patch, compute bg mean */
            bg_sum = 0.0f;
            for (int r = 0; r < pps; r++) {
                for (int c = 0; c < pps; c++) {
                    int pi = r * pps + c;
                    float p = (float)frame[(y0 + r) * frame_w + (x0 + c)];
                    float b = (float)bg  [(y0 + r) * frame_w + (x0 + c)];
                    patch[pi] = p;
                    bg_sum   += b;
                }
            }
            float bg_mean = bg_sum / (pps * pps);

            /* threshold-subtract and accumulate CoG */
            float sum_w = 0.0f, sum_wx = 0.0f, sum_wy = 0.0f;
            for (int r = 0; r < pps; r++) {
                for (int c = 0; c < pps; c++) {
                    float w = patch[r * pps + c] - bg_mean;
                    if (w < 0.0f) w = 0.0f;
                    sum_w  += w;
                    sum_wx += w * c;
                    sum_wy += w * r;
                }
            }

            if (sum_w < 1e-6f) {
                cx_out[idx] = (j + 0.5f) * pps;
                cy_out[idx] = (i + 0.5f) * pps;
            } else {
                cx_out[idx] = x0 + sum_wx / sum_w;
                cy_out[idx] = y0 + sum_wy / sum_w;
            }
        }
    }
}
