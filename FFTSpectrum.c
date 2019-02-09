
/*
 * Copyright (c) 2008 Hiroaki Gotou.
 * Copyright (c) 2019 Evgeny Marchenkov.
 * 
 * This program is free software : you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.If not, see < https://www.gnu.org/licenses/>.
 */

#include <stdbool.h>
#include <math.h>

#if defined(_MSC_VER)
#include <intrin.h>

#define USE_SSE_AUTO
#define __SSE4_2__
#define __x86_64__
#define SSE_MATHFUN_WITH_CODE
#include "sse_mathfun.h"
#undef SSE_MATHFUN_WITH_CODE
#undef __x86_64__
#undef __SSE4_2__
#undef USE_SSE_AUTO
#undef inline

#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>

#define USE_SSE4
#define SSE_MATHFUN_WITH_CODE
#include "sse_mathfun.h"

#endif

#include "fftw3.h"

#include "vapoursynth/VapourSynth.h"
#include "vapoursynth/VSHelper.h"


typedef struct {
    VSNodeRef *node;
    const VSVideoInfo *in_vi;
    VSVideoInfo out_vi;

    bool show_grid;

    fftwf_complex *fft_in;
    fftwf_complex *fft_out;
    fftwf_plan p;
    float *abs_array;
} FFTSpectrumData;

static void fill_fft_input_array(fftwf_complex *dst, const uint8_t *src, int width, int height, int stride) {
    fftwf_complex *dstp = dst;
    const uint8_t *srcp = src;
    const int mod16_width = width - (width % 16);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < mod16_width; x += 16) {
            __m128i in_buffer, epu32_buffer;
            __m128  cvt_buffer, out_buffer[2];
            const __m128 sse_zero = _mm_setzero_ps();

            in_buffer = _mm_load_si128((const __m128i *)srcp);

            for (int j = 0; j < 4; j++) {
                epu32_buffer = _mm_cvtepu8_epi32(in_buffer);
                cvt_buffer = _mm_cvtepi32_ps(epu32_buffer);

                out_buffer[0] = _mm_unpacklo_ps(cvt_buffer, sse_zero);
                out_buffer[1] = _mm_unpackhi_ps(cvt_buffer, sse_zero);

                _mm_store_ps((float *)(dstp), out_buffer[0]);
                _mm_store_ps((float *)(dstp + 2), out_buffer[1]);

                in_buffer = _mm_shuffle_epi32(in_buffer, _MM_SHUFFLE(0, 3, 2, 1));

                dstp += 4;
            }

            srcp += 16;
        }
        for (int x = mod16_width; x < width; x++) {
            *dstp[0] = (float)*srcp;
            *dstp[1] = 0.0;
            srcp++;
            dstp++;
        }
        srcp += stride - width;
    }
}

static void calculate_absolute_values(float *dst, fftwf_complex *src, int length) {
    fftwf_complex *srcp = src;
    float *dstp = dst;
    const int mod4_length = length - (length % 4);

    for (int i = 0; i < mod4_length; i += 4) {
        __m128 in_buffer[2], mul_buffer[2], add_buffer, out_buffer;
        const __m128 sse_one = _mm_set_ps1(1.0f);

        in_buffer[0] = _mm_load_ps((float *)(srcp));
        in_buffer[1] = _mm_load_ps((float *)(srcp + 2));

        mul_buffer[0] = _mm_mul_ps(in_buffer[0], in_buffer[0]);
        mul_buffer[1] = _mm_mul_ps(in_buffer[1], in_buffer[1]);

        add_buffer = _mm_hadd_ps(mul_buffer[0], mul_buffer[1]);
        add_buffer = _mm_sqrt_ps(add_buffer);
        add_buffer = _mm_add_ps(add_buffer, sse_one);

        out_buffer = log_ps(add_buffer);

        _mm_store_ps(dstp, out_buffer);

        srcp += 4;
        dstp += 4;
    }
    for (int i = mod4_length; i < length; i++) {
       dstp[i] = logf(sqrtf(src[i][0] * src[i][0] + src[i][1] * src[i][1]) + 1.0);
    }
}

static void draw_fft_spectrum(uint8_t *dst, float *src, int width, int height, int stride) {
    uint8_t *dstp = dst;
    float *srcp = src;
    float max = 0;

    memset(dstp, 0, stride * height);

    for (int i = 1; i < height * width; i++) {
        if (srcp[i] > max) {
            max = srcp[i];
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float buf;
            buf = srcp[x + y * width] > max / 2 ? srcp[x + y * width] : 0;
            buf = 255 * buf / max;
            if (buf < 0) buf = 0;
            if (buf > 255) buf = 255;

            if (y < height / 2) {
                if (x < width / 2) {
                    dstp[x + (width / 2) + stride * (y + height / 2)] = (uint8_t)buf;
                }
                else {
                    dstp[x - (width / 2) + stride * (y + height / 2)] = (uint8_t)buf;
                }
            }
            else {
                if (x < width / 2) {
                    dstp[x + (width / 2) + stride * (y - height / 2)] = (uint8_t)buf;
                }
                else {
                    dstp[x - (width / 2) + stride * (y - height / 2)] = (uint8_t)buf;
                }
            }
        }
    }
}

static void draw_grid(uint8_t *buf, int width, int height, int stride) {
    for (int x = (width / 2) % 100; x < width; x += 100) {
        for (int y = 0; y < height; y++) {
            buf[x + y * stride] = 255;
        }
    }

    for (int y = (height / 2) % 100; y < height; y += 100) {
        for (int x = 0; x < width; x++) {
            buf[x + y * stride] = 255;
        }
    }
}

static void VS_CC fftSpectrumInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    FFTSpectrumData *d = (FFTSpectrumData *) * instanceData;
    d->out_vi = *d->in_vi;
    d->out_vi.format = vsapi->getFormatPreset(pfGray8, core);
    vsapi->setVideoInfo(&d->out_vi, 1, node);
}

static const VSFrameRef *VS_CC fftSpectrumGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    FFTSpectrumData *d = (FFTSpectrumData *) * instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {

        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrameRef *dst = vsapi->newVideoFrame(d->out_vi.format, d->out_vi.width, d->out_vi.height, src, core);

        fill_fft_input_array(d->fft_in, vsapi->getReadPtr(src, 0), d->in_vi->width, d->in_vi->height, vsapi->getStride(src, 0));

        fftwf_execute_dft(d->p, d->fft_in, d->fft_out);

        calculate_absolute_values(d->abs_array, d->fft_out, (d->in_vi->width * d->in_vi->height));

        draw_fft_spectrum(vsapi->getWritePtr(dst, 0), d->abs_array, d->out_vi.width, d->out_vi.height, vsapi->getStride(dst, 0));

        if (d->show_grid) {
            draw_grid(vsapi->getWritePtr(dst, 0), d->out_vi.width, d->out_vi.height, vsapi->getStride(dst, 0));
        }

        vsapi->freeFrame(src);

        return dst;
    }

    return 0;
}

static void VS_CC fftSpectrumFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    FFTSpectrumData *d = (FFTSpectrumData *)instanceData;
    vsapi->freeNode(d->node);
    VS_ALIGNED_FREE(d->fft_in);
    VS_ALIGNED_FREE(d->fft_out);
    VS_ALIGNED_FREE(d->abs_array);
    fftwf_destroy_plan(d->p);
    free(d);
}

static void VS_CC fftSpectrumCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    FFTSpectrumData *d;
    d = malloc(sizeof(FFTSpectrumData));

    int err;

    d->node = vsapi->propGetNode(in, "clip", 0, 0);
    d->in_vi = vsapi->getVideoInfo(d->node);

    if (!isConstantFormat(d->in_vi) || d->in_vi->format->sampleType != stInteger || d->in_vi->format->bitsPerSample != 8 ||
            d->in_vi->format->colorFamily == cmRGB || d->in_vi->format->colorFamily == cmCompat) {
        vsapi->setError(out, "FFTSpectrum: only constant format 8bit integer luma-containing input supported");
        vsapi->freeNode(d->node);
        free(d);
        return;
    }

    d->show_grid = (bool)vsapi->propGetInt(in, "grid", 0, &err);
    if (err) {
        d->show_grid = false;
    }

    VS_ALIGNED_MALLOC(&d->fft_in,    (d->in_vi->width * d->in_vi->height * sizeof(fftw_complex)), 32);
    VS_ALIGNED_MALLOC(&d->fft_out,   (d->in_vi->width * d->in_vi->height * sizeof(fftw_complex)), 32);
    VS_ALIGNED_MALLOC(&d->abs_array, (d->in_vi->width * d->in_vi->height * sizeof(float)), 32);

    memset(d->fft_in,    0, (d->in_vi->width * d->in_vi->height * sizeof(fftw_complex)));
    memset(d->fft_out,   0, (d->in_vi->width * d->in_vi->height * sizeof(fftw_complex)));
    memset(d->abs_array, 0, (d->in_vi->width * d->in_vi->height * sizeof(float)));

    d->p = fftwf_plan_dft_2d(d->in_vi->height, d->in_vi->width, d->fft_in, d->fft_out, FFTW_FORWARD, FFTW_MEASURE | FFTW_DESTROY_INPUT);

    vsapi->createFilter(in, out, "FFTSpectrum", fftSpectrumInit, fftSpectrumGetFrame, fftSpectrumFree, fmParallelRequests, 0, d, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("org.beatrice-raws.fftspectrum", "fftspectrum", "FFT Spectrum plugin", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("FFTSpectrum", "clip:clip;grid:int:opt;", fftSpectrumCreate, 0, plugin);
}
