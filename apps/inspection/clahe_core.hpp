// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// CLAHE arithmetic shared by the CPU path (clahe.cpp) and the CUDA kernels (clahe_cuda.cu),
// so both produce the same per-tile LUTs. Follows cv::CLAHE (modules/imgproc/src/clahe.cpp)
// for 8-bit input whose size is a multiple of the tile grid.

#ifndef OPENCV_APPS_INSPECTION_CLAHE_CORE_HPP
#define OPENCV_APPS_INSPECTION_CLAHE_CORE_HPP

#include <cmath>

#ifndef __CUDACC__
#include <opencv2/core/fast_math.hpp> // cvRound: inline, unlike lrintf (an external call on GCC)
#endif

#ifdef __CUDACC__
#define CLAHE_HD __host__ __device__
#else
#define CLAHE_HD
#endif

namespace inspection {
namespace clahe {

enum { kHistSize = 256, kMaxTiles = 16 };

// cv::CLAHE's clip limit for one tile, in pixels per bin (0 = no clipping).
inline int clipLimitPx(double clipLimit, int tileArea)
{
    if (!(clipLimit > 0.0))
        return 0;
    const int limit = static_cast<int>(clipLimit * tileArea / kHistSize);
    return limit > 1 ? limit : 1;
}

inline float lutScale(int tileArea) { return static_cast<float>(kHistSize - 1) / tileArea; }

CLAHE_HD inline int roundEven(float v) // cvRound / saturate_cast: round half to even
{
#if defined(__CUDA_ARCH__)
    return __float2int_rn(v);
#elif defined(__CUDACC__)
    return static_cast<int>(std::lrint(v)); // nvcc host pass: compiled, never called
#else
    return cvRound(v);
#endif
}

CLAHE_HD inline unsigned char saturateU8(float v)
{
    const int i = roundEven(v);
    return static_cast<unsigned char>(i < 0 ? 0 : i > 255 ? 255 : i);
}

// Clip, redistribute and integrate one tile histogram into its LUT. hist is modified.
CLAHE_HD inline void tileLut(int* hist, int clipLimit, float scale, unsigned char* lut)
{
    if (clipLimit > 0)
    {
        int clipped = 0;
        for (int i = 0; i < kHistSize; ++i)
        {
            if (hist[i] > clipLimit)
            {
                clipped += hist[i] - clipLimit;
                hist[i] = clipLimit;
            }
        }
        const int redistBatch = clipped / kHistSize;
        int residual = clipped - redistBatch * kHistSize;
        for (int i = 0; i < kHistSize; ++i)
            hist[i] += redistBatch;
        if (residual != 0)
        {
            const int residualStep = kHistSize / residual > 1 ? kHistSize / residual : 1;
            for (int i = 0; i < kHistSize && residual > 0; i += residualStep, residual--)
                hist[i]++;
        }
    }
    int sum = 0;
    for (int i = 0; i < kHistSize; ++i)
    {
        sum += hist[i];
        lut[i] = saturateU8(sum * scale);
    }
}

// Tile pair and weight along one axis for pixel coordinate p (tile size `size`, `tiles` tiles).
CLAHE_HD inline void axisWeights(int p, int size, int tiles, int& t1, int& t2, float& w)
{
    const float tf = p * (1.0f / size) - 0.5f;
    t1 = static_cast<int>(floorf(tf));
    w = tf - t1;
    t2 = t1 + 1 < tiles - 1 ? t1 + 1 : tiles - 1;
    t1 = t1 > 0 ? t1 : 0;
}

} // namespace clahe
} // namespace inspection

#endif // OPENCV_APPS_INSPECTION_CLAHE_CORE_HPP
