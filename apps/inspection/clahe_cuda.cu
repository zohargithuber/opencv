// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "clahe_cuda.hpp"
#include "clahe_core.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace inspection {
namespace clahe {

static void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("clahe cuda: ") + what + ": " + cudaGetErrorString(e));
}

struct CudaContext
{
    int width, height, tilesX, tilesY, tileW, tileH, clipLimit;
    float scale;
    cudaStream_t stream = nullptr;
    unsigned char *src = nullptr, *dst = nullptr, *lut = nullptr, *hostDst = nullptr;
    size_t srcPitch = 0, dstPitch = 0;
};

// One block per tile: shared-memory histogram, then thread 0 builds the LUT with the same
// code as the CPU path (clahe_core.hpp).
__global__ void lutKernel(const unsigned char* src, size_t pitch, int tileW, int tileH, int tilesX,
                          int clipLimit, float scale, unsigned char* lut)
{
    __shared__ int hist[kHistSize];
    for (int i = threadIdx.x; i < kHistSize; i += blockDim.x)
        hist[i] = 0;
    __syncthreads();

    const int k = blockIdx.x;
    const unsigned char* tile = src + (size_t)(k / tilesX) * tileH * pitch + (k % tilesX) * tileW;
    for (int y = 0; y < tileH; y++)
    {
        const unsigned char* row = tile + (size_t)y * pitch;
        for (int x = threadIdx.x; x < tileW; x += blockDim.x)
            atomicAdd(&hist[row[x]], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0)
        tileLut(hist, clipLimit, scale, lut + (size_t)k * kHistSize);
}

// One thread per pixel: bilinear blend of the four surrounding tile LUTs (cv::CLAHE's formula).
__global__ void interpKernel(const unsigned char* src, size_t srcPitch, unsigned char* dst, size_t dstPitch,
                             int width, int height, int tileW, int tileH, int tilesX, int tilesY,
                             const unsigned char* __restrict__ lut)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int tx1, tx2, ty1, ty2;
    float xa, ya;
    axisWeights(x, tileW, tilesX, tx1, tx2, xa);
    axisWeights(y, tileH, tilesY, ty1, ty2, ya);
    const int v = src[(size_t)y * srcPitch + x];
    const unsigned char* l1 = lut + (size_t)ty1 * tilesX * kHistSize;
    const unsigned char* l2 = lut + (size_t)ty2 * tilesX * kHistSize;
    const int i1 = tx1 * kHistSize + v, i2 = tx2 * kHistSize + v;
    const float res = (__ldg(l1 + i1) * (1.0f - xa) + __ldg(l1 + i2) * xa) * (1.0f - ya) +
                      (__ldg(l2 + i1) * (1.0f - xa) + __ldg(l2 + i2) * xa) * ya;
    dst[(size_t)y * dstPitch + x] = saturateU8(res);
}

CudaContext* cudaCreate(int width, int height, int tilesX, int tilesY, int clipLimit, float scale)
{
    CudaContext* c = new CudaContext();
    c->width = width; c->height = height; c->tilesX = tilesX; c->tilesY = tilesY;
    c->tileW = width / tilesX; c->tileH = height / tilesY;
    c->clipLimit = clipLimit; c->scale = scale;
    try
    {
        check(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "stream");
        check(cudaMallocPitch((void**)&c->src, &c->srcPitch, width, height), "src buffer");
        check(cudaMallocPitch((void**)&c->dst, &c->dstPitch, width, height), "dst buffer");
        check(cudaMalloc((void**)&c->lut, (size_t)tilesX * tilesY * kHistSize), "lut buffer");
        check(cudaHostAlloc((void**)&c->hostDst, (size_t)width * height, cudaHostAllocDefault), "pinned dst");
    }
    catch (...)
    {
        cudaDestroy(c);
        throw;
    }
    return c;
}

unsigned char* cudaHostDst(CudaContext* c, size_t* step)
{
    *step = (size_t)c->width;
    return c->hostDst;
}

void cudaApply(CudaContext* c, const unsigned char* src, size_t srcStep)
{
    check(cudaMemcpy2DAsync(c->src, c->srcPitch, src, srcStep, c->width, c->height,
                            cudaMemcpyHostToDevice, c->stream), "upload");
    lutKernel<<<c->tilesX * c->tilesY, 256, 0, c->stream>>>(c->src, c->srcPitch, c->tileW, c->tileH, c->tilesX,
                                                            c->clipLimit, c->scale, c->lut);
    const dim3 block(32, 8), grid((c->width + 31) / 32, (c->height + 7) / 8);
    interpKernel<<<grid, block, 0, c->stream>>>(c->src, c->srcPitch, c->dst, c->dstPitch, c->width, c->height,
                                                c->tileW, c->tileH, c->tilesX, c->tilesY, c->lut);
    check(cudaGetLastError(), "kernel launch");
    check(cudaMemcpy2DAsync(c->hostDst, c->width, c->dst, c->dstPitch, c->width, c->height,
                            cudaMemcpyDeviceToHost, c->stream), "download");
    check(cudaStreamSynchronize(c->stream), "sync");
}

void cudaDestroy(CudaContext* c)
{
    if (!c)
        return;
    cudaFree(c->src);
    cudaFree(c->dst);
    cudaFree(c->lut);
    cudaFreeHost(c->hostDst);
    if (c->stream)
        cudaStreamDestroy(c->stream);
    delete c;
}

} // namespace clahe
} // namespace inspection
