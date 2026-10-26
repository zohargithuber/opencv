// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// CUDA backend of inspection::Clahe (clahe: cuda). Plain C++ interface so clahe.cpp needs no
// CUDA headers. Every device buffer and the stream are created by cudaCreate; cudaApply only
// copies the ROI in, runs two kernels and copies the result out.

#ifndef OPENCV_APPS_INSPECTION_CLAHE_CUDA_HPP
#define OPENCV_APPS_INSPECTION_CLAHE_CUDA_HPP

#include <cstddef>

namespace inspection {
namespace clahe {

struct CudaContext;

// width, height: ROI size, multiples of tilesX, tilesY. clipLimit, scale: clipLimitPx, lutScale.
CudaContext* cudaCreate(int width, int height, int tilesX, int tilesY, int clipLimit, float scale);
// Pinned host buffer (width x height, *step bytes per row) that cudaApply writes the result into;
// pinned, so the download is a direct DMA. Owned by ctx.
unsigned char* cudaHostDst(CudaContext* ctx, size_t* step);
// Synchronous: returns when cudaHostDst() holds the result. src: host 8UC1 image (a camera
// buffer allocated pinned/mapped avoids the driver's staging copy on upload).
void cudaApply(CudaContext* ctx, const unsigned char* src, size_t srcStep);
void cudaDestroy(CudaContext* ctx); // null is a no-op

} // namespace clahe
} // namespace inspection

#endif // OPENCV_APPS_INSPECTION_CLAHE_CUDA_HPP
