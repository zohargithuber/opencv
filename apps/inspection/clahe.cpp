// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "inspection.hpp"
#include "clahe_core.hpp"

#include <algorithm>
#include <cstring>

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/ocl.hpp>

#ifdef INSPECTION_WITH_CUDA
#include "clahe_cuda.hpp"
#endif

namespace inspection {

using namespace cv;

#ifdef INSPECTION_WITH_CUDA
struct Clahe::Cuda
{
    clahe::CudaContext* ctx = nullptr;
    ~Cuda() { clahe::cudaDestroy(ctx); }
};
#else
struct Clahe::Cuda {};
#endif

Clahe::Clahe(const Config& cfg)
    : cfg_(cfg)
{
    lutPass_.self = interpPass_.self = this;
    lutPass_.lut = true;
    if (!enabled())
        return;

    const cv::Size size = cfg.roi.size();
    const int tilesX = cfg.claheTiles.width, tilesY = cfg.claheTiles.height;
    CV_Assert(tilesX >= 1 && tilesY >= 1 && tilesX <= clahe::kMaxTiles && tilesY <= clahe::kMaxTiles);
    CV_Assert(size.width % tilesX == 0 && size.height % tilesY == 0);
    tile_ = cv::Size(size.width / tilesX, size.height / tilesY);
    clipLimit_ = clahe::clipLimitPx(cfg.claheClipLimit, tile_.area());
    scale_ = clahe::lutScale(tile_.area());
    dst_.create(size, CV_8UC1);

    if (cfg.claheBackend == Config::CLAHE_OPENCL)
    {
        if (!cv::ocl::haveOpenCL())
            CV_Error(cv::Error::StsNotImplemented, "clahe: opencl needs OpenCV built WITH_OPENCL and an OpenCL device");
        cv::ocl::setUseOpenCL(true);
        ocl_ = cv::createCLAHE(cfg.claheClipLimit, cfg.claheTiles);
        return;
    }
    if (cfg.claheBackend == Config::CLAHE_CUDA)
    {
#ifdef INSPECTION_WITH_CUDA
        cuda_.reset(new Cuda);
        cuda_->ctx = clahe::cudaCreate(size.width, size.height, tilesX, tilesY, clipLimit_, scale_);
        size_t step = 0;
        uchar* host = clahe::cudaHostDst(cuda_->ctx, &step);
        dst_ = cv::Mat(size, CV_8UC1, host, step); // pinned, owned by the CUDA context
        return;
#else
        CV_Error(cv::Error::StsNotImplemented, "clahe: cuda needs the app built with a CUDA toolkit (nvcc)");
#endif
    }

    // cpu: per-column interpolation tables, as cv::CLAHE builds per call.
    lut_.create(tilesX * tilesY, clahe::kHistSize, CV_8UC1);
    ind1_.resize(size.width);
    ind2_.resize(size.width);
    xa_.resize(size.width);
    xa1_.resize(size.width);
    for (int x = 0; x < size.width; x++)
    {
        int tx1, tx2;
        clahe::axisWeights(x, tile_.width, tilesX, tx1, tx2, xa_[x]);
        xa1_[x] = 1.0f - xa_[x];
        ind1_[x] = tx1 * clahe::kHistSize;
        ind2_[x] = tx2 * clahe::kHistSize;
    }
    spanEnd_.resize(size.width);
    for (int x = size.width - 1, end = size.width; x >= 0; x--)
    {
        if (x + 1 < size.width && (ind1_[x] != ind1_[x + 1] || ind2_[x] != ind2_[x + 1]))
            end = x + 1;
        spanEnd_[x] = end;
    }
    if (cfg.stripeThreads > 1)
        pool_.reset(new StripePool(std::min(cfg.stripeThreads, tilesX * tilesY)));
}

Clahe::~Clahe() = default;

void Clahe::lutTiles(int k0, int k1)
{
    const int tilesX = cfg_.claheTiles.width;
    for (int k = k0; k < k1; k++)
    {
        // Four interleaved histograms: consecutive equal pixels don't wait on each other's store.
        int h[4][clahe::kHistSize];
        std::memset(h, 0, sizeof(h));
        const int x0 = (k % tilesX) * tile_.width, y0 = (k / tilesX) * tile_.height;
        for (int y = y0; y < y0 + tile_.height; y++)
        {
            const uchar* p = src_->ptr<uchar>(y) + x0;
            int x = 0;
            for (; x <= tile_.width - 4; x += 4)
            {
                h[0][p[x]]++;
                h[1][p[x + 1]]++;
                h[2][p[x + 2]]++;
                h[3][p[x + 3]]++;
            }
            for (; x < tile_.width; x++)
                h[0][p[x]]++;
        }
        for (int i = 0; i < clahe::kHistSize; i++)
            h[0][i] += h[1][i] + h[2][i] + h[3][i];
        clahe::tileLut(h[0], clipLimit_, scale_, lut_.ptr<uchar>(k));
    }
}

void Clahe::interpRows(int y0, int y1)
{
    const int tilesX = cfg_.claheTiles.width, tilesY = cfg_.claheTiles.height;
    const int cols = src_->cols, n = tilesX * clahe::kHistSize;
    const int* __restrict ind1 = ind1_.data();
    const int* __restrict ind2 = ind2_.data();
    const float* __restrict xa = xa_.data();
    const float* __restrict xa1 = xa1_.data();
    // The two tile rows a pixel row interpolates between, blended once per row (cv::CLAHE
    // blends them per pixel). Stack buffer: no allocation.
    float rowLut[clahe::kMaxTiles * clahe::kHistSize];
    for (int y = y0; y < y1; y++)
    {
        int ty1, ty2;
        float ya;
        clahe::axisWeights(y, tile_.height, tilesY, ty1, ty2, ya);
        const float ya1 = 1.0f - ya;
        const uchar* __restrict l1 = lut_.ptr<uchar>(ty1 * tilesX);
        const uchar* __restrict l2 = lut_.ptr<uchar>(ty2 * tilesX);
        for (int i = 0; i < n; i++)
            rowLut[i] = l1[i] * ya1 + l2[i] * ya;

        const uchar* __restrict s = src_->ptr<uchar>(y);
        uchar* __restrict d = dst_.ptr<uchar>(y);
        // Columns between two tile centres share one LUT pair, so each span is a gather loop.
        for (int x0 = 0; x0 < cols;)
        {
            const float* __restrict a = rowLut + ind1[x0];
            const float* __restrict b = rowLut + ind2[x0];
            const int x1 = spanEnd_[x0];
            int x = x0;
#if CV_SIMD && !defined(CLAHE_NO_SIMD)
            // Universal intrinsics: vgatherdps on AVX2 (x86 built with CPU_BASELINE=AVX2).
            const int VL = VTraits<v_float32>::vlanes();
            for (; x <= x1 - 4 * VL; x += 4 * VL)
            {
                v_uint16 w0, w1;
                v_uint32 q[4];
                v_int32 r[4];
                v_expand(vx_load(s + x), w0, w1);
                v_expand(w0, q[0], q[1]);
                v_expand(w1, q[2], q[3]);
                for (int j = 0; j < 4; j++)
                {
                    const v_int32 v = v_reinterpret_as_s32(q[j]);
                    r[j] = v_round(v_fma(v_lut(b, v), vx_load(xa + x + j * VL),
                                         v_mul(v_lut(a, v), vx_load(xa1 + x + j * VL))));
                }
                v_store(d + x, v_pack_u(v_pack(r[0], r[1]), v_pack(r[2], r[3])));
            }
#endif
            for (; x < x1; x++)
            {
                const int v = s[x];
                d[x] = cv::saturate_cast<uchar>(a[v] * xa1[x] + b[v] * xa[x]);
            }
            x0 = x1;
        }
    }
}

const cv::Mat& Clahe::apply(const cv::Mat& src)
{
    CV_Assert(enabled() && src.type() == CV_8UC1 && src.size() == dst_.size());
    if (ocl_)
    {
        src.copyTo(usrc_);
        ocl_->apply(usrc_, udst_);
        udst_.copyTo(dst_);
        return dst_;
    }
#ifdef INSPECTION_WITH_CUDA
    if (cuda_)
    {
        clahe::cudaApply(cuda_->ctx, src.data, src.step);
        return dst_;
    }
#endif
    src_ = &src;
    const int tiles = cfg_.claheTiles.area();
    if (pool_)
    {
        pool_->run(lutPass_, tiles);
        pool_->run(interpPass_, src.rows);
    }
    else
    {
        lutTiles(0, tiles);
        interpRows(0, src.rows);
    }
    src_ = nullptr;
    return dst_;
}

} // namespace inspection
