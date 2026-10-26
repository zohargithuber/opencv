// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "inspection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <opencv2/imgproc.hpp>

namespace inspection {

ScratchDiffCheck::ScratchDiffCheck(const Config& cfg, const cv::Mat& goldenGray)
    : cfg_(cfg)
{
    if (cfg.scratchMethod != Config::SCRATCH_DIFF)
        return;

    // A part shifted by 1 px leaves |roi - golden| up to the golden's local 3x3 range
    // along its edges -- thin and long, exactly like a scratch. Ignore those pixels.
    cv::Mat range, edges;
    cv::morphologyEx(goldenGray, range, cv::MORPH_GRADIENT, cv::Mat::ones(3, 3, CV_8U));
    cv::threshold(range, edges, cfg.scratchDiffThreshold, 255, cv::THRESH_BINARY);
    if (cfg.scratchEdgeMargin > 0)
    {
        const int k = 2 * cfg.scratchEdgeMargin + 1;
        cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
    }
    cv::bitwise_not(edges, allow_);
    mask_.create(goldenGray.size(), CV_8U);

    // Every per-frame buffer is sized here, so run() never allocates. Each candidate pixel
    // starts at most one provisional label, so scratch_max_candidates bounds the label count.
    prevRow_.assign(goldenGray.cols, 0);
    curRow_.assign(goldenGray.cols, 0);
    rowCount_.assign(goldenGray.rows, 0);
    if (cfg.stripeThreads > 1)
        pool_.reset(new StripePool(std::min(cfg.stripeThreads, goldenGray.rows)));
    parent_.assign(cfg.scratchMaxCandidates + 1, 0);
    blobs_.assign(cfg.scratchMaxCandidates + 1, Blob());
}

int ScratchDiffCheck::findRoot(int l)
{
    while (parent_[l] != l)
    {
        parent_[l] = parent_[parent_[l]]; // path halving
        l = parent_[l];
    }
    return l;
}

void ScratchDiffCheck::maskRows(const cv::Mat& diff, int y0, int y1)
{
    // Branch-free body and a hoisted bound so the compiler vectorizes it (NEON on aarch64).
    const uchar level = cv::saturate_cast<uchar>(cvFloor(cfg_.scratchDiffThreshold)); // 8U: pixel > floor(thresh)
    const int cols = diff.cols;
    for (int y = y0; y < y1; y++)
    {
        const uchar* __restrict d = diff.ptr<uchar>(y);
        const uchar* __restrict a = allow_.ptr<uchar>(y);
        uchar* __restrict m = mask_.ptr<uchar>(y);
        unsigned rowCount = 0;
        for (int x = 0; x < cols; x++)
        {
            const uchar v = (uchar)(a[x] & -(uchar)(d[x] > level)); // allow_ is 0 or 255
            m[x] = v;
            rowCount += v >> 7;
        }
        rowCount_[y] = (int)rowCount;
    }
}

ScratchDiffCheck::Outcome ScratchDiffCheck::run(const cv::Mat& diff, cv::Rect& box, float& length)
{
    CV_Assert(diff.size() == allow_.size() && diff.type() == CV_8U);

    // threshold + AND allow_ + countNonZero fused into one pass (maskRows), split into
    // stripe_threads horizontal stripes. Hand-written rather than cv::threshold, whose
    // parallel_for_ path allocates on every call (steady-state contract).
    struct MaskJob : StripePool::Job
    {
        ScratchDiffCheck* self;
        const cv::Mat* diff;
        void run(int y0, int y1) override { self->maskRows(*diff, y0, y1); }
    } job;
    job.self = this;
    job.diff = &diff;
    const int rows = diff.rows, cols = diff.cols;
    if (pool_)
        pool_->run(job, rows);
    else
        maskRows(diff, 0, rows);
    int candidates = 0;
    for (int y = 0; y < rows; y++)
        candidates += rowCount_[y];

    // Fast exit: fewer differing pixels than one minimum-length scratch.
    length = 0.f;
    if (candidates < cfg_.scratchMinLength)
        return CLEAN;
    // Bounded work: noise that leaves more candidates than the budget allows is not searched.
    if (candidates > cfg_.scratchMaxCandidates)
        return NOISY;

    // One-pass 8-connected labelling with union-find; blob moments accumulate per label.
    // Work follows the candidates, not the ROI: rows without candidates are skipped outright
    // and 16-px chunks without candidates cost one 16-byte test. Labels are written only for
    // candidate pixels; a neighbour's label is read only where its mask pixel is set, so the
    // label rows never need clearing.
    int* prev = prevRow_.data();
    int* cur = curRow_.data();
    int next = 1;
    for (int y = 0; y < rows; y++)
    {
        if (rowCount_[y] == 0)
            continue; // all-zero mask row: the next row's mask lookups see no neighbours here
        const uchar* m = mask_.ptr<uchar>(y);
        const uchar* mp = y > 0 ? mask_.ptr<uchar>(y - 1) : nullptr;
        for (int x0 = 0; x0 < cols; x0 += 16)
        {
            const int x1 = std::min(x0 + 16, cols);
            if (x1 - x0 == 16)
            {
                uint64_t w[2];
                std::memcpy(w, m + x0, 16);
                if ((w[0] | w[1]) == 0)
                    continue;
            }
            for (int x = x0; x < x1; x++)
            {
                if (!m[x])
                    continue;
                int l = 0;
                const int nbr[4] = { x > 0 && m[x - 1] ? cur[x - 1] : 0,
                                     mp && x > 0 && mp[x - 1] ? prev[x - 1] : 0,
                                     mp && mp[x] ? prev[x] : 0,
                                     mp && x + 1 < cols && mp[x + 1] ? prev[x + 1] : 0 };
                for (int k = 0; k < 4; k++)
                {
                    if (!nbr[k])
                        continue;
                    int r = findRoot(nbr[k]);
                    if (!l)
                        l = r;
                    else if (r != l)
                    {
                        if (r < l)
                            std::swap(r, l);
                        parent_[r] = l;
                    }
                }
                if (!l)
                {
                    l = next++;
                    parent_[l] = l;
                    blobs_[l] = Blob{ 0, x, y, x, y, 0, 0, 0, 0, 0 };
                }
                cur[x] = l;
                Blob& b = blobs_[l];
                b.n++;
                b.x0 = std::min(b.x0, x); b.x1 = std::max(b.x1, x);
                b.y0 = std::min(b.y0, y); b.y1 = std::max(b.y1, y);
                b.sx += x; b.sy += y;
                b.sxx += (double)x * x; b.syy += (double)y * y; b.sxy += (double)x * y;
            }
        }
        std::swap(prev, cur);
    }

    // Merge provisional labels into their roots (children always have larger labels).
    for (int l = next - 1; l >= 1; l--)
    {
        const int r = findRoot(l);
        if (r == l)
            continue;
        Blob& a = blobs_[r];
        const Blob& b = blobs_[l];
        a.n += b.n;
        a.x0 = std::min(a.x0, b.x0); a.x1 = std::max(a.x1, b.x1);
        a.y0 = std::min(a.y0, b.y0); a.y1 = std::max(a.y1, b.y1);
        a.sx += b.sx; a.sy += b.sy; a.sxx += b.sxx; a.syy += b.syy; a.sxy += b.sxy;
    }

    // Long and short side from the blob's second moments: a thin segment of length L has
    // variance L^2/12 along its axis. Same test as the former minAreaRect criterion.
    for (int l = 1; l < next; l++)
    {
        if (parent_[l] != l)
            continue;
        const Blob& b = blobs_[l];
        const double mx = b.sx / b.n, my = b.sy / b.n;
        const double a = b.sxx / b.n - mx * mx, c = b.syy / b.n - my * my, d = b.sxy / b.n - mx * my;
        const double half = 0.5 * (a + c), dev = std::sqrt(std::max(0.0, 0.25 * (a - c) * (a - c) + d * d));
        const float longSide = (float)std::sqrt(12.0 * (half + dev));
        const float shortSide = std::max((float)std::sqrt(12.0 * std::max(0.0, half - dev)), 1.f);
        if (longSide >= cfg_.scratchMinLength && longSide / shortSide >= cfg_.scratchMinElongation &&
            longSide > length)
        {
            length = longSide;
            box = cv::Rect(b.x0, b.y0, b.x1 - b.x0 + 1, b.y1 - b.y0 + 1);
        }
    }
    return length > 0.f ? SCRATCH : CLEAN;
}

} // namespace inspection
