// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "inspection.hpp"

namespace inspection {

ScratchCheck::ScratchCheck(const Config& cfg)
    : cfg_(cfg)
{
    if (cfg.modelPath.empty())
        return;
    net_ = cv::dnn::readNetFromONNX(cfg.modelPath);
    CV_Assert(!net_.empty());

    // Warm up once so the first production frame does not pay for allocation.
    score(cv::Mat(cfg.modelInput, CV_8UC3, cv::Scalar::all(0)));
}

float ScratchCheck::score(const cv::Mat& roiBgr)
{
    // The network always sees the ROI at its native input size, never the full frame.
    blob_ = cv::dnn::blobFromImage(roiBgr, cfg_.modelScale, cfg_.modelInput,
                                   cfg_.modelMean, cfg_.modelSwapRB, false);
    net_.setInput(blob_);
    cv::Mat out = net_.forward();
    CV_CheckType(out.depth(), out.depth() == CV_32F, "scratch model must output float32");
    CV_CheckLT((size_t)cfg_.scratchIndex, out.total(), "scratch_index outside model output");
    return out.ptr<float>()[cfg_.scratchIndex];
}

} // namespace inspection
