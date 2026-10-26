// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "inspection.hpp"

#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

namespace inspection {

MissingPartCheck::MissingPartCheck(const Config& cfg, const cv::Mat& goldenBgr, Clahe& clahe)
    : cfg_(cfg)
{
    CV_CheckType(goldenBgr.type(), goldenBgr.type() == frameType(cfg), "golden type must match pixel_format");
    CV_Assert((cfg.roi & cv::Rect(0, 0, goldenBgr.cols, goldenBgr.rows)) == cfg.roi);
    cv::Mat gray;
    toGray(goldenBgr(cfg.roi), gray, cfg);
    cv::GaussianBlur(clahe.enabled() ? clahe.apply(gray) : gray, golden_, cv::Size(5, 5), 0);
}

bool MissingPartCheck::run(const cv::Mat& roiGray, cv::Rect& box, double& area)
{
    // BORDER_ISOLATED: with mono8 roiGray is a view into the camera frame; never read pixels
    // outside the ROI (keeps the result identical to the bgr path and avoids extra buffers).
    cv::GaussianBlur(roiGray, blurred_, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT | cv::BORDER_ISOLATED);
    cv::absdiff(blurred_, golden_, diff_);
    cv::threshold(diff_, mask_, cfg_.diffThreshold, 255, cv::THRESH_BINARY);

    // Fast exit: a clean part leaves (almost) no differing pixels.
    diffPixels_ = cv::countNonZero(mask_);
    if (diffPixels_ <= cfg_.maxDiffArea)
        return false;

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    area = 0;
    for (const auto& c : contours)
    {
        double a = cv::contourArea(c);
        if (a > area)
        {
            area = a;
            box = cv::boundingRect(c);
        }
    }
    return area > cfg_.maxDiffArea;
}

} // namespace inspection
