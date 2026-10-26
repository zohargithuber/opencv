// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "inspection.hpp"

namespace inspection {

static Config slotConfig(const Config& cfg, const cv::Rect& roi)
{
    Config c = cfg;
    c.roi = roi;
    return c;
}

DualInspector::DualInspector(const Config& cfg, const cv::Mat& goldenBgr)
    : cfg_{ cfg, slotConfig(cfg, cfg.roi2) },
      a_(cfg_[0], goldenBgr), b_(cfg_[1], goldenBgr),
      thread_(&DualInspector::worker, this)
{
    CV_Assert(!cfg.roi2.empty());
}

DualInspector::~DualInspector()
{
    {
        std::lock_guard<std::mutex> lock(m_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
}

void DualInspector::worker()
{
    uint64_t seen = 0;
    for (;;)
    {
        const cv::Mat* frame;
        {
            std::unique_lock<std::mutex> lock(m_);
            cv_.wait(lock, [&] { return stop_ || posted_ != seen; });
            if (stop_)
                return;
            seen = posted_;
            frame = job_;
        }
        try
        {
            resB_ = b_.inspectFrame(*frame);
        }
        catch (...)
        {
            err_ = std::current_exception(); // error path only; steady state never gets here
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            done_ = seen;
        }
        cv_.notify_all();
    }
}

void DualInspector::inspectFrame(const cv::Mat& bgr, InspectionResult out[2])
{
    uint64_t ticket;
    {
        std::lock_guard<std::mutex> lock(m_);
        job_ = &bgr;
        ticket = ++posted_;
    }
    cv_.notify_all();

    // Slot A on this thread while the worker runs slot B. Wait for B even if A throws, so the
    // worker never reads a frame the caller has already released.
    std::exception_ptr errA;
    try
    {
        out[0] = a_.inspectFrame(bgr);
    }
    catch (...)
    {
        errA = std::current_exception();
    }
    {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return done_ == ticket; });
    }
    if (errA)
        std::rethrow_exception(errA);
    if (err_)
    {
        std::exception_ptr e = err_;
        err_ = nullptr;
        std::rethrow_exception(e);
    }
    out[1] = resB_;
}

} // namespace inspection
