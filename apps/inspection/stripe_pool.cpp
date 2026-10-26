// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "inspection.hpp"

namespace inspection {

StripePool::StripePool(int threads)
    : n_(threads)
{
    CV_Assert(threads >= 1);
    workers_.reserve(threads - 1);
    for (int i = 1; i < threads; i++)
        workers_.emplace_back(&StripePool::worker, this, i);
}

StripePool::~StripePool()
{
    {
        std::lock_guard<std::mutex> lock(m_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_)
        t.join();
}

void StripePool::worker(int index)
{
    uint64_t seen = 0;
    for (;;)
    {
        Job* job;
        int rows;
        {
            std::unique_lock<std::mutex> lock(m_);
            cv_.wait(lock, [&] { return stop_ || posted_ != seen; });
            if (stop_)
                return;
            seen = posted_;
            job = job_;
            rows = rows_;
        }
        std::exception_ptr err;
        try
        {
            job->run(rows * index / n_, rows * (index + 1) / n_);
        }
        catch (...)
        {
            err = std::current_exception(); // error path only
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            if (err && !err_)
                err_ = err;
            if (--pending_ == 0)
                done_.notify_one();
        }
    }
}

void StripePool::run(Job& job, int rows)
{
    if (n_ == 1)
    {
        job.run(0, rows);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_);
        job_ = &job;
        rows_ = rows;
        pending_ = n_ - 1;
        ++posted_;
    }
    cv_.notify_all();

    // Stripe 0 on this thread; wait for the others even if it throws (they use job).
    std::exception_ptr err;
    try
    {
        job.run(0, rows / n_);
    }
    catch (...)
    {
        err = std::current_exception();
    }
    std::unique_lock<std::mutex> lock(m_);
    done_.wait(lock, [&] { return pending_ == 0; });
    if (!err && err_)
        err = err_;
    err_ = nullptr;
    lock.unlock();
    if (err)
        std::rethrow_exception(err);
}

} // namespace inspection
