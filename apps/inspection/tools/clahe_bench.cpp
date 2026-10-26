// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// CLAHE stage benchmark for the target board: inspection::Clahe per backend vs cv::CLAHE.
// For each backend / ROI scale / stripe count: exactness against cv::CLAHE, then the time of
// one part and of two parts in parallel (slot B on a persistent thread, as DualInspector).

#include "inspection.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <sstream>
#include <thread>
#include <vector>

using namespace inspection;

static const char* keys =
    "{ help h   |          | print this help }"
    "{ backends | cpu,cuda | comma-separated: cpu, opencl, cuda }"
    "{ scales   | 1,6      | ROI scale per side vs 640x480 }"
    "{ stripes  | 1,3      | stripe_threads values (cpu only) }"
    "{ clip     | 2.0      | clahe_clip_limit }"
    "{ tiles    | 8        | clahe_tiles (square) }"
    "{ iters    | 200      | timed iterations }"
    "{ budget   | 5.0      | ms, two-part CLAHE budget checked against the worst iteration }";

static std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty())
            out.push_back(t);
    return out;
}

// Synthetic part: gradient (uneven light), sensor noise, and blobs of random gray.
static cv::Mat scene(cv::Size sz, int seed)
{
    cv::RNG rng(seed);
    cv::Mat m(sz, CV_8UC1), n(sz, CV_8UC1);
    for (int y = 0; y < sz.height; y++)
        for (int x = 0; x < sz.width; x++)
            m.at<uchar>(y, x) = cv::saturate_cast<uchar>(40 + 120.0 * x / sz.width + 50.0 * y / sz.height);
    rng.fill(n, cv::RNG::NORMAL, 0, 10);
    cv::add(m, n, m);
    for (int i = 0; i < 30; i++)
        cv::circle(m, cv::Point(rng.uniform(0, sz.width), rng.uniform(0, sz.height)),
                   rng.uniform(5, sz.width / 6), cv::Scalar(rng.uniform(0, 256)), cv::FILLED);
    return m;
}

struct Stats { double median, p99, worst; };

static Stats stats(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return { v[v.size() / 2], v[std::min(v.size() - 1, v.size() * 99 / 100)], v.back() };
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    const int iters = parser.get<int>("iters"), tiles = parser.get<int>("tiles");
    const double clip = parser.get<double>("clip"), budget = parser.get<double>("budget");
    int fails = 0;

    for (const std::string& backend : split(parser.get<std::string>("backends")))
    for (const std::string& scaleStr : split(parser.get<std::string>("scales")))
    for (const std::string& stripeStr : split(backend == "cpu" ? parser.get<std::string>("stripes") : "1"))
    {
        const int scale = std::stoi(scaleStr);
        const cv::Size sz(640 * scale, 480 * scale);
        Config cfg;
        cfg.roi = cv::Rect(cv::Point(), sz);
        cfg.claheBackend = backend == "cpu" ? Config::CLAHE_CPU
                         : backend == "opencl" ? Config::CLAHE_OPENCL
                         : backend == "cuda" ? Config::CLAHE_CUDA : Config::CLAHE_NONE;
        CV_Assert(cfg.claheBackend != Config::CLAHE_NONE);
        cfg.claheClipLimit = clip;
        cfg.claheTiles = cv::Size(tiles, tiles);
        cfg.stripeThreads = std::stoi(stripeStr);
        const Config cfgB = cfg;

        std::printf("%-6s %dx %4dx%-4d stripes=%d  ", backend.c_str(), scale, sz.width, sz.height, cfg.stripeThreads);
        std::fflush(stdout);
        try
        {
            Clahe a(cfg), b(cfgB);
            const cv::Mat srcA = scene(sz, 1), srcB = scene(sz, 2);

            cv::Mat ref, d;
            cv::createCLAHE(clip, cfg.claheTiles)->apply(srcA, ref);
            cv::absdiff(a.apply(srcA), ref, d);
            double maxDiff = 0;
            cv::minMaxLoc(d, nullptr, &maxDiff);
            const int differ = cv::countNonZero(d);

            for (int i = 0; i < 5; i++) // warm-up: first CUDA launch, page faults, thread wake
            {
                a.apply(srcA);
                b.apply(srcB);
            }

            std::vector<double> one, two;
            cv::TickMeter tm;
            for (int i = 0; i < iters; i++)
            {
                tm.reset(); tm.start();
                a.apply(srcA);
                tm.stop();
                one.push_back(tm.getTimeMilli());
            }

            // Slot B on a persistent thread; per-iteration hand-off through two counters.
            std::atomic<int> go(0), done(0);
            std::atomic<bool> stop(false);
            std::thread worker([&] {
                for (int seen = 0;;)
                {
                    while (go.load(std::memory_order_acquire) == seen)
                        if (stop.load(std::memory_order_acquire))
                            return;
                    seen++;
                    b.apply(srcB);
                    done.fetch_add(1, std::memory_order_release);
                }
            });
            for (int i = 0; i < iters; i++)
            {
                tm.reset(); tm.start();
                go.fetch_add(1, std::memory_order_release);
                a.apply(srcA);
                while (done.load(std::memory_order_acquire) != i + 1)
                    ;
                tm.stop();
                two.push_back(tm.getTimeMilli());
            }
            stop.store(true, std::memory_order_release);
            worker.join();

            const Stats s1 = stats(one), s2 = stats(two);
            const bool pass = s2.worst <= budget;
            fails += !pass;
            std::printf("vs cv::CLAHE: %d px differ (max %g)  1 part: median %.2f p99 %.2f max %.2f ms  "
                        "2 parts: median %.2f p99 %.2f max %.2f ms  [%s %.1f ms]\n",
                        differ, maxDiff, s1.median, s1.p99, s1.worst, s2.median, s2.p99, s2.worst,
                        pass ? "PASS" : "FAIL", budget);
        }
        catch (const std::exception& e)
        {
            fails++;
            std::printf("ERROR: %s\n", e.what());
        }
    }
    return fails ? 2 : 0;
}
