// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_APPS_INSPECTION_HPP
#define OPENCV_APPS_INSPECTION_HPP

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace inspection {

// All tuning values come from inspection.yaml; nothing is defaulted in code.
struct Config
{
    cv::Rect roi;                 // part fixture location in the full frame
    cv::Rect roi2;                // optional second fixture (roi_2); empty -> single-part mode

    // Required with roi_2: a slot is EMPTY (holds no part) when check 1's diff covers more than
    // emptyMinDiffFraction of the ROI and at most max_diff_area of its blurred pixels are
    // further than emptyBeltTolerance from beltGray. A fragment or shifted part stays MISSING.
    double emptyMinDiffFraction = 0;
    double beltGray = 0;
    double emptyBeltTolerance = 0;
    bool mono = false;            // pixel_format: mono8 -> frames are 8-bit gray, used in place
    bool grayFromChannel = false; // bgr only. false: cvtColor BGR2GRAY, true: extractChannel
    int grayChannel = 1;          // used when grayFromChannel is true (0=B, 1=G, 2=R)

    // Optional contrast-limited adaptive histogram equalization of the gray ROI, applied to
    // live frames and the golden alike before check 1. Absent key -> CLAHE_NONE.
    enum ClaheBackend { CLAHE_NONE, CLAHE_CPU, CLAHE_OPENCL, CLAHE_CUDA };
    ClaheBackend claheBackend = CLAHE_NONE;
    double claheClipLimit = 0;    // cv::CLAHE clip limit (contrast multiple per tile)
    cv::Size claheTiles;          // tile grid; roi width/height must be multiples of it

    double diffThreshold = 0;     // threshold() level on |roi - golden|
    double maxDiffArea = 0;       // largest diff blob above this (px) -> NOK: MISSING

    enum ScratchMethod { SCRATCH_NONE, SCRATCH_DIFF, SCRATCH_DNN };
    ScratchMethod scratchMethod = SCRATCH_NONE;

    // scratch_method: diff -- thin, elongated blobs in check 1's |roi - golden| image
    double scratchDiffThreshold = 0; // threshold() level, below diffThreshold
    double scratchMinLength = 0;     // px, long side of the blob's minAreaRect
    double scratchMinElongation = 0; // long side / short side
    int scratchEdgeMargin = 0;       // px around golden part edges that are ignored
    int scratchMaxCandidates = 0;    // most mask pixels examined per frame; above -> NOK: NOISY

    // scratch_method: dnn
    std::string modelPath;        // empty -> scratch check disabled
    cv::Size modelInput;
    double modelScale = 1.0;
    cv::Scalar modelMean;
    bool modelSwapRB = false;
    int scratchIndex = 0;         // output element holding the scratch score
    float scratchThreshold = 0.f;

    double frameBudgetMs = 0;
    int stripeThreads = 1;        // threads per part for the scratch pixel pass (1 = this thread)

    static Config load(const std::string& path);
};

struct StageTimes
{
    double gray = 0, clahe = 0, missing = 0, scratch = 0, total = 0; // milliseconds
};

struct InspectionResult
{
    bool ok = true;
    bool present = true; // false: EMPTY slot, no part to judge (ok stays true, nothing rejected)
    std::string reason;  // "MISSING", "SCRATCH" or "NOISY" when !ok; "EMPTY" when !present
    cv::Rect box;        // in full-frame coordinates
    float score = 0.f;   // scratch score, or diff area for MISSING
    StageTimes ms;
};

class Clahe;

// Check 1: compare the part ROI against the golden reference.
class MissingPartCheck
{
public:
    // clahe: applied to the golden ROI here when enabled, as Inspector does to every frame.
    MissingPartCheck(const Config& cfg, const cv::Mat& goldenBgr, Clahe& clahe);
    // roiGray: grayscale ROI. Returns true if a part is missing; box is ROI-relative.
    bool run(const cv::Mat& roiGray, cv::Rect& box, double& area);
    // Valid after run(): |blurred roi - golden|, reused by the scratch check.
    const cv::Mat& diff() const { return diff_; }
    const cv::Mat& golden() const { return golden_; }
    const cv::Mat& blurred() const { return blurred_; }
    // Valid after run(): pixels of diff() above diff_threshold.
    int diffPixels() const { return diffPixels_; }

private:
    const Config& cfg_;
    cv::Mat golden_, blurred_, diff_, mask_;
    int diffPixels_ = 0;
};

// Persistent stripe workers: run(job, rows) splits [0, rows) into threads() horizontal stripes,
// stripe 0 on the calling thread. Threads are created once; run() does not allocate.
class StripePool
{
public:
    struct Job
    {
        virtual ~Job() = default;
        virtual void run(int y0, int y1) = 0;
    };
    explicit StripePool(int threads);
    ~StripePool();
    void run(Job& job, int rows);
    int threads() const { return n_; }

private:
    void worker(int index);

    const int n_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    Job* job_ = nullptr;
    int rows_ = 0, pending_ = 0;
    uint64_t posted_ = 0;
    bool stop_ = false;
    std::exception_ptr err_;
};

// CLAHE on the gray ROI (clahe: cpu | opencl | cuda), same result as cv::CLAHE for 8-bit input.
// cpu: per-tile LUTs, then a bilinear pass that blends the two tile rows once per image row;
// both passes split into stripes on persistent workers (stripe_threads). cuda: one kernel for
// the LUTs, one for the pass, buffers and stream created here. Neither allocates in apply().
// opencl: cv::CLAHE on a UMat (Intel iGPU), which allocates per call.
class Clahe
{
public:
    explicit Clahe(const Config& cfg); // sized for cfg.roi
    ~Clahe();
    bool enabled() const { return cfg_.claheBackend != Config::CLAHE_NONE; }
    // src: 8UC1, roi-sized, may be a view. The result is valid until the next call.
    const cv::Mat& apply(const cv::Mat& src);

private:
    struct Pass : StripePool::Job
    {
        Clahe* self = nullptr;
        bool lut = false;
        void run(int y0, int y1) override { lut ? self->lutTiles(y0, y1) : self->interpRows(y0, y1); }
    };
    struct Cuda;
    void lutTiles(int k0, int k1);   // tiles [k0, k1) of src_ -> lut_ rows
    void interpRows(int y0, int y1); // image rows [y0, y1) of src_ -> dst_

    const Config& cfg_;
    cv::Size tile_;
    int clipLimit_ = 0;
    float scale_ = 0.f;
    const cv::Mat* src_ = nullptr;
    cv::Mat dst_, lut_;              // lut_: one 256-entry row per tile
    std::vector<int> ind1_, ind2_;   // per column: left/right tile LUT offset
    std::vector<int> spanEnd_;       // per column: end of its run of equal (ind1_, ind2_)
    std::vector<float> xa_, xa1_;    // per column: right/left tile weight
    Pass lutPass_, interpPass_;
    std::unique_ptr<StripePool> pool_; // cpu, stripe_threads > 1 only
    std::unique_ptr<Cuda> cuda_;
    cv::Ptr<cv::CLAHE> ocl_;
    cv::UMat usrc_, udst_;
};

// Check 2 (scratch_method: diff): thin, elongated blobs in check 1's difference image.
// Costs a few ROI passes on top of check 1; no model needed. Per-frame work is bounded by
// scratch_max_candidates and run() does not allocate: all buffers are sized in the constructor.
class ScratchDiffCheck
{
public:
    enum Outcome { CLEAN, SCRATCH, NOISY };

    ScratchDiffCheck(const Config& cfg, const cv::Mat& goldenGray);
    // diff: MissingPartCheck::diff(). box is ROI-relative and set only for SCRATCH.
    // NOISY: more candidate pixels than scratch_max_candidates; the part was not examined.
    Outcome run(const cv::Mat& diff, cv::Rect& box, float& length);

private:
    struct Blob { int n, x0, y0, x1, y1; double sx, sy, sxx, syy, sxy; };
    int findRoot(int l);
    // threshold + AND allow_ for rows [y0, y1): writes mask_ rows and rowCount_[y].
    void maskRows(const cv::Mat& diff, int y0, int y1);

    const Config& cfg_;
    cv::Mat allow_, mask_;        // allow_: 255 away from golden part edges
    std::vector<int> prevRow_, curRow_, parent_; // provisional labels; parent_ is union-find
    std::vector<int> rowCount_;   // candidates per row from the fused pass; 0 -> row skipped
    std::vector<Blob> blobs_;     // per provisional label, merged into roots after the scan
    std::unique_ptr<StripePool> pool_; // stripe_threads > 1 only
};

// Check 2 (scratch_method: dnn): neural-network scratch classifier on the part ROI.
class ScratchCheck
{
public:
    explicit ScratchCheck(const Config& cfg);
    bool enabled() const { return !net_.empty(); }
    float score(const cv::Mat& roiBgr);

private:
    const Config& cfg_;
    cv::dnn::Net net_;
    cv::Mat blob_;
};

// mono8: gray becomes a view of src (no copy, no allocation). bgr: converted into gray.
void toGray(const cv::Mat& src, cv::Mat& gray, const Config& cfg);
// The frame/golden type the config expects: CV_8UC1 for mono8, CV_8UC3 for bgr.
inline int frameType(const Config& cfg) { return cfg.mono ? CV_8UC1 : CV_8UC3; }

class Inspector
{
public:
    Inspector(const Config& cfg, const cv::Mat& goldenBgr);
    InspectionResult inspectFrame(const cv::Mat& bgr);
    bool scratchEnabled() const
    {
        return cfg_.scratchMethod == Config::SCRATCH_DIFF ||
               (cfg_.scratchMethod == Config::SCRATCH_DNN && scratch_.enabled());
    }

private:
    const Config& cfg_;
    Clahe clahe_;              // before missing_: it equalizes the golden
    MissingPartCheck missing_; // before scratchDiff_: it is built from missing_.golden()
    ScratchDiffCheck scratchDiff_;
    ScratchCheck scratch_;
    cv::Mat gray_;
};

// Two fixtures in one frame: slot A on the calling thread, slot B on a persistent worker.
// The worker is created once; the per-frame hand-off (mutex + condition variable) does not
// allocate. Each slot is an independent Inspector with its own Config copy and buffers.
class DualInspector
{
public:
    DualInspector(const Config& cfg, const cv::Mat& goldenBgr);
    ~DualInspector();
    // out[0] = slot A (roi), out[1] = slot B (roi_2).
    void inspectFrame(const cv::Mat& bgr, InspectionResult out[2]);

private:
    void worker();

    Config cfg_[2];
    Inspector a_, b_;
    std::mutex m_;
    std::condition_variable cv_;
    const cv::Mat* job_ = nullptr;
    uint64_t posted_ = 0, done_ = 0;
    bool stop_ = false;
    InspectionResult resB_;
    std::exception_ptr err_;
    std::thread thread_; // last: starts after every member it uses is constructed
};

} // namespace inspection

#endif // OPENCV_APPS_INSPECTION_HPP
