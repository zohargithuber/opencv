// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "inspection.hpp"

#include <opencv2/imgproc.hpp>

namespace inspection {

template <typename T>
static void readRequired(const cv::FileNode& root, const char* key, T& value)
{
    cv::FileNode n = root[key];
    if (n.empty())
        CV_Error(cv::Error::StsBadArg, std::string("inspection config: missing required key '") + key + "'");
    n >> value;
}

// A length (power 1) or area (power 2) given either in pixels as `key`, or in millimetres as
// `key_mm` / `key_mm2` (converted with px_per_mm). Exactly one form must be present.
static double readPxOrMm(const cv::FileNode& root, const std::string& key, int power, double pxPerMm)
{
    const std::string mmKey = key + (power == 1 ? "_mm" : "_mm2");
    const bool px = !root[key].empty(), mm = !root[mmKey].empty();
    if (px == mm)
        CV_Error(cv::Error::StsBadArg, "inspection config: give exactly one of '" + key + "' (px) or '" + mmKey + "'");
    double v = 0;
    if (px)
    {
        root[key] >> v;
        return v;
    }
    if (!(pxPerMm > 0))
        CV_Error(cv::Error::StsBadArg, "inspection config: '" + mmKey + "' needs px_per_mm > 0");
    root[mmKey] >> v;
    return v * (power == 1 ? pxPerMm : pxPerMm * pxPerMm);
}

Config Config::load(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    CV_Assert(fs.isOpened());
    cv::FileNode root = fs.root();
    // Optional: camera scale, needed only when a length/area key is given in mm.
    double pxPerMm = 0;
    root["px_per_mm"] >> pxPerMm;

    Config c;
    std::vector<int> roi;
    readRequired(root, "roi", roi);
    CV_CheckEQ((int)roi.size(), 4, "roi must be [x, y, width, height]");
    c.roi = cv::Rect(roi[0], roi[1], roi[2], roi[3]);

    // Optional second fixture: two parts per frame (DualInspector).
    if (!root["roi_2"].empty())
    {
        std::vector<int> roi2;
        root["roi_2"] >> roi2;
        CV_CheckEQ((int)roi2.size(), 4, "roi_2 must be [x, y, width, height]");
        c.roi2 = cv::Rect(roi2[0], roi2[1], roi2[2], roi2[3]);
        if (c.roi2.size() != c.roi.size() || !(c.roi2 & c.roi).empty())
            CV_Error(cv::Error::StsBadArg, "inspection config: roi_2 must match roi in size and not overlap it");
        readRequired(root, "empty_min_diff_fraction", c.emptyMinDiffFraction);
        readRequired(root, "belt_gray", c.beltGray);
        readRequired(root, "empty_belt_tolerance", c.emptyBeltTolerance);
    }

    std::string pixelFormat;
    readRequired(root, "pixel_format", pixelFormat);
    if (pixelFormat != "bgr" && pixelFormat != "mono8")
        CV_Error(cv::Error::StsBadArg, "inspection config: pixel_format must be bgr or mono8");
    c.mono = pixelFormat == "mono8";
    if (!c.mono)
    {
        std::string grayMode;
        readRequired(root, "gray_mode", grayMode);
        if (grayMode != "bgr2gray" && grayMode != "channel")
            CV_Error(cv::Error::StsBadArg, "inspection config: gray_mode must be bgr2gray or channel");
        c.grayFromChannel = grayMode == "channel";
        if (c.grayFromChannel)
            readRequired(root, "gray_channel", c.grayChannel);
    }

    // Optional CLAHE before check 1 (absent -> none).
    if (!root["clahe"].empty())
    {
        std::string clahe;
        root["clahe"] >> clahe;
        if (clahe == "cpu")
            c.claheBackend = Config::CLAHE_CPU;
        else if (clahe == "opencl")
            c.claheBackend = Config::CLAHE_OPENCL;
        else if (clahe == "cuda")
            c.claheBackend = Config::CLAHE_CUDA;
        else if (clahe != "none")
            CV_Error(cv::Error::StsBadArg, "inspection config: clahe must be none, cpu, opencl or cuda");
    }
    if (c.claheBackend != Config::CLAHE_NONE)
    {
        std::vector<int> tiles;
        readRequired(root, "clahe_clip_limit", c.claheClipLimit);
        readRequired(root, "clahe_tiles", tiles);
        CV_CheckEQ((int)tiles.size(), 2, "clahe_tiles must be [x, y]");
        c.claheTiles = cv::Size(tiles[0], tiles[1]);
        if (tiles[0] < 1 || tiles[1] < 1 || tiles[0] > 16 || tiles[1] > 16)
            CV_Error(cv::Error::StsBadArg, "inspection config: clahe_tiles must be 1..16 each");
        if (c.roi.width % tiles[0] != 0 || c.roi.height % tiles[1] != 0)
            CV_Error(cv::Error::StsBadArg, "inspection config: roi width/height must be multiples of clahe_tiles");
        CV_CheckGE(c.claheClipLimit, 0.0, "clahe_clip_limit must be >= 0");
    }

    readRequired(root, "diff_threshold", c.diffThreshold);
    c.maxDiffArea = readPxOrMm(root, "max_diff_area", 2, pxPerMm);
    readRequired(root, "frame_budget_ms", c.frameBudgetMs);
    readRequired(root, "stripe_threads", c.stripeThreads);
    CV_CheckGE(c.stripeThreads, 1, "stripe_threads must be >= 1");

    std::string scratchMethod;
    readRequired(root, "scratch_method", scratchMethod);
    if (scratchMethod == "none")
        c.scratchMethod = Config::SCRATCH_NONE;
    else if (scratchMethod == "diff")
        c.scratchMethod = Config::SCRATCH_DIFF;
    else if (scratchMethod == "dnn")
        c.scratchMethod = Config::SCRATCH_DNN;
    else
        CV_Error(cv::Error::StsBadArg, "inspection config: scratch_method must be none, diff or dnn");

    if (c.scratchMethod == Config::SCRATCH_DIFF)
    {
        readRequired(root, "scratch_diff_threshold", c.scratchDiffThreshold);
        c.scratchMinLength = readPxOrMm(root, "scratch_min_length", 1, pxPerMm);
        readRequired(root, "scratch_min_elongation", c.scratchMinElongation);
        c.scratchEdgeMargin = cvRound(readPxOrMm(root, "scratch_edge_margin", 1, pxPerMm));
        c.scratchMaxCandidates = cvRound(readPxOrMm(root, "scratch_max_candidates", 2, pxPerMm));
        CV_CheckLT(c.scratchDiffThreshold, c.diffThreshold,
                   "scratch_diff_threshold must be below diff_threshold");
        CV_CheckGE(c.scratchMinLength, 1.0, "scratch_min_length must be >= 1");
        CV_CheckGE(c.scratchEdgeMargin, 0, "scratch_edge_margin must be >= 0");
        CV_CheckGE((double)c.scratchMaxCandidates, c.scratchMinLength,
                   "scratch_max_candidates must be >= scratch_min_length");
    }

    if (c.scratchMethod == Config::SCRATCH_DNN && c.mono)
        CV_Error(cv::Error::StsBadArg, "inspection config: scratch_method dnn needs pixel_format bgr");
    if (c.scratchMethod == Config::SCRATCH_DNN)
    {
        root["model_path"] >> c.modelPath; // may be empty and set by --model instead
        std::vector<int> size;
        std::vector<double> mean;
        readRequired(root, "model_input", size);
        CV_CheckEQ((int)size.size(), 2, "model_input must be [width, height]");
        c.modelInput = cv::Size(size[0], size[1]);
        readRequired(root, "model_scale", c.modelScale);
        readRequired(root, "model_mean", mean);
        for (size_t i = 0; i < mean.size() && i < 4; i++)
            c.modelMean[(int)i] = mean[i];
        int swapRB = 0;
        readRequired(root, "model_swap_rb", swapRB);
        c.modelSwapRB = swapRB != 0;
        readRequired(root, "scratch_index", c.scratchIndex);
        readRequired(root, "scratch_threshold", c.scratchThreshold);
    }
    return c;
}

void toGray(const cv::Mat& src, cv::Mat& gray, const Config& cfg)
{
    if (cfg.mono)
        gray = src; // header only: the camera buffer is used in place
    else if (cfg.grayFromChannel)
        cv::extractChannel(src, gray, cfg.grayChannel);
    else
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
}

// Pixels further than empty_belt_tolerance from belt_gray. Branch-free so it vectorizes;
// no allocation.
static int countNonBelt(const cv::Mat& gray, const Config& cfg)
{
    const uchar lo = cv::saturate_cast<uchar>(cfg.beltGray - cfg.emptyBeltTolerance);
    const uchar hi = cv::saturate_cast<uchar>(cfg.beltGray + cfg.emptyBeltTolerance);
    int count = 0;
    for (int y = 0; y < gray.rows; y++)
    {
        const uchar* __restrict g = gray.ptr<uchar>(y);
        unsigned rowCount = 0;
        for (int x = 0; x < gray.cols; x++)
            rowCount += (unsigned)(g[x] < lo) + (unsigned)(g[x] > hi);
        count += (int)rowCount;
    }
    return count;
}

Inspector::Inspector(const Config& cfg, const cv::Mat& goldenBgr)
    : cfg_(cfg), clahe_(cfg), missing_(cfg, goldenBgr, clahe_), scratchDiff_(cfg, missing_.golden()), scratch_(cfg)
{
}

InspectionResult Inspector::inspectFrame(const cv::Mat& bgr)
{
    CV_CheckType(bgr.type(), bgr.type() == frameType(cfg_), "frame type must match pixel_format (bgr: 8UC3, mono8: 8UC1)");
    InspectionResult r;
    cv::TickMeter total, stage;
    total.start();

    // Crop before any per-pixel work: every stage below scales with ROI area.
    const cv::Mat roi = bgr(cfg_.roi);

    stage.start();
    toGray(roi, gray_, cfg_);
    stage.stop(); r.ms.gray = stage.getTimeMilli(); stage.reset();

    const cv::Mat* gray = &gray_;
    if (clahe_.enabled())
    {
        stage.start();
        gray = &clahe_.apply(gray_);
        stage.stop(); r.ms.clahe = stage.getTimeMilli(); stage.reset();
    }

    cv::Rect box;
    double area = 0;
    stage.start();
    bool missing = missing_.run(*gray, box, area);
    stage.stop(); r.ms.missing = stage.getTimeMilli(); stage.reset();

    if (missing)
    {
        // Two-slot mode: an empty fixture differs everywhere AND is bare belt almost everywhere.
        // Anything else that differs this much (fragment, shifted part) stays MISSING, i.e.
        // rejected. Runs only on this path, never on a good part. belt_gray is a raw gray level,
        // so with CLAHE the test reads the unequalized (and unblurred) ROI.
        const bool empty = !cfg_.roi2.empty() &&
            missing_.diffPixels() > cfg_.emptyMinDiffFraction * cfg_.roi.area() &&
            countNonBelt(clahe_.enabled() ? gray_ : missing_.blurred(), cfg_) <= cfg_.maxDiffArea;
        if (empty)
        {
            r.present = false;
            r.reason = "EMPTY";
        }
        else
        {
            r.ok = false;
            r.reason = "MISSING";
            r.box = box + cfg_.roi.tl();
            r.score = (float)area;
        }
    }
    else if (cfg_.scratchMethod == Config::SCRATCH_DIFF)
    {
        stage.start();
        ScratchDiffCheck::Outcome scratch = scratchDiff_.run(missing_.diff(), box, r.score);
        stage.stop(); r.ms.scratch = stage.getTimeMilli();
        if (scratch == ScratchDiffCheck::SCRATCH)
        {
            r.ok = false;
            r.reason = "SCRATCH";
            r.box = box + cfg_.roi.tl();
        }
        else if (scratch == ScratchDiffCheck::NOISY)
        {
            // Fail safe: a part we could not examine within budget is never passed.
            r.ok = false;
            r.reason = "NOISY";
            r.box = cfg_.roi;
        }
    }
    else if (cfg_.scratchMethod == Config::SCRATCH_DNN && scratch_.enabled())
    {
        stage.start();
        r.score = scratch_.score(roi);
        stage.stop(); r.ms.scratch = stage.getTimeMilli();
        if (r.score > cfg_.scratchThreshold)
        {
            r.ok = false;
            r.reason = "SCRATCH";
            r.box = cfg_.roi;
        }
    }

    total.stop();
    r.ms.total = total.getTimeMilli();
    if (r.ok)
        r.box = cfg_.roi;
    return r;
}

} // namespace inspection
