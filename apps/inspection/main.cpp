// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "inspection.hpp"

#include <cstdio>
#include <memory>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>

static const char* keys =
    "{ help h  |       | print this help }"
    "{ config  |       | inspection.yaml with ROI, thresholds, model and frame budget }"
    "{ golden  |       | golden reference frame (8-bit BGR, same size as production frames) }"
    "{ model   |       | ONNX scratch model (overrides model_path in config) }"
    "{ out     |       | directory for OK/NOK overlay images (optional) }"
    "{ @input  |       | frame file or directory of frames }";

static std::string baseName(const std::string& path)
{
    size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

// Overlays are drawn in colour; mono frames are expanded for the --out images only.
static cv::Mat toOverlayImage(const cv::Mat& frame)
{
    if (frame.channels() == 3)
        return frame;
    cv::Mat vis;
    cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
    return vis;
}

static void drawOverlay(cv::Mat& img, const inspection::InspectionResult& r)
{
    const cv::Scalar color = !r.present ? cv::Scalar(140, 140, 140)
                           : r.ok ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 230);
    const std::string label = !r.present ? "EMPTY" : r.ok ? "OK" : "NOK: " + r.reason;
    cv::rectangle(img, r.box, color, 3);
    int baseline = 0;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
    cv::Point org(r.box.x, std::max(r.box.y - 8, ts.height + 8));
    cv::rectangle(img, cv::Rect(org.x, org.y - ts.height - 8, ts.width + 12, ts.height + 14), color, cv::FILLED);
    cv::putText(img, label, org + cv::Point(6, 0), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar::all(255), 2);
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV inspection: missing-part check + scratch check per frame");
    if (parser.has("help") || !parser.has("config") || !parser.has("golden") || !parser.has("@input"))
    {
        parser.printMessage();
        return parser.has("help") ? 0 : 1;
    }

    inspection::Config cfg = inspection::Config::load(parser.get<std::string>("config"));
    if (parser.has("model"))
    {
        // The model's preprocessing keys are only read for scratch_method: dnn.
        CV_Check(cfg.scratchMethod, cfg.scratchMethod == inspection::Config::SCRATCH_DNN,
                 "--model needs scratch_method: dnn in the config");
        cfg.modelPath = parser.get<std::string>("model");
    }

    const int readMode = cfg.mono ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    cv::Mat golden = cv::imread(parser.get<std::string>("golden"), readMode);
    CV_Assert(!golden.empty());
    const bool dual = !cfg.roi2.empty();
    inspection::Inspector inspector(cfg, golden);
    std::unique_ptr<inspection::DualInspector> pair;
    if (dual)
        pair.reset(new inspection::DualInspector(cfg, golden)); // slot A = roi, slot B = roi_2
    if (!inspector.scratchEnabled())
        std::printf("note: scratch_method is none or has no model -- check 2 (SCRATCH) is disabled\n");

    const std::string input = parser.get<std::string>("@input");
    std::vector<cv::String> frames;
    if (cv::utils::fs::isDirectory(input))
        cv::glob(input, frames, false);
    else
        frames.push_back(input);

    const std::string outDir = parser.get<std::string>("out");
    if (!outDir.empty())
        cv::utils::fs::createDirectories(outDir);

    int nok = 0, overBudget = 0, n = 0;
    double worstMs = 0;
    for (const auto& path : frames)
    {
        cv::Mat frame = cv::imread(path, readMode);
        if (frame.empty())
            continue;
        CV_CheckEQ(frame.size(), golden.size(), "frame size must match the golden frame");
        if (dual)
        {
            // The budget applies to the whole frame: both slots, measured around the pair.
            inspection::InspectionResult rs[2];
            cv::TickMeter tm;
            tm.start();
            pair->inspectFrame(frame, rs);
            tm.stop();
            const double frameMs = tm.getTimeMilli();
            n++;
            worstMs = std::max(worstMs, frameMs);
            const bool over = frameMs > cfg.frameBudgetMs;
            overBudget += over;
            std::printf("%s", baseName(path).c_str());
            for (int s = 0; s < 2; s++)
            {
                const inspection::InspectionResult& r = rs[s];
                nok += r.present && !r.ok;
                std::printf("  %c: %-14s %5.2f ms", s ? 'B' : 'A',
                            !r.present ? "EMPTY" : r.ok ? "OK" : ("NOK: " + r.reason).c_str(), r.ms.total);
            }
            std::printf("  frame=%.2f ms%s\n", frameMs, over ? "  OVER BUDGET" : "");
            if (!outDir.empty())
            {
                cv::Mat vis = toOverlayImage(frame);
                drawOverlay(vis, rs[0]);
                drawOverlay(vis, rs[1]);
                std::string name = baseName(path);
                cv::imwrite(cv::utils::fs::join(outDir, name.substr(0, name.find_last_of('.')) + ".png"), vis);
            }
            continue;
        }
        inspection::InspectionResult r = inspector.inspectFrame(frame);
        n++;
        nok += !r.ok;
        worstMs = std::max(worstMs, r.ms.total);
        const bool over = r.ms.total > cfg.frameBudgetMs;
        overBudget += over;
        char clahe[32] = "";
        if (cfg.claheBackend != inspection::Config::CLAHE_NONE)
            std::snprintf(clahe, sizeof(clahe), " clahe=%.2f", r.ms.clahe);
        std::printf("%s  %-14s score=%.3f  gray=%.2f%s missing=%.2f scratch=%.2f total=%.2f ms%s\n",
                    baseName(path).c_str(),
                    r.ok ? "OK" : ("NOK: " + r.reason).c_str(), r.score,
                    r.ms.gray, clahe, r.ms.missing, r.ms.scratch, r.ms.total,
                    over ? "  OVER BUDGET" : "");
        if (!outDir.empty())
        {
            cv::Mat vis = toOverlayImage(frame);
            drawOverlay(vis, r);
            std::string name = baseName(path);
            cv::imwrite(cv::utils::fs::join(outDir, name.substr(0, name.find_last_of('.')) + ".png"), vis);
        }
    }
    std::printf("frames=%d  nok=%d  worst=%.2f ms  budget=%.2f ms  over_budget=%d\n",
                n, nok, worstMs, cfg.frameBudgetMs, overBudget);
    return overBudget > 0 ? 2 : 0;
}
