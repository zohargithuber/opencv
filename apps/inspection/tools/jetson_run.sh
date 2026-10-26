#!/bin/bash
# Build and measure apps/inspection on the target board (Jetson Orin Nano). Run ON the board.
#
#   apps/inspection/tools/jetson_run.sh                                  CLAHE stage benchmark only
#   apps/inspection/tools/jetson_run.sh <config.yaml> <golden> <frames>  + full frame, clahe none/cpu/cuda
#
# Results go to ./jetson-results-<date>/: env.txt (board, clocks, toolchain), build.log,
# clahe-bench.txt, frame-<backend>.txt. The script records clocks but never changes them; for
# max-clock numbers run "sudo nvpmodel -m 0 && sudo jetson_clocks" first.
# Env: BUILD (build dir), OUT (results dir), SKIP_BUILD=1, CLAHE_CLIP (2.0), CLAHE_TILES (8),
#      ITERS (300), STRIPES (1,3), SCALES (1,6).
set +e

SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BUILD=${BUILD:-$SRC/build-jetson}
OUT=${OUT:-$PWD/jetson-results-$(date +%Y%m%d-%H%M%S)}
CLIP=${CLAHE_CLIP:-2.0}
TILES=${CLAHE_TILES:-8}
mkdir -p "$OUT"
[ -x /usr/local/cuda/bin/nvcc ] && export PATH=/usr/local/cuda/bin:$PATH

{
    date
    uname -a
    cat /etc/nv_tegra_release 2>/dev/null
    command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "nvcc: not found"
    echo "cores: $(getconf _NPROCESSORS_ONLN)"
    nvpmodel -q 2>/dev/null
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do [ -r "$f" ] && echo "$f $(cat "$f")"; done
    [ -r /sys/devices/platform/gpu.0/devfreq/17000000.gpu/cur_freq ] && \
        echo "gpu cur_freq $(cat /sys/devices/platform/gpu.0/devfreq/17000000.gpu/cur_freq)"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

if [ "${SKIP_BUILD:-0}" != 1 ]; then
    echo "== building into $BUILD (log: $OUT/build.log)"
    cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_LIST=core,imgproc,imgcodecs,dnn,geometry -DBUILD_APPS_LIST=inspection \
        -DWITH_CUDA=OFF -DINSPECTION_WITH_CUDA=ON -DINSPECTION_CUDA_ARCHITECTURES=87 \
        -DCMAKE_CXX_FLAGS="-mcpu=native" -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
        > "$OUT/build.log" 2>&1 &&
    ninja -C "$BUILD" opencv_inspection opencv_inspection_clahe_bench >> "$OUT/build.log" 2>&1
    if [ $? != 0 ]; then
        echo "BUILD FAILED -- see $OUT/build.log"; tail -20 "$OUT/build.log"; exit 1
    fi
fi
grep -h "opencv_inspection: " "$OUT/build.log" 2>/dev/null

BACKENDS=cpu
grep -q "CUDA CLAHE backend enabled" "$OUT/build.log" 2>/dev/null && BACKENDS=cpu,cuda
[ "${SKIP_BUILD:-0}" = 1 ] && BACKENDS=${BACKENDS_OVERRIDE:-cpu,cuda}
[ "$BACKENDS" = cpu ] && echo "WARNING: CUDA backend not built (nvcc not found) -- cpu only"

echo "== CLAHE stage: $BACKENDS, clip $CLIP, tiles ${TILES}x$TILES"
"$BUILD/bin/opencv_inspection_clahe_bench" --backends=$BACKENDS --scales=${SCALES:-1,6} \
    --stripes=${STRIPES:-1,3} --clip=$CLIP --tiles=$TILES --iters=${ITERS:-300} | tee "$OUT/clahe-bench.txt"

if [ $# -ge 3 ]; then
    CONFIG=$1 GOLDEN=$2 FRAMES=$3
    for b in none ${BACKENDS//,/ }; do
        cfg="$OUT/config-$b.yaml"
        # Same config with only the clahe keys replaced. Thresholds are not re-tuned for CLAHE,
        # so verdicts with cpu/cuda show its effect on the current settings, not tuned results.
        awk -v b="$b" -v clip="$CLIP" -v t="$TILES" '
            /^clahe/ { next }
            { print }
            /^%YAML/ { printf "clahe: %s\nclahe_clip_limit: %s\nclahe_tiles: [ %s, %s ]\n", b, clip, t, t }
        ' "$CONFIG" > "$cfg"
        echo "== full frame, clahe: $b"
        for pass in 1 2 3; do # pass 1 warms caches and the GPU; all three are kept
            echo "-- pass $pass"
            "$BUILD/bin/opencv_inspection" --config="$cfg" --golden="$GOLDEN" "$FRAMES"
        done > "$OUT/frame-$b.txt" 2>&1
        grep "^frames=" "$OUT/frame-$b.txt"
    done
fi
echo "== results in $OUT"
