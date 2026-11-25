# -*- coding: utf-8 -*-
"""一条命令重跑网页与 PPT 上的所有数字。

    python run.py --all        # 全部步骤（约 30 秒，纯 CPU）
    python run.py --infer      # 1) ONNX 推理 valid / test
    python run.py --eval       # 2) 与真值比对，出指标 json
    python run.py --failures   # 3) 挑出代价最高的错例
    python run.py --painting   # 4) 跑 demo/demo2/fails 并渲染三视图
    python run.py --web        # 5) 重建离线网页技术报告
    python run.py --ppt        # 6) 重建答辩 PPT
    python run.py --check      # 7) 校验网页素材与内嵌数据

每一步都只是调用 tools/ 下的脚本，所以网页上的每个数字都能追到一条
实际执行过的命令。
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(name, args, quiet=False):
    cmd = [PY] + args
    print(f"\n[{name}] {' '.join(args)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  !! 失败（退出码 {r.returncode}）")
        sys.exit(r.returncode)
    print(f"  完成 {dt:.1f}s")


STEPS = {
    "infer": [
        ("推理 valid", ["tools/onnx_predict.py", "--split", "valid", "--json", "out/pred_valid.json"]),
        ("推理 test", ["tools/onnx_predict.py", "--split", "test", "--json", "out/pred_test.json"]),
    ],
    "eval": [
        ("评估 valid", ["tools/evaluate.py", "--pred", "out/pred_valid.json", "--json", "out/metrics_valid.json"]),
        ("评估 test", ["tools/evaluate.py", "--pred", "out/pred_test.json", "--json", "out/metrics_test.json"]),
    ],
    "failures": [
        ("挑错例", ["tools/build_failures.py", "--top", "9"]),
    ],
    "painting": [
        ("推理 demo", ["tools/onnx_predict.py", "--images", "demo/images", "--json", "out/pred_demo.json"]),
        ("推理 demo2", ["tools/onnx_predict.py", "--images", "demo2/images", "--json", "out/pred_demo2.json"]),
        ("推理 fails", ["tools/onnx_predict.py", "--images", "fails/images", "--json", "out/pred_fails.json"]),
        ("渲染 demo", ["tools/render_views.py", "--set", "demo"]),
        ("渲染 demo2", ["tools/render_views.py", "--set", "demo2"]),
        ("渲染 fails", ["tools/render_views.py", "--set", "fails"]),
    ],
    "web": [
        ("重建网页", ["tools/build_web.py"]),
    ],
    "ppt": [
        ("重建 PPT", ["tools/build_ppt.py"]),
    ],
    "check": [
        ("校验网页", ["tools/check_web.py"]),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in STEPS:
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    todo = list(STEPS) if a.all else [k for k in STEPS if getattr(a, k)]
    if not todo:
        ap.print_help()
        return 0

    print("=" * 60)
    print("  光伏电池片 EL 缺陷分级 · 重跑全部结果")
    print(f"  python: {PY}")
    print("=" * 60)
    t0 = time.time()
    for k in todo:
        for name, args in STEPS[k]:
            run(name, args)
    print(f"\n全部完成，用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
