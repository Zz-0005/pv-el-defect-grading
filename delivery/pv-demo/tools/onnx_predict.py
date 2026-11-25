# -*- coding: utf-8 -*-
"""
用 ONNX Runtime 跑推理，输出逐图概率 JSON（以及可选的标注图）。

    python tools/onnx_predict.py --split test --json out/pred_test.json
    python tools/onnx_predict.py --images demo/images --json out/pred_demo.json

不需要 torch、不需要 CUDA。CPU 单张约 15~30 ms。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASSES, IMGSZ, MODEL_DIR, OUT, ROOT, dump_json, imread_gray, iter_split, preprocess,
)

MODEL = MODEL_DIR / "pv_el_yolov8s_cls.onnx"


def load_session(model_path: Path = MODEL):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def predict_probs(sess, gray: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    x = preprocess(gray, IMGSZ)
    return sess.run(None, {name: x})[0][0].astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="", help="train / valid / test")
    ap.add_argument("--images", default="", help="自定义图片目录")
    ap.add_argument("--json", default="out/pred.json")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    sess = load_session()
    items = []
    if a.split:
        items = [(p, lbl, a.split) for p, lbl, _ in iter_split(a.split)]
    elif a.images:
        d = Path(a.images)
        if not d.is_absolute():
            d = ROOT / d
        # 同名标签放在同级 labels/ 目录下；读不到就留空（原图/真值视图会显示"无标签"）
        lbl_dir = d.parent / "labels"
        items = []
        for p in sorted(d.glob("*.png")):
            lp = lbl_dir / f"{p.stem}.txt"
            true = lp.read_text(encoding="utf-8").strip().splitlines()[0].strip() if lp.exists() else ""
            items.append((p, true, a.images))
    else:
        print("需要 --split 或 --images")
        return 2

    if a.limit:
        items = items[: a.limit]

    print(f"推理 {len(items)} 张（ONNX Runtime, CPU）...", flush=True)
    recs = []
    t0 = time.time()
    for i, (p, true, split) in enumerate(items, 1):
        gray = imread_gray(p)
        t1 = time.perf_counter()
        probs = predict_probs(sess, gray)
        dt = (time.perf_counter() - t1) * 1000
        idx = int(probs.argmax())
        recs.append({
            "image": p.name,
            "split": split,
            "true": true,
            "pred": CLASSES[idx],
            "conf": float(probs[idx]),
            "probs": {c: float(v) for c, v in zip(CLASSES, probs)},
            "ms": round(dt, 2),
        })
        if i % 100 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    total = time.time() - t0
    out = Path(a.json)
    if not out.is_absolute():
        out = ROOT / out
    dump_json(out, recs)
    ms = np.array([r["ms"] for r in recs])
    print(f"完成 {len(recs)} 张，总耗时 {total:.1f}s，单张 "
          f"中位 {np.median(ms):.1f} ms / P95 {np.percentile(ms, 95):.1f} ms")
    print(f"结果 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
