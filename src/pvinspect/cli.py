# -*- coding: utf-8 -*-
"""命令行入口：python -m pvinspect <command>。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import GRADES, GRADES3, WORK


def cmd_prepare(args) -> int:
    from . import prepare
    prepare.run()
    return 0


def cmd_train(args) -> int:
    from . import train
    from .config import TrainCfg
    cfg = TrainCfg(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                   device=args.device or "", workers=0)
    train.run(cfg, only=args.only.split(",") if args.only else None)
    return 0


def cmd_eval(args) -> int:
    from . import evaluate
    from .config import GRADE_CN

    models = WORK / "models"
    wanted = {
        "E1_grouped": ("分组切分 · 四级分级（本方案）", "ds4", GRADES),
        "E1_random": ("随机切分 · 四级分级（泄漏对照）", "ds4_rand", GRADES),
        "E2_binary": ("分组切分 · 二分类基线", "ds2", ["normal", "defect"]),
        "E3_three": ("分组切分 · 三档操作级", "ds3", GRADES3),
    }
    results: dict[str, dict] = {}
    only = set(args.only.split(",")) if getattr(args, "only", "") else None
    for name, (desc, ds, classes) in wanted.items():
        if only and name not in only:
            continue
        w = models / f"{name}.pt"
        if not w.exists():
            print(f"跳过 {name}（缺权重）")
            continue
        root = WORK / ds
        if not (root / "test").exists():
            print(f"跳过 {name}（缺测试集 {root / 'test'}）")
            continue
        print(f"评测 {name} …", flush=True)
        dirs = {"train": root / "train", "val": root / "val", "test": root / "test"}
        out = evaluate.evaluate_adjusted(w, dirs, imgsz=args.imgsz, device=args.device or None)

        raw = out["raw"]
        raw["desc"] = f"{desc}（argmax）"
        raw["tau"] = 0.0
        results[name] = raw

        adj = out["adjusted"]
        adj["desc"] = f"{desc}（先验调整 τ={out['tau']:.2f}）"
        adj["tau"] = out["tau"]
        adj["tau_scores"] = out["tau_scores"]
        adj["desc_short"] = desc
        results[f"{name}+adj"] = adj

    if not results:
        print("没有可评测的模型")
        return 1

    out_dir = WORK / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate.report(results, out_dir / "results.csv", out_dir / "results.json")

    for name, r in results.items():
        names = r["classes"]
        print(f"\n{name}  {r['desc']}")
        print(f"  样本 {r['n']}  准确率 {r['accuracy']:.4f}  宏平均F1 {r['macro_f1']:.4f}"
              + (f"  严重漏判率 {r['severe_miss_rate']:.4f}" if "severe_miss_rate" in r else "")
              + f"  代价/千张 {r['cost_per_1000']:.1f}")
        for c in names:
            m = r["per_class"][c]
            print(f"    {GRADE_CN.get(c, c):6s} recall={m['recall']:.3f} "
                  f"prec={m['precision']:.3f} f1={m['f1']:.3f} n={m['support']}")
        evaluate.plot_confusion(
            __import__("numpy").array(r["confusion"]), names,
            out_dir / f"confusion_{name.replace('+','_')}.png", f"{name} 混淆矩阵")
    evaluate.plot_bars({k: v for k, v in results.items() if "+" not in k},
                       out_dir / "accuracy_compare.png")

    # 错例图：只对主方案做，避免耗时
    main_w = models / "E1_grouped.pt"
    if main_w.exists():
        try:
            info = evaluate.save_error_gallery(
                main_w, WORK / "ds4" / "test", GRADES,
                out_dir / "errors_E1_grouped.png", imgsz=args.imgsz,
                device=args.device or None)
            print(f"\n错例图：{info}")
        except Exception as e:
            print(f"错例图生成失败（不影响指标）：{type(e).__name__}: {e}")

    print(f"\n结果已写入 {out_dir}")
    return 0


def cmd_serve(args) -> int:
    from . import serve
    w = Path(args.weights) if args.weights else None
    serve.run(w, host=args.host, port=args.port, device=args.device or "",
              open_browser=args.open)
    return 0


def cmd_selftest(args) -> int:
    """不依赖训练权重的自检：数据、分组、泄漏、算子。"""
    from . import selftest
    return selftest.main()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pvinspect", description="光伏电池片 EL 缺陷分级")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="构建分级数据集与分组切分")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train", help="训练对照实验模型")
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--imgsz", type=int, default=224)
    s.add_argument("--device", default="")
    s.add_argument("--only", default="", help="逗号分隔的实验名")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eval", help="在冻结测试集上评测并出图")
    s.add_argument("--imgsz", type=int, default=224)
    s.add_argument("--device", default="")
    s.add_argument("--only", default="", help="逗号分隔的实验名")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("serve", help="启动本地演示服务（支持批量检测）")
    s.add_argument("--weights", default="")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000,
                   help="0 = 让系统分配空闲端口（端口被占用时用这个）")
    s.add_argument("--device", default="")
    s.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("selftest", help="自检（无需权重）")
    s.set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
