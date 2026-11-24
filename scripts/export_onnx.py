# -*- coding: utf-8 -*-
"""把 E3_three.pt 导出为 ONNX，并校验 onnxruntime 的输出与 torch 一致。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvinspect import compat  # noqa: E402

compat.configure_dirs()
compat.patch_ultralytics()

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from pvinspect.cam import preprocess  # noqa: E402

W = ROOT / "artifacts" / "models" / "E3_three.pt"
OUT = ROOT / "artifacts" / "models" / "pv_el_yolov8s_cls.onnx"

print("导出 ONNX ...", flush=True)
model = YOLO(str(W))
model.export(format="onnx", imgsz=224, opset=12, simplify=True, dynamic=False)
produced = Path(str(W).replace(".pt", ".onnx"))
if produced.exists() and produced != OUT:
    produced.replace(OUT)
print("ONNX:", OUT, f"{OUT.stat().st_size/1e6:.1f} MB")

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]
print("输入:", inp.name, inp.shape, inp.type)
print("输出:", out.name, out.shape, out.type)

# 对比一张图
img = sorted((ROOT / "artifacts" / "ds3" / "test" / "P2_判废").glob("*.png"))[0]
from PIL import Image  # noqa: E402

arr = np.asarray(Image.open(img).convert("L"), dtype=np.uint8)
x = preprocess(arr, 224).numpy()

sess_out = sess.run(None, {inp.name: x})[0][0]

net = model.model.cpu().eval()
with torch.no_grad():
    t = net(torch.from_numpy(x))
    if isinstance(t, (list, tuple)):
        t = t[0]
torch_out = t.numpy()[0]

print("\n图:", img.name)
print("  onnxruntime:", np.round(sess_out, 4))
print("  torch      :", np.round(torch_out, 4))
print("  最大偏差   :", float(np.abs(sess_out - torch_out).max()))
print("  和是否为1  :", float(sess_out.sum()))
