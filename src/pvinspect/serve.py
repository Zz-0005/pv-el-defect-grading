# -*- coding: utf-8 -*-
"""
本地演示服务：上传一张 EL 图 → 输出分级 + 处置建议 + 缺陷热力图。

只用标准库 http.server，不引入 Flask/FastAPI：
答辩现场最怕"依赖没装好"，少一个依赖就少一个失败点。

接口：
  GET  /            演示页面
  POST /predict     multipart 上传字段 file → JSON
  GET  /health      健康检查（含模型路径与设备）
"""
from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np

from .config import GRADE_ACTION, GRADE_CN, GRADES, WORK

_LOCK = threading.Lock()
_STATE: dict = {}

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>光伏电池片 EL 缺陷分级</title>
<style>
 body{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#f4f7f9;margin:0;color:#152430}
 .wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
 h1{font-size:22px;margin:0 0 6px}
 .sub{color:#7c8c99;font-size:13px;margin-bottom:22px}
 .row{display:flex;gap:22px;flex-wrap:wrap}
 .card{background:#fff;border:1px solid #d9e1e7;border-radius:10px;padding:18px;flex:1;min-width:320px}
 .drop{border:2px dashed #b9c7cf;border-radius:8px;padding:26px;text-align:center;color:#7c8c99;cursor:pointer}
 .drop.on{border-color:#0e6f73;color:#0e6f73;background:#f0f8f8}
 img.prev{max-width:100%;border-radius:6px;margin-top:12px}
 button{background:#0e6f73;color:#fff;border:0;border-radius:6px;padding:10px 20px;font-size:14px;cursor:pointer}
 button:disabled{background:#b9c7cf}
 .grade{font-size:30px;font-weight:700;margin:8px 0}
 .action{display:inline-block;padding:5px 12px;border-radius:20px;font-size:13px;background:#e3f0f0;color:#0e6f73}
 .bar{height:8px;background:#eaf0f4;border-radius:4px;overflow:hidden;margin:5px 0 12px}
 .bar i{display:block;height:100%;background:#0e6f73}
 .lbl{display:flex;justify-content:space-between;font-size:12.5px;color:#3d5261}
 .meta{font-size:12px;color:#7c8c99;margin-top:14px;border-top:1px solid #eaf0f4;padding-top:10px}
 .err{color:#c0392b;font-size:13px}
</style></head>
<body><div class="wrap">
<h1>光伏电池片 EL 图像缺陷分级</h1>
<div class="sub">YOLOv8n-cls · 四级分级（良好 / 轻微 / 中度 / 严重）· 附 Grad-CAM 缺陷定位</div>
<div class="row">
  <div class="card">
    <div class="drop" id="drop">点击选择或拖入 EL 图像（PNG / JPG）
      <input type="file" id="file" accept="image/*" hidden>
    </div>
    <img class="prev" id="prev" hidden>
    <div style="margin-top:14px"><button id="go" disabled>开始检测</button></div>
    <div class="meta" id="meta">模型未加载</div>
  </div>
  <div class="card">
    <div class="lbl"><span>分级结果</span><span id="conf"></span></div>
    <div class="grade" id="grade">—</div>
    <div class="action" id="action">等待检测</div>
    <div style="margin-top:18px" id="probs"></div>
    <img class="prev" id="cam" hidden>
    <div class="err" id="err"></div>
  </div>
</div>
</div>
<script>
const drop=document.getElementById('drop'),file=document.getElementById('file'),
      prev=document.getElementById('prev'),go=document.getElementById('go'),
      cam=document.getElementById('cam'),grade=document.getElementById('grade'),
      action=document.getElementById('action'),probs=document.getElementById('probs'),
      conf=document.getElementById('conf'),err=document.getElementById('err'),
      meta=document.getElementById('meta');
let blob=null;
fetch('/health').then(r=>r.json()).then(d=>{meta.textContent='模型 '+(d.weights||'')+' · 设备 '+d.device;});
drop.onclick=()=>file.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('on')};
drop.ondragleave=()=>drop.classList.remove('on');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('on');if(e.dataTransfer.files[0])load(e.dataTransfer.files[0])};
file.onchange=e=>{if(e.target.files[0])load(e.target.files[0])};
function load(f){blob=f;prev.src=URL.createObjectURL(f);prev.hidden=false;go.disabled=false;
  grade.textContent='—';action.textContent='等待检测';probs.innerHTML='';cam.hidden=true;err.textContent='';conf.textContent='';}
go.onclick=async()=>{
  if(!blob)return; go.disabled=true; go.textContent='检测中…';
  const fd=new FormData(); fd.append('file',blob);
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){err.textContent=d.error;return}
    grade.textContent=d.label_cn;
    conf.textContent=(d.confidence*100).toFixed(1)+'%';
    action.textContent=d.action;
    probs.innerHTML=d.probs.map(p=>`<div class="lbl"><span>${p.name}</span><span>${(p.p*100).toFixed(1)}%</span></div>
      <div class="bar"><i style="width:${(p.p*100).toFixed(1)}%"></i></div>`).join('');
    cam.src='data:image/png;base64,'+d.cam_png; cam.hidden=false;
  }catch(e){err.textContent='请求失败：'+e}
  finally{go.disabled=false;go.textContent='开始检测'}
};
</script></body></html>
"""


def _load_model(weights: Path, device: str = ""):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    return m


def _predict(state: dict, img_bytes: bytes) -> dict:
    from PIL import Image
    from .cam import GradCAM, overlay, preprocess

    im = Image.open(io.BytesIO(img_bytes)).convert("L")
    gray = np.asarray(im, dtype=np.uint8)

    model = state["model"]
    with _LOCK:
        r = model.predict(gray, imgsz=224, device=state["device"] or None, verbose=False)[0]
    probs = r.probs.data.cpu().numpy()
    names = [model.names[i] for i in range(len(model.names))] if isinstance(model.names, dict) \
        else list(model.names)
    idx = int(probs.argmax())
    label = names[idx]

    cam_png = ""
    try:
        gcam = state.get("gcam")
        if gcam is not None:
            with _LOCK:
                x = preprocess(gray, 224)
                cam, _, _ = gcam(x.to(state["device"]) if state["device"] else x, idx)
            rgb = overlay(gray, cam)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG")
            cam_png = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:      # CAM 失败不影响分级结论
        cam_png = ""
        state["cam_error"] = f"{type(e).__name__}: {e}"

    return {
        "label": label,
        "label_cn": GRADE_CN.get(label, label),
        "action": GRADE_ACTION.get(label, ""),
        "confidence": float(probs[idx]),
        "probs": [{"name": GRADE_CN.get(names[i], names[i]), "p": float(probs[i])}
                  for i in np.argsort(-probs)],
        "cam_png": cam_png,
        "cam_error": state.get("cam_error", ""),
    }


def make_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/health":
                body = json.dumps({
                    "ok": True,
                    "weights": str(state.get("weights", "")),
                    "device": state.get("device") or "cpu",
                    "classes": state.get("classes", []),
                }, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                if self.path != "/predict":
                    self._send(404, b"not found", "text/plain")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                img = _extract_file(raw, self.headers.get("Content-Type", ""))
                if img is None:
                    self._send(400, json.dumps({"error": "未收到文件"}).encode(), "application/json")
                    return
                res = _predict(state, img)
                self._send(200, json.dumps(res, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as e:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"},
                                           ensure_ascii=False).encode(), "application/json")

    return Handler


def _extract_file(raw: bytes, ctype: str) -> bytes | None:
    """从 multipart/form-data 里取出第一个文件的字节。"""
    if "boundary=" not in ctype:
        return None
    boundary = ctype.split("boundary=")[-1].strip().strip('"').encode()
    if boundary not in raw:
        return None
    parts = raw.split(b"--" + boundary)
    for part in parts:
        if b"filename=" not in part:
            continue
        head, _, body = part.partition(b"\r\n\r\n")
        if not _:
            continue
        return body.rsplit(b"\r\n", 1)[0]
    return None


def run(weights: Path | None = None, host: str = "127.0.0.1", port: int = 8000,
        device: str = "") -> None:
    from .cam import GradCAM

    # 默认优先用三档操作级模型：它的加权代价远低于四级模型
    # （实测 238.8 vs 2715.6 / 千张），更适合演示"判废/返修/流转"这条业务链路。
    if weights is None:
        for cand in ("E3_three.pt", "E1_grouped.pt"):
            p = WORK / "models" / cand
            if p.exists():
                weights = p
                break
    weights = weights or (WORK / "models" / "E1_grouped.pt")
    if not weights.exists():
        raise FileNotFoundError(f"找不到权重 {weights}，先运行 train")

    model = _load_model(weights, device)
    state = {
        "model": model,
        "weights": str(weights),
        "device": device,
        "classes": list(model.names.values()) if isinstance(model.names, dict) else list(model.names),
    }
    try:
        state["gcam"] = GradCAM(model.model)
    except Exception as e:
        state["gcam"] = None
        state["cam_error"] = f"GradCAM 初始化失败：{e}"

    srv = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"演示地址： http://{host}:{port}   模型：{weights.name}   设备：{device or 'auto'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if state.get("gcam"):
            state["gcam"].close()
