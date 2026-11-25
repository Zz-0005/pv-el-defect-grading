# -*- coding: utf-8 -*-
"""
本地演示服务：把任意一张 EL 图片拖进去，返回三档判定与处置动作。

    python tools/serve_onnx.py            # 默认 http://127.0.0.1:8000
    python tools/serve_onnx.py --port 8010

只用标准库 + onnxruntime，**不需要 torch**。现场演示最怕环境没装好，
少一个依赖就少一个失败点。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASS_ACTION, CLASS_CN, CLASS_COLOR, CLASSES, ROOT, font,
)

LOCK = threading.Lock()
STATE: dict = {}

PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>光伏电池片 EL 缺陷分级 · 现场演示</title>
<style>
body{margin:0;font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#f4f7f9;color:#152430}
.wrap{max-width:1080px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:21px;margin:0 0 4px}
.sub{color:#7c8c99;font-size:13px;margin-bottom:20px}
.row{display:flex;gap:20px;flex-wrap:wrap}
.card{background:#fff;border:1px solid #d9e1e7;border-radius:10px;padding:18px;flex:1;min-width:320px}
.drop{border:2px dashed #b9c7cf;border-radius:8px;padding:28px;text-align:center;color:#7c8c99;cursor:pointer}
.drop.on{border-color:#0e6f73;color:#0e6f73;background:#f0f8f8}
img{max-width:100%;border-radius:6px;margin-top:12px}
button{background:#0e6f73;color:#fff;border:0;border-radius:6px;padding:10px 20px;font-size:14px;cursor:pointer}
button:disabled{background:#b9c7cf}
.grade{font-size:30px;font-weight:800;margin:8px 0}
.act{display:inline-block;padding:5px 12px;border-radius:20px;font-size:13px;background:#e3f0f0;color:#0e6f73}
.bar{height:8px;background:#eaf0f4;border-radius:4px;overflow:hidden;margin:4px 0 10px}
.bar i{display:block;height:100%}
.lbl{display:flex;justify-content:space-between;font-size:12.5px;color:#3d5261}
#band{font-weight:700}
.basis{font-size:13px;color:#3d5261;background:#f4f7f9;border-radius:6px;padding:9px 12px;margin-top:12px;line-height:1.6}
.warnbox{margin-top:10px;background:#fdf6ec;border-left:4px solid #c9762a;border-radius:5px;
 padding:9px 12px;font-size:13px;color:#6b4a25}
.meta{font-size:12px;color:#7c8c99;margin-top:14px;border-top:1px solid #eaf0f4;padding-top:10px}
</style></head><body><div class="wrap">
<h1>光伏电池片 EL 缺陷分级 · 现场演示</h1>
<div class="sub">YOLOv8s-cls（ONNX，CPU）· 三档：合格 / 待复核 / 判废 · 单张约 8 ms</div>
<div class="row">
 <div class="card">
  <div class="drop" id="drop">点击选择或拖入 EL 图片（png / jpg）<input type="file" id="f" accept="image/*" hidden></div>
  <img id="prev" hidden><div style="margin-top:14px"><button id="go" disabled>开始检测</button></div>
  <div class="meta" id="meta">模型加载中…</div>
 </div>
 <div class="card">
  <div class="lbl"><span>判定结果</span><span id="band"></span></div>
  <div class="grade" id="grade">—</div>
  <div class="act" id="act">等待检测</div>
  <div id="basis" class="basis"></div>
  <div id="warn" class="warnbox" hidden></div>
  <div style="margin-top:14px" id="probs"></div>
  <div class="meta" id="tip"></div>
 </div>
</div></div>
<script>
const d=document.getElementById('drop'),f=document.getElementById('f'),p=document.getElementById('prev'),
 g=document.getElementById('go'),gr=document.getElementById('grade'),ac=document.getElementById('act'),
 pr=document.getElementById('probs'),cf=document.getElementById('band'),me=document.getElementById('meta'),
 tp=document.getElementById('tip'),ba=document.getElementById('basis'),wn=document.getElementById('warn');
let blob=null;
fetch('/health').then(r=>r.json()).then(j=>{me.textContent='模型 '+j.model+' · 设备 '+j.device;});
d.onclick=()=>f.click();
d.ondragover=e=>{e.preventDefault();d.classList.add('on')};
d.ondragleave=()=>d.classList.remove('on');
d.ondrop=e=>{e.preventDefault();d.classList.remove('on');if(e.dataTransfer.files[0])load(e.dataTransfer.files[0])};
f.onchange=e=>{if(e.target.files[0])load(e.target.files[0])};
function load(x){blob=x;p.src=URL.createObjectURL(x);p.hidden=false;g.disabled=false;
 gr.textContent='—';ac.textContent='等待检测';pr.innerHTML='';cf.textContent='';tp.textContent='';
 ba.textContent='';ba.style.display='none';wn.hidden=true;}
g.onclick=async()=>{if(!blob)return;g.disabled=true;g.textContent='检测中…';
 const fd=new FormData();fd.append('file',blob);
 try{const r=await fetch('/predict',{method:'POST',body:fd});const j=await r.json();
  if(j.error){tp.textContent=j.error;return}
  gr.textContent=j.cn;gr.style.color=j.color;ac.textContent=j.action;
  const bc={高:'#1a7f37',中:'#c9762a',低:'#c0392b'}[j.band]||'#7c8c99';
  cf.innerHTML='把握 <b style="color:'+bc+'">'+j.band+'</b>';
  ba.style.display='block';
  ba.innerHTML='判定依据：<b>'+j.cn+' '+j.conf.toFixed(2)+'</b>，'
    +'次高 <b>'+j.second_cn+' '+j.second.toFixed(2)+'</b>，差距 <b>'+j.margin.toFixed(2)+'</b>';
  wn.hidden=!j.hint; if(j.hint) wn.textContent='⚠ '+j.hint;
  pr.innerHTML=j.probs.map(o=>`<div class="lbl"><span>${o.cn}</span><span>${(o.p*100).toFixed(1)}%</span></div>
   <div class="bar"><i style="width:${(o.p*100).toFixed(1)}%;background:${o.color}"></i></div>`).join('');
  tp.textContent='推理耗时 '+j.ms.toFixed(1)+' ms';
 }catch(e){tp.textContent='请求失败：'+e}
 finally{g.disabled=false;g.textContent='开始检测'}};
</script></body></html>"""


def _predict(img_bytes: bytes) -> dict:
    import time
    t0 = time.perf_counter()
    im = Image.open(io.BytesIO(img_bytes)).convert("L")
    gray = np.asarray(im, dtype=np.uint8)

    from common import preprocess
    x = preprocess(gray)
    sess = STATE["sess"]
    name = sess.get_inputs()[0].name
    with LOCK:
        probs = sess.run(None, {name: x})[0][0].astype(np.float32)
    ms = (time.perf_counter() - t0) * 1000

    idx = int(probs.argmax())
    cls = CLASSES[idx]
    order = np.argsort(-probs)
    p1 = float(probs[order[0]])
    p2 = float(probs[order[1]])
    margin = p1 - p2

    # 把握度用「top1 与 top2 的差距」判断，而不是看裸置信度。
    # 原因：良品没有明显缺陷特征，概率天然分散（中位 0.80），
    # 而判废有明确证据（中位 1.00）。拿同一把尺子量两者会误导。
    if margin >= 0.5 and p1 >= 0.6:
        band, hint = "高", ""
    elif margin >= 0.2:
        band, hint = "中", ""
    else:
        band, hint = "低", "系统把握不足，建议人工复核后再放行"

    return {
        "label": cls,
        "cn": CLASS_CN[cls],
        "action": CLASS_ACTION[cls],
        "color": "rgb(%d,%d,%d)" % CLASS_COLOR[cls],
        "conf": p1,
        "second_cn": CLASS_CN[CLASSES[order[1]]],
        "second": p2,
        "margin": margin,
        "band": band,
        "hint": hint,
        "ms": ms,
        "probs": [{"cn": CLASS_CN[CLASSES[i]], "p": float(probs[i]),
                   "color": "rgb(%d,%d,%d)" % CLASS_COLOR[CLASSES[i]]} for i in order],
    }


def _extract(raw: bytes, ctype: str):
    if "boundary=" not in ctype:
        return None
    b = ctype.split("boundary=")[-1].strip().strip('"').encode()
    for part in raw.split(b"--" + b):
        if b"filename=" not in part:
            continue
        head, sep, body = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        return body.rsplit(b"\r\n", 1)[0]
    return None


def make_handler():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/health":
                self._send(200, json.dumps({
                    "ok": True, "model": STATE["model"],
                    "device": "CPU (onnxruntime)", "classes": CLASSES,
                }, ensure_ascii=False).encode("utf-8"))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                if self.path != "/predict":
                    self._send(404, b"not found", "text/plain")
                    return
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                img = _extract(raw, self.headers.get("Content-Type", ""))
                if img is None:
                    self._send(400, json.dumps({"error": "未收到图片"}).encode())
                    return
                self._send(200, json.dumps(_predict(img), ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self._send(500, json.dumps(
                    {"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False).encode("utf-8"))

    return H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    import onnxruntime as ort
    mp = ROOT / "models" / "pv_el_yolov8s_cls.onnx"
    sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
    STATE["sess"] = sess
    STATE["model"] = mp.name

    srv = ThreadingHTTPServer((a.host, a.port), make_handler())
    print(f"演示地址： http://{a.host}:{a.port}")
    print(f"模型： {mp.name}   设备： CPU（onnxruntime）")
    print("关闭本窗口即停止服务")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
