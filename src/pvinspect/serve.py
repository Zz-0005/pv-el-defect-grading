# -*- coding: utf-8 -*-
"""
本地演示服务：批量上传 EL 图像 → 分级 + 处置建议 + 缺陷热力图 + 汇总报表。

只用标准库 http.server，不引入 Flask/FastAPI：
现场最怕"依赖没装好"，少一个依赖就少一个失败点。

接口：
  GET  /             演示页面
  GET  /health       健康检查（模型、设备、类别）
  GET  /samples      样例图列表
  GET  /sample/<n>   取某张样例图（只允许 demo/样本 目录下的文件）
  POST /predict      multipart 上传单个文件 → JSON（含耗时）
                     —— 批量检测由前端逐张调用，服务端保持无状态、无队列，
                        这样单张失败不会拖垮整批。
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from .config import GRADE_ACTION, GRADE_CN, WORK

_LOCK = threading.Lock()
_STATE: dict = {}

SAMPLES_DIR = WORK.parent / "demo" / "样本"

# 三档的语义色：合格=绿，待复核=琥珀，判废=红
GRADE_TONE = {
    "P0_合格": "ok", "G0_良好": "ok", "normal": "ok",
    "P1_待复核": "warn", "G1_轻微": "warn", "G2_中度": "warn",
    "P2_判废": "bad", "G3_严重": "bad", "defect": "bad",
}

PAGE = """<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>光伏电池片 EL 缺陷分级 · 批量检测</title>
<style>
  :root{
    --ink:#152430; --ink2:#3d5261; --muted:#7c8c99;
    --line:#dfe6ec; --bg:#f2f6f8; --card:#fff;
    --accent:#0e6f73; --accent-soft:#e6f2f2;
    --ok:#1a7f37; --ok-bg:#e8f5ec;
    --warn:#a86400; --warn-bg:#fdf3e2;
    --bad:#c0392b; --bad-bg:#fdeceb;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
       -webkit-font-smoothing:antialiased}
  .wrap{max-width:1240px;margin:0 auto;padding:24px 20px 48px}

  header{display:flex;align-items:flex-end;justify-content:space-between;
         gap:16px;flex-wrap:wrap;margin-bottom:16px}
  h1{font-size:21px;margin:0;font-weight:700}
  .sub{color:var(--muted);font-size:12.5px;margin-top:5px}
  .chips{display:flex;gap:8px;flex-wrap:wrap}
  .chip{background:#fff;border:1px solid var(--line);border-radius:999px;
        padding:5px 11px;font-size:11.5px;color:var(--ink2);
        display:flex;align-items:center;gap:6px}
  .dot{width:6px;height:6px;border-radius:50%;background:var(--ok)}
  .chip b{font-weight:600;color:var(--ink)}

  .grid{display:grid;grid-template-columns:minmax(300px,.85fr) minmax(520px,1.6fr);
        gap:16px;align-items:start}
  @media (max-width:980px){.grid{grid-template-columns:1fr}}

  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
        padding:16px;box-shadow:0 1px 2px rgba(21,36,48,.04)}

  .drop{border:1.5px dashed #c3d0d9;border-radius:11px;padding:22px 16px;
        text-align:center;color:var(--muted);cursor:pointer;transition:all .15s}
  .drop:hover{border-color:#9fb4c0;color:var(--ink2)}
  .drop.on{border-color:var(--accent);background:var(--accent-soft);color:var(--accent)}
  .drop .big{font-size:13.5px;font-weight:600;color:var(--ink2);margin-bottom:5px}
  .drop .hint{font-size:11px;line-height:1.65}
  kbd{background:#eef3f6;border:1px solid var(--line);border-radius:4px;padding:1px 5px;
      font-size:10.5px;font-family:ui-monospace,Consolas,monospace}

  .samples{margin-top:12px}
  .t{font-size:11px;color:var(--muted);margin-bottom:7px}
  .sbtn{background:#fff;border:1px solid var(--line);border-radius:7px;padding:5px 9px;
        font-size:11.5px;color:var(--ink2);cursor:pointer;margin:0 5px 5px 0;transition:all .12s}
  .sbtn:hover{border-color:var(--accent);color:var(--accent)}

  .queue{margin-top:12px;max-height:230px;overflow-y:auto;display:grid;
         grid-template-columns:repeat(auto-fill,minmax(74px,1fr));gap:7px}
  .qi{position:relative;aspect-ratio:1/1;background:#eef3f6;border-radius:7px;
      overflow:hidden;border:1px solid var(--line)}
  .qi img{width:100%;height:100%;object-fit:cover;display:block;opacity:.86}
  .qi .st{position:absolute;left:3px;bottom:3px;font-size:9px;padding:1px 5px;
          border-radius:4px;background:rgba(15,28,36,.78);color:#fff}
  .qi.ok{border-color:#b6dfc4} .qi.warn{border-color:#efd3a3} .qi.bad{border-color:#efbdb7}
  .qi .x{position:absolute;right:2px;top:2px;width:15px;height:15px;border-radius:50%;
         background:rgba(15,28,36,.7);color:#fff;font-size:10px;line-height:15px;
         text-align:center;cursor:pointer;opacity:0}
  .qi:hover .x{opacity:1}

  .row2{display:flex;gap:8px;margin-top:12px}
  .go{flex:1;background:var(--accent);color:#fff;border:0;border-radius:9px;
      padding:11px;font-size:13.5px;font-weight:600;cursor:pointer;transition:background .15s}
  .go:hover:not(:disabled){background:#0b5a5d}
  .go:disabled{background:#c3d0d9;cursor:not-allowed}
  .ghost{background:#fff;border:1px solid var(--line);color:var(--ink2);
         border-radius:9px;padding:11px 14px;font-size:13px;cursor:pointer}
  .ghost:hover{border-color:var(--accent);color:var(--accent)}

  .prog{height:4px;background:#eef3f6;border-radius:3px;overflow:hidden;margin-top:10px}
  .prog i{display:block;height:100%;width:0;background:var(--accent);
          transition:width .2s linear}
  .stat{font-size:11px;color:var(--muted);margin-top:6px;min-height:15px}

  .sum{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-bottom:12px}
  .sbox{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:#f8fafb}
  .sbox .k{font-size:10.5px;color:var(--muted);margin-bottom:3px}
  .sbox .v{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
  .sbox.ok{background:var(--ok-bg);border-color:#cfe8d8} .sbox.ok .v{color:var(--ok)}
  .sbox.warn{background:var(--warn-bg);border-color:#f2dcb8} .sbox.warn .v{color:var(--warn)}
  .sbox.bad{background:var(--bad-bg);border-color:#f4cfcb} .sbox.bad .v{color:var(--bad)}

  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th{text-align:left;font-weight:600;color:var(--muted);font-size:11px;
     padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
  td{padding:7px 8px;border-bottom:1px solid #eef3f6;vertical-align:middle}
  tr.row{cursor:pointer} tr.row:hover{background:#f7fafb}
  tr.row.sel{background:var(--accent-soft)}
  .tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600}
  .tag.ok{background:var(--ok-bg);color:var(--ok)}
  .tag.warn{background:var(--warn-bg);color:var(--warn)}
  .tag.bad{background:var(--bad-bg);color:var(--bad)}
  .num{font-family:ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums}
  .empty{color:var(--muted);font-size:12px;padding:26px 0;text-align:center}

  .detail{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
  .dimg{position:relative;background:#0f1c24;border-radius:9px;overflow:hidden;aspect-ratio:1/1}
  .dimg img{width:100%;height:100%;object-fit:contain;display:block}
  .dimg span{position:absolute;left:8px;top:8px;background:rgba(10,20,26,.72);color:#dbe7ec;
             font-size:10.5px;padding:3px 8px;border-radius:999px}
  .dimg.empty{display:flex;align-items:center;justify-content:center;color:#4b616e;font-size:11.5px}
  .err{margin-top:10px;color:var(--bad);font-size:12px;min-height:16px}
</style></head>
<body><div class="wrap">

<header>
  <div>
    <h1>光伏电池片 EL 缺陷分级</h1>
    <div class="sub">批量检测 · 三档判定（合格 / 待复核 / 判废）· 附 Grad-CAM 热力图</div>
  </div>
  <div class="chips">
    <div class="chip"><span class="dot" id="dot"></span><b id="dev">加载中</b></div>
    <div class="chip">模型 <b id="mdl">—</b></div>
    <div class="chip">已载入 <b id="cnt">0</b> 张</div>
  </div>
</header>

<div class="grid">
  <section class="card">
    <div class="drop" id="drop">
      <div class="big">拖入一批 EL 图像，或点击选择</div>
      <div class="hint">支持多选 · PNG / JPG · 也可 <kbd>Ctrl</kbd>+<kbd>V</kbd> 粘贴</div>
      <input type="file" id="file" accept="image/*" multiple hidden>
    </div>

    <div class="samples">
      <div class="t">或一键载入样例</div>
      <button class="sbtn" id="loadAll">载入全部样例</button>
      <div id="slist" style="display:inline"></div>
    </div>

    <div class="queue" id="queue"></div>

    <div class="row2">
      <button class="go" id="go" disabled>开始检测</button>
      <button class="ghost" id="clear">清空</button>
      <button class="ghost" id="csv" disabled>导出 CSV</button>
    </div>
    <div class="prog"><i id="pbar"></i></div>
    <div class="stat" id="stat"></div>
    <div class="err" id="err"></div>
  </section>

  <section class="card">
    <div class="sum">
      <div class="sbox"><div class="k">已检测</div><div class="v num" id="sTotal">0</div></div>
      <div class="sbox ok"><div class="k">合格</div><div class="v num" id="sOk">0</div></div>
      <div class="sbox warn"><div class="k">待复核</div><div class="v num" id="sWarn">0</div></div>
      <div class="sbox bad"><div class="k">判废</div><div class="v num" id="sBad">0</div></div>
    </div>

    <table>
      <thead><tr>
        <th style="width:34px">#</th><th>文件</th><th style="width:86px">判定</th>
        <th style="width:72px">置信度</th><th style="width:66px">耗时</th>
      </tr></thead>
      <tbody id="rows"><tr><td colspan="5" class="empty">还没有结果。载入图像后点“开始检测”。</td></tr></tbody>
    </table>

    <div class="detail">
      <div class="dimg empty" id="dOrig">原图</div>
      <div class="dimg empty" id="dCam">热力图</div>
    </div>
  </section>
</div>

<script>
const $ = id => document.getElementById(id);
const drop=$('drop'), file=$('file'), go=$('go'), err=$('err');
let items = [];          // {id, name, blob, url, res}
let seq = 0;

// ---------- 元信息
fetch('/health').then(r=>r.json()).then(d=>{
  const dev = d.device || 'cpu';
  $('dev').textContent = dev.startsWith('cuda') ? 'GPU 推理' : 'CPU 推理';
  $('mdl').textContent = (d.model||'').replace(/\\.pt$/,'') || '—';
  $('dot').style.background = dev.startsWith('cuda') ? 'var(--ok)' : 'var(--warn)';
}).catch(()=>{ $('dev').textContent='离线'; });

// ---------- 样例
let sampleNames = [];
fetch('/samples').then(r=>r.json()).then(d=>{
  sampleNames = d.samples || [];
  const box = $('slist');
  if(!sampleNames.length){ box.innerHTML='<span class="t">无样例</span>'; return; }
  sampleNames.forEach((n,i)=>{
    const b=document.createElement('button');
    b.className='sbtn'; b.textContent='样本'+String(i+1).padStart(2,'0');
    b.onclick=()=>fetch('/sample/'+encodeURIComponent(n)).then(r=>r.blob())
                   .then(bl=>addFiles([new File([bl], n, {type:'image/png'})]));
    box.appendChild(b);
  });
}).catch(()=>{});

$('loadAll').onclick = async ()=>{
  for(const n of sampleNames){
    const bl = await fetch('/sample/'+encodeURIComponent(n)).then(r=>r.blob());
    addFiles([new File([bl], n, {type:'image/png'})]);
  }
};

// ---------- 入队
function addFiles(files){
  for(const f of files){
    if(!f.type.startsWith('image/')) continue;
    items.push({id:'i'+(seq++), name:f.name, blob:f, url:URL.createObjectURL(f), res:null});
  }
  refresh();
}

function refresh(){
  $('cnt').textContent = items.length;
  const q = $('queue');
  q.innerHTML = items.map(it=>{
    const tone = it.res ? (it.res.tone||'') : '';
    const st = it.res ? (it.res.label_cn||'') : '待检测';
    return `<div class="qi ${tone}" data-id="${it.id}">
      <img src="${it.url}" alt="">
      <div class="st">${st}</div><div class="x" data-del="${it.id}">×</div></div>`;
  }).join('');
  q.querySelectorAll('[data-del]').forEach(el=>{
    el.onclick = ev=>{ ev.stopPropagation();
      const id = el.getAttribute('data-del');
      items = items.filter(x=>x.id!==id); refresh(); };
  });
  go.disabled = items.length===0;
}

// ---------- 拖拽 / 选择 / 粘贴
drop.onclick = ()=>file.click();
drop.ondragover = e=>{e.preventDefault();drop.classList.add('on')};
drop.ondragleave = ()=>drop.classList.remove('on');
drop.ondrop = e=>{e.preventDefault();drop.classList.remove('on');
  if(e.dataTransfer.files.length) addFiles(e.dataTransfer.files)};
file.onchange = e=>{ if(e.target.files.length){ addFiles(e.target.files); file.value=''; } };
document.addEventListener('paste', e=>{
  const fs=[...(e.clipboardData?.items||[])].filter(x=>x.type.startsWith('image/'))
             .map(x=>x.getAsFile()).filter(Boolean);
  if(fs.length) addFiles(fs);
});

$('clear').onclick = ()=>{ items=[]; renderSummary(); refresh(); $('rows').innerHTML=
  '<tr><td colspan="5" class="empty">还没有结果。载入图像后点“开始检测”。</td></tr>';
  $('dOrig').className='dimg empty'; $('dOrig').textContent='原图';
  $('dCam').className='dimg empty'; $('dCam').textContent='热力图';
  $('csv').disabled=true; $('pbar').style.width='0'; $('stat').textContent=''; };

// ---------- 批量检测
go.onclick = async ()=>{
  const todo = items.filter(x=>!x.res);
  if(!todo.length){ $('stat').textContent='全部已检测，点“清空”可重来。'; return; }
  go.disabled=true; $('csv').disabled=true; err.textContent='';
  const t0=performance.now();
  let done=0, fail=0;
  for(const it of todo){
    go.textContent = `检测中 ${done+1}/${todo.length}`;
    try{
      const fd=new FormData(); fd.append('file', it.blob, it.name);
      const r=await fetch('/predict',{method:'POST',body:fd});
      const d=await r.json();
      if(d.error){ fail++; it.res={error:d.error}; }
      else it.res=d;
    }catch(e){ fail++; it.res={error:String(e)}; }
    done++;
    $('pbar').style.width = (done/todo.length*100).toFixed(1)+'%';
    $('stat').textContent = `已完成 ${done}/${todo.length}` + (fail?`，失败 ${fail}`:'');
    renderSummary(); renderRows();
    refresh();
  }
  const dt=(performance.now()-t0)/1000;
  $('stat').textContent = `完成 ${done} 张，用时 ${dt.toFixed(1)} s，平均 ${(dt/done*1000).toFixed(0)} ms/张`
    + (fail?`，失败 ${fail}`:'');
  go.disabled=false; go.textContent='开始检测'; $('csv').disabled = !items.some(x=>x.res && !x.res.error);
};

function renderSummary(){
  const ok = items.filter(x=>x.res && x.res.tone==='ok').length;
  const warn = items.filter(x=>x.res && x.res.tone==='warn').length;
  const bad = items.filter(x=>x.res && x.res.tone==='bad').length;
  $('sTotal').textContent = ok+warn+bad;
  $('sOk').textContent = ok; $('sWarn').textContent = warn; $('sBad').textContent = bad;
}

function renderRows(){
  const rows = items.filter(x=>x.res);
  if(!rows.length){ $('rows').innerHTML =
    '<tr><td colspan="5" class="empty">还没有结果。</td></tr>'; return; }
  $('rows').innerHTML = rows.map((it,i)=>{
    const d=it.res;
    if(d.error) return `<tr><td>${i+1}</td><td>${it.name}</td>
      <td colspan="3" style="color:var(--bad)">${d.error}</td></tr>`;
    return `<tr class="row" data-id="${it.id}">
      <td class="num">${i+1}</td>
      <td title="${it.name}">${it.name.length>26?it.name.slice(0,24)+'…':it.name}</td>
      <td><span class="tag ${d.tone}">${d.label_cn}</span></td>
      <td class="num">${(d.confidence*100).toFixed(1)}%</td>
      <td class="num">${d.elapsed_ms||0} ms</td></tr>`;
  }).join('');
  $('rows').querySelectorAll('tr.row').forEach(tr=>{
    tr.onclick = ()=>{
      $('rows').querySelectorAll('tr.row').forEach(x=>x.classList.remove('sel'));
      tr.classList.add('sel'); showDetail(tr.getAttribute('data-id'));
    };
  });
}

function showDetail(id){
  const it = items.find(x=>x.id===id); if(!it||!it.res) return;
  const o=$('dOrig'); o.className='dimg'; o.innerHTML='<span>原图</span><img src="'+it.url+'">';
  const c=$('dCam');
  if(it.res.cam_png){ c.className='dimg';
    c.innerHTML='<span>Grad-CAM</span><img src="data:image/png;base64,'+it.res.cam_png+'">'; }
  else { c.className='dimg empty'; c.textContent = it.res.cam_error?'热力图失败':'无热力图'; }
}

// ---------- 导出 CSV
$('csv').onclick = ()=>{
  const rows = items.filter(x=>x.res && !x.res.error);
  const head = ['文件名','判定','处置动作','置信度','耗时ms','合格概率','待复核概率','判废概率'];
  const lines = [head.join(',')];
  for(const it of rows){
    const d=it.res;
    const p = n => { const hit=(d.probs||[]).find(x=>x.name===n); return hit?(hit.p*100).toFixed(1):''; };
    lines.push([it.name, d.label_cn, d.action, (d.confidence*100).toFixed(1),
                d.elapsed_ms||0, p('合格'), p('待复核'), p('判废')]
               .map(v=>`"${String(v).replace(/"/g,'""')}"`).join(','));
  }
  const blob = new Blob(['\\ufeff'+lines.join('\\r\\n')], {type:'text/csv;charset=utf-8'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='EL缺陷分级_结果_'+new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')+'.csv';
  a.click();
};

refresh();
</script>
</div></body></html>
"""


def _tone(label: str) -> str:
    return GRADE_TONE.get(label, "idle")


def _load_model(weights: Path, device: str = ""):
    from ultralytics import YOLO
    return YOLO(str(weights))


def _actual_device(explicit: str) -> str:
    if explicit:
        return explicit
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda · " + torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def _predict(state: dict, img_bytes: bytes) -> dict:
    from PIL import Image
    from .cam import GradCAM, overlay, preprocess

    t_start = time.perf_counter()
    im = Image.open(io.BytesIO(img_bytes)).convert("L")
    gray = np.asarray(im, dtype=np.uint8)

    model = state["model"]
    with _LOCK:
        r = model.predict(gray, imgsz=224, device=state["device"], verbose=False)[0]
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
                x = preprocess(gray, 224).to(state["device"])
                cam, _, _ = gcam(x, idx)
            rgb = overlay(gray, cam)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG")
            cam_png = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        cam_png = ""
        state["cam_error"] = f"{type(e).__name__}: {e}"

    return {
        "label": label,
        "label_cn": GRADE_CN.get(label, label),
        "tone": _tone(label),
        "action": GRADE_ACTION.get(label, ""),
        "confidence": float(probs[idx]),
        "probs": [{"name": GRADE_CN.get(names[i], names[i]), "p": float(probs[i]),
                   "tone": _tone(names[i])}
                  for i in np.argsort(-probs)],
        "cam_png": cam_png,
        "cam_error": state.get("cam_error", ""),
        "elapsed_ms": int((time.perf_counter() - t_start) * 1000),
    }


def _list_samples() -> list[str]:
    if not SAMPLES_DIR.exists():
        return []
    return [p.name for p in sorted(SAMPLES_DIR.glob("*.png"))]


def make_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/health":
                self._json({
                    "ok": True,
                    "weights": str(state.get("weights", "")),
                    "model": Path(state.get("weights", "")).name,
                    "device": state.get("device_label") or "cpu",
                    "classes": state.get("classes", []),
                })
            elif path == "/samples":
                self._json({"samples": _list_samples()})
            elif path.startswith("/sample/"):
                name = unquote(path[len("/sample/"):])
                if name in _list_samples():
                    self._send(200, (SAMPLES_DIR / name).read_bytes(), "image/png")
                else:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                if urlparse(self.path).path != "/predict":
                    self._send(404, b"not found", "text/plain")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                img = _extract_file(raw, self.headers.get("Content-Type", ""))
                if img is None:
                    self._json({"error": "未收到文件"}, 400)
                    return
                self._json(_predict(state, img))
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def _extract_file(raw: bytes, ctype: str) -> bytes | None:
    """从 multipart/form-data 里取出第一个文件的字节。"""
    if "boundary=" not in ctype:
        return None
    boundary = ctype.split("boundary=")[-1].strip().strip('"').encode()
    if boundary not in raw:
        return None
    for part in raw.split(b"--" + boundary):
        if b"filename=" not in part:
            continue
        head, sep, body = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        return body.rsplit(b"\r\n", 1)[0]
    return None


def _port_in_use(host: str, port: int) -> bool:
    """
    端口是否真的有人在服务。

    Windows 上 SO_REUSEADDR 会让两个进程都能 bind 同一个端口，
    于是 bind 不报错、但连接全被先启动的那个进程接走（表现为"服务启动了却打不开"）。
    所以不能靠 bind 是否抛异常判断，必须实际连一下。
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.35)
        return s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0


def _bind(host: str, port: int, state: dict, span: int = 20) -> ThreadingHTTPServer:
    """
    绑定端口：从指定端口往上找第一个真正空闲的。

    不用"直接给 0 让系统分配"：那样拿到的是 49xxx 的临时端口，
    URL 每次都不一样，现场要重新找地址。
    """
    handler = make_handler(state)
    if port <= 0:
        return ThreadingHTTPServer((host, 0), handler)
    for p in range(port, port + span):
        if _port_in_use(host, p):
            continue
        try:
            srv = ThreadingHTTPServer((host, p), handler)
        except OSError:
            continue
        if p != port:
            print(f"[提示] {port} 被占用，改用 {p}")
        return srv
    print(f"[警告] {port}-{port + span - 1} 都被占用，改用系统分配端口")
    return ThreadingHTTPServer((host, 0), handler)


def run(weights: Path | None = None, host: str = "127.0.0.1", port: int = 8000,
        device: str = "", open_browser: bool = False) -> None:
    from .cam import GradCAM

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
    import torch as _torch
    dev = device or ("cuda" if _torch.cuda.is_available() else "cpu")
    model.model.to(dev)
    state = {
        "model": model,
        "weights": str(weights),
        "device": dev,
        "device_label": _actual_device(device),
        "classes": list(model.names.values()) if isinstance(model.names, dict) else list(model.names),
    }
    try:
        state["gcam"] = GradCAM(model.model)
    except Exception as e:
        state["gcam"] = None
        state["cam_error"] = f"GradCAM 初始化失败：{e}"

    srv = _bind(host, port, state)
    real_port = srv.server_address[1]        # port=0 时由系统分配，回读真实端口
    url = f"http://{host}:{real_port}"
    print(f"演示地址： {url}   模型：{weights.name}   设备：{state['device_label']}")
    print(f"样例目录： {SAMPLES_DIR}（{len(_list_samples())} 张）")
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if state.get("gcam"):
            state["gcam"].close()
