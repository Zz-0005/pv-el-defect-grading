# -*- coding: utf-8 -*-
"""批量检测的端到端检查：模拟前端逐张调用 /predict，核对汇总与耗时。"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8020"


def get(p, timeout=8):
    return urllib.request.urlopen(BASE + p, timeout=timeout).read()


def post(name: str, blob: bytes):
    b = "----ui"
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
            f"Content-Type: image/png\r\n\r\n").encode() + blob + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + "/predict", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    d["_wall"] = (time.time() - t0) * 1000
    return d


def main() -> int:
    h = json.loads(get("/health"))
    print("模型:", h["model"], "| 设备:", h["device"].split("·")[0].strip())

    page = get("/").decode("utf-8", "replace")
    for k, ok in {
        "多选上传": 'multiple' in page,
        "队列缩略图": 'id="queue"' in page,
        "汇总面板": 'id="sOk"' in page and 'id="sBad"' in page,
        "进度条": 'id="pbar"' in page,
        "CSV 导出": '导出 CSV' in page,
        "详情双图": 'id="dOrig"' in page and 'id="dCam"' in page,
    }.items():
        print(f"   {'OK ' if ok else 'FAIL'} {k}")

    samples = json.loads(get("/samples"))["samples"]
    print(f"\n批量检测 {len(samples)} 张样例：")
    tally = {"ok": 0, "warn": 0, "bad": 0}
    t0 = time.time()
    for n in samples:
        blob = get("/sample/" + urllib.parse.quote(n), timeout=10)
        d = post(n, blob)
        tally[d.get("tone", "?")] = tally.get(d.get("tone", "?"), 0) + 1
        print(f"  {n:12s} {d['label_cn']:4s} {d['confidence']:.2f}  "
              f"服务端 {d['elapsed_ms']:4d} ms  端到端 {d['_wall']:5.0f} ms  "
              f"热力图={'有' if d.get('cam_png') else '无'}")
    dt = time.time() - t0
    print(f"\n合计 {len(samples)} 张，用时 {dt:.1f}s，平均 {dt/len(samples)*1000:.0f} ms/张")
    print(f"汇总：合格 {tally.get('ok',0)} / 待复核 {tally.get('warn',0)} / 判废 {tally.get('bad',0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
