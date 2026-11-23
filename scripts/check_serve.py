# -*- coding: utf-8 -*-
"""端到端验证演示服务：上传一张测试图，检查返回的分级与热力图。"""
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"


def get(path):
    with urllib.request.urlopen(HOST + path, timeout=20) as r:
        return r.read()


def post_file(path: Path):
    boundary = "----pvtest"
    data = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        HOST + "/predict", data=data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> int:
    # 等服务器起来
    for _ in range(30):
        try:
            health = json.loads(get("/health").decode())
            break
        except Exception:
            time.sleep(1)
    else:
        print("服务未启动")
        return 1
    print("health:", health)

    page = get("/").decode("utf-8", "replace")
    print("页面长度:", len(page), "含上传控件:", "type=\"file\"" in page)

    cls = sys.argv[2] if len(sys.argv) > 2 else "G3_严重"
    img = None
    for ds in ("ds4", "ds3", "ds2"):
        cand = sorted((ROOT / "artifacts" / ds / "test" / cls).glob("*.png"))
        if cand:
            img = cand[0]
            break
    if img is None:
        print(f"在测试集里找不到类别 {cls}")
        return 1
    t0 = time.time()
    res = post_file(img)
    dt = (time.time() - t0) * 1000
    print(f"\n上传 {img.name}（真实 {cls}）")
    print(f"  判定：{res['label_cn']}  置信度 {res['confidence']:.3f}  动作：{res['action']}")
    print("  各类概率：" + "  ".join(f"{p['name']}={p['p']:.3f}" for p in res["probs"]))
    print(f"  热力图：{'有' if res.get('cam_png') else '无'}  耗时 {dt:.0f} ms")
    if res.get("cam_error"):
        print("  CAM 错误：", res["cam_error"])
    if res.get("cam_png"):
        out = ROOT / "artifacts" / "cam_check" / "serve_demo.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(res["cam_png"]))
        print(f"  已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
