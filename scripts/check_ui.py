# -*- coding: utf-8 -*-
"""检查新版演示页：接口、样例、防穿越、页面关键元素。"""
import json
import time
import urllib.parse
import urllib.request

B = "http://127.0.0.1:8000"


def get(path, timeout=8):
    return urllib.request.urlopen(B + path, timeout=timeout).read()


h = json.loads(get("/health"))
print("health :", {k: h[k] for k in ("model", "device", "classes")})

s = json.loads(get("/samples"))
samples = s["samples"]
print("samples:", samples)

name = samples[0]
t0 = time.time()
data = get("/sample/" + urllib.parse.quote(name), timeout=10)
print(f"sample : {name}  {len(data)/1024:.0f} KB  {time.time()-t0:.2f}s  PNG头={data[:4]!r}")

page = get("/").decode("utf-8", "replace")
checks = {
    "样例按钮容器": 'id="slist"' in page,
    "粘贴支持": "paste" in page,
    "原图/热力图双栏": 'id="tOrig"' in page and 'id="tCam"' in page,
    "语义配色": "tone-ok" in page and "tone-bad" in page,
    "响应式": "@media" in page,
    "旧文案已清除": "YOLOv8n" not in page,
}
print("page   :", len(page), "字符")
for k, v in checks.items():
    print(f"   {'OK ' if v else 'FAIL'} {k}")

try:
    urllib.request.urlopen(B + "/sample/..%2f..%2f..%2fWindows%2fwin.ini", timeout=5)
    print("穿越防护: FAIL 未拦截")
except Exception as e:
    print(f"穿越防护: OK 已拦截（{type(e).__name__}）")
