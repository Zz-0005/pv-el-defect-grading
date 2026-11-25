# -*- coding: utf-8 -*-
"""
用海康 MVS SDK（Python 绑定）采集真实 EL/可见光图像，存成与 ELPV 同构的目录。

本机已装 MVS 4.6，Python 绑定默认在：
    C:\\Program Files (x86)\\MVS\\Development\\Samples\\Python\\MvImport
（装在别的盘就改下面的 SDK_PATHS）
项目不复制这些文件（版权归海康），运行时把该目录加入 sys.path。

重要：当前机器**未接工业相机**，本脚本会检测不到设备并给出提示。
这是"有相机时怎么接"的通道，不是必需步骤 —— 数据集默认来自公开集。

用法：
    python scripts/mvs_capture.py --count 50 --out data/captured
    # 有相机：直接采；无相机：自动退化为"从现有图像复制"，保证目录结构可用
"""
from __future__ import annotations

import argparse
import ctypes
import shutil
import sys
import time
from pathlib import Path

SDK_PATHS = [
    Path(r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"),
    Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
]

# MVS SDK 常量（MvCameraControl.h）
MV_OK = 0
MV_GIGE_DEVICE = 0x00000001
MV_USB_DEVICE = 0x00000004


def find_sdk() -> Path | None:
    for p in SDK_PATHS:
        if (p / "MvCameraControl_class.py").exists():
            return p
    return None


def enumerate_cameras(sdk: Path) -> list[dict]:
    sys.path.insert(0, str(sdk))
    import MvCameraControl_class as Mv  # type: ignore

    device_list = Mv.MV_CC_DEVICE_INFO_LIST()
    mask = Mv.MV_GIGE_DEVICE | Mv.MV_USB_DEVICE
    ret = Mv.MvCamera.MV_CC_EnumDevices(mask, device_list)
    if ret != MV_OK:
        return []
    cam_list = []
    for i in range(device_list.nDeviceNum):
        info = ctypes.cast(device_list.pDeviceInfo[i],
                           ctypes.POINTER(Mv.MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType == Mv.MV_GIGE_DEVICE:
            n_ip = info.SpecialInfo.stGigEInfo.nCurrentIp
            name = bytes(info.SpecialInfo.stGigEInfo.chModelName).split(b"\x00")[0].decode("utf-8", "ignore")
            cam_list.append({"type": "GigE", "ip": f"{(n_ip >> 24) & 0xFF}.{(n_ip >> 16) & 0xFF}."
                                              f"{(n_ip >> 8) & 0xFF}.{n_ip & 0xFF}", "name": name})
        else:
            name = bytes(info.SpecialInfo.stUsb3VInfo.chModelName).split(b"\x00")[0].decode("utf-8", "ignore")
            cam_list.append({"type": "USB3", "ip": "-", "name": name})
    return cam_list


def grab(out_dir: Path, count: int) -> int:
    sdk = find_sdk()
    if sdk is None:
        print("未找到 MVS Python 绑定。请确认 MVS 已安装，或手动指定 SDK 路径。")
        return 2
    cams = enumerate_cameras(sdk)
    if not cams:
        print("MVS SDK 可用，但当前未检测到工业相机（设备列表为空）。")
        print("  接上相机后重跑本脚本即可采集；数据集默认使用公开的 ELPV 图像。")
        return 3

    import MvCameraControl_class as Mv  # type: ignore
    from PIL import Image
    import numpy as np

    print(f"检测到 {len(cams)} 台相机：{cams}")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for ci in range(len(cams)):
        cam = Mv.MvCamera()
        if cam.MV_CC_CreateHandleByIndex(ci) != MV_OK:
            continue
        if cam.MV_CC_OpenDevice(Mv.MV_ACCESS_Exclusive, 0) != MV_OK:
            cam.MV_CC_DestroyHandle()
            continue
        cam.MV_CC_SetEnumValue("TriggerMode", 0)     # 连续采集
        cam.MV_CC_StartGrabbing()
        try:
            while saved < count:
                data = Mv.MV_FRAME_OUT()
                if cam.MV_CC_GetImageBuffer(data, 1000) != MV_OK:
                    continue
                try:
                    info = data.stFrameInfo
                    n = info.nWidth * info.nHeight
                    buf = ctypes.string_at(data.pBufAddr, n)
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape(info.nHeight, info.nWidth)
                    Image.fromarray(arr, mode="L").save(
                        out_dir / f"cap_{ci}_{saved:04d}.png", optimize=True)
                    saved += 1
                finally:
                    cam.MV_CC_FreeImageBuffer(data)
        finally:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
    print(f"采集完成：{saved} 张 -> {out_dir}")
    return 0


def fallback_copy(src: Path, out: Path, count: int) -> int:
    """没有相机时，从现成图像复制若干张，保证下游目录结构可跑通。"""
    if not src.exists():
        print(f"备用源不存在：{src}")
        return 1
    imgs = sorted(src.glob("*.png"))[:count]
    out.mkdir(parents=True, exist_ok=True)
    for p in imgs:
        shutil.copyfile(p, out / p.name)
    print(f"无相机，已从 {src} 复制 {len(imgs)} 张 -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--out", default="data/captured")
    ap.add_argument("--fallback-src", default="")
    a = ap.parse_args()
    rc = grab(Path(a.out), a.count)
    if rc == 3 and a.fallback_src:
        return fallback_copy(Path(a.fallback_src), Path(a.out), a.count)
    return rc


if __name__ == "__main__":
    sys.exit(main())
