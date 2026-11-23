# -*- coding: utf-8 -*-
"""
Grad-CAM：让分级结果落到"缺陷在哪"。

分级模型只输出一个级别，评委自然会问"你凭什么说它是严重缺陷"。
Grad-CAM 把最后一片特征图上对目标类贡献最大的区域映射回原图，
既是可解释性，也顺带验证模型没有"看错地方"（比如只看边框不看电池片）。

实现要点：
  · 不依赖第三方 CAM 库，直接挂 hook；
  · 自动选取最后一个输出为 4D 的模块作为目标层，避免写死层名
    （不同 ults 版本/不同 backbone 层名不一致，写死必然失效）；
  · 输出与原图同尺寸的 float[0,1] 热力图。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, torch_model: torch.nn.Module):
        self.model = torch_model.eval()
        self._acts: list[tuple[torch.nn.Module, torch.Tensor]] = []
        self._handles = []
        for mod in self.model.modules():
            if len(list(mod.children())) == 0:      # 只看叶子模块
                self._handles.append(mod.register_forward_hook(self._hook))

    def _hook(self, mod, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4 and out.requires_grad:
            out.retain_grad()
            self._acts.append((mod, out))

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
        """x: (1,3,H,W)。返回 (cam[0,1] HxW, 预测类索引, 类别概率)。"""
        self._acts.clear()
        self.model.zero_grad(set_to_none=True)
        # ultralytics 的推理路径会包在 no_grad / inference_mode 里，
        # 那样特征图 requires_grad=False，拿不到梯度。这里强制开启梯度。
        self.model.requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if logits.dim() == 3:                       # 某些版本输出 (1,1,C)
            logits = logits[:, 0]

        # 关键：ultralytics 的分类头输出**已经是 softmax 概率**（各元素和为 1）。
        # 再套一次 softmax 会把分布二次压缩，实测与 predict 的概率最大偏差可达 0.52，
        # 等于 CAM 解释的不是模型真正给出的结论。这里按"和是否为 1"判断，避免重复归一化。
        row = logits[0].detach()
        already_prob = bool(row.min() >= -1e-6 and abs(float(row.sum()) - 1.0) < 1e-3)
        probs_t = row if already_prob else F.softmax(logits, dim=1)[0]
        probs = probs_t.cpu().numpy()
        idx = int(probs.argmax()) if class_idx is None else int(class_idx)

        if not self._acts:
            raise RuntimeError("未捕获到 4D 特征图，无法计算 CAM")

        # 目标层选择：**不要用最后一层**。
        # 实测（scripts/layer_probe.py，224 输入，G3 样本）：
        #   最后一层 7×7 特征图  → 缺陷与背景的 CAM 对比度只有 1.2~1.4
        #   中间 14×14 特征图    → 3~5
        #   较浅 28×28 特征图    → 2~9
        # 原因是深层特征图空间分辨率太低，上采样回原图后糊成一片，
        # 热力图看起来"哪里都亮"，无法指示缺陷位置。
        # 规则：取"最深的、空间尺寸不小于输入 1/16 的特征图"。
        h_in, w_in = x.shape[-2:]
        min_side = max(8, int(round(min(h_in, w_in) / 16)))
        target = None
        for _mod, act in self._acts:                 # 由浅到深，持续覆盖
            if act.shape[-1] >= min_side and act.shape[-2] >= min_side:
                target = act
        if target is None:
            target = self._acts[-1][1]
        # 再排除整层为常数的退化情况（ReLU 之后全零在实测中确实出现）
        if float(target.detach().std()) < 1e-9:
            for _mod, act in reversed(self._acts):
                if float(act.detach().std()) > 1e-6:
                    target = act
                    break
        self.model.zero_grad(set_to_none=True)
        logits[0, idx].backward(retain_graph=False)

        grad = target.grad
        if grad is None:
            raise RuntimeError("目标层梯度为空")
        w = grad.mean(dim=(2, 3), keepdim=True)          # (1,C,1,1)
        cam = F.relu((w * target).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        lo, hi = float(cam.min()), float(cam.max())
        if hi - lo < 1e-8:
            # 梯度贡献全被 ReLU 置零时的退化情况（实测约 1/8 的样本会这样）。
            # 从后往前找第一个"在空间上有变化"的特征图，用它的通道均值当热力图。
            # 加一步形状检查是必要的：某些层取完均值后是常数，归一化会得到全 0。
            for _mod, act in reversed(self._acts):
                a = act.detach().mean(dim=1, keepdim=True).abs()
                a = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                a = a.cpu().numpy()
                if float(a.max() - a.min()) > 1e-6:
                    cam, lo, hi = a, float(a.min()), float(a.max())
                    break
        if hi - lo < 1e-8:
            return np.full(x.shape[-2:], 0.5, dtype=np.float32), idx, probs
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam.astype(np.float32), idx, probs


def preprocess(gray_or_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    与 ultralytics 推理**完全一致**的预处理。

    坑：自己写 Resize(224,224)+/255 看起来等价，但 ultralytics 实际是
    `Resize(短边) → CenterCrop(size) → ToTensor → Normalize(mean,std)`。
    所以这里直接复用 ultralytics 自己的变换函数，不重复实现。

    （另有一个更隐蔽的坑在 __call__ 里：分类头输出已经是 softmax 概率，
    二次 softmax 才是概率对不上的主因，比预处理差异严重得多。）
    """
    from PIL import Image
    from ultralytics.data.augment import classify_transforms

    if gray_or_rgb.ndim == 2:
        im = Image.fromarray(gray_or_rgb.astype(np.uint8), mode="L").convert("RGB")
    else:
        im = Image.fromarray(gray_or_rgb.astype(np.uint8), mode="RGB")
    return classify_transforms(size)(im).unsqueeze(0)


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """把热力图叠加到灰度原图上，返回 RGB uint8。"""
    from PIL import Image
    h, w = gray.shape[:2]
    base = np.stack([gray] * 3, axis=-1).astype(np.float32)
    cam_r = np.asarray(Image.fromarray((cam * 255).astype(np.uint8), mode="L")
                       .resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

    # 简单 jet 色标：低→蓝，中→绿/黄，高→红
    r = np.clip(1.5 * cam_r - 0.3, 0, 1)
    g = np.clip(1.6 * np.minimum(cam_r, 1 - cam_r) * 2, 0, 1)
    b = np.clip(1.2 - 1.8 * cam_r, 0, 1)
    heat = np.stack([r, g, b], axis=-1) * 255.0
    out = base * (1 - alpha) + heat * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
