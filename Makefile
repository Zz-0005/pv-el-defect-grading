# pv-inspect 便捷入口
# Windows 无 make 时可直接用： python -m pvinspect <cmd>
PY ?= .venv\Scripts\python.exe

.PHONY: help env data prepare train eval serve test clean all

help:
	@echo "make env      建虚拟环境并装依赖"
	@echo "make data     下载 ELPV 数据集（93 MB）"
	@echo "make prepare  分级标注 + 分组切分 + 泄漏��查"
	@echo "make train    训练 E1_grouped / E1_random / E2_binary"
	@echo "make eval     冻结测试集评测并出图"
	@echo "make serve    启动本地演示服务"
	@echo "make test     21 项自检"
	@echo "make all      data -> prepare -> train -> eval"

env:
	python -m venv .venv
	python -m pip --python .venv\Scripts\python.exe install -r requirements.txt

data:
	$(PY) scripts/fetch_data.py

prepare:
	$(PY) -m pvinspect prepare

train:
	$(PY) -m pvinspect train --epochs 40

eval:
	$(PY) -m pvinspect eval

serve:
	$(PY) -m pvinspect serve

test:
	$(PY) -m pvinspect selftest

clean:
	rmdir /s /q artifacts 2>nul || true
	rmdir /s /q .tmp 2>nul || true

all: data prepare train eval
