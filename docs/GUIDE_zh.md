# 使用与维护完整指南

本仓库对应当前 v4.3 manuscript。英文正文位于 `paper/manuscript_v4_3_en.md`，PDF 同目录；中文初稿和对齐后的中文补充材料也保留。英文稿的数据可用性段落已改为当前仓库信息，其余研究内容按新版翻译。补充材料本次没有翻译成英文。

## 为什么保留这些文件

| 文件/目录 | 必要性 |
| --- | --- |
| `src/phase_reference/` | 几何、参考相位、网络、推理、训练及评估的便携实现 |
| `third_party/PIC-Flow/FDTD/` | 原始几何、源、端口、有效折射率表；只复制脚本而遗漏材料表会导致输入不一致 |
| `third_party/NeurOLight/` | 网络和波动先验原始函数体；保留原 MIT 许可证 |
| `configs/calibration.json` | 训练集拟合的 beta 与振幅尺度，测试时必须冻结 |
| `configs/checkpoints.json`、12 个 `best.pt` | 三个种子 × 四种方法及其校验和，避免误传其他种子或旧方法的权重 |
| `data/manifests/` | 保留原划分、几何和端口，才能进行相同结构上的配对比较 |
| `data/fields/` | 原始 180、独立 72、正弦 36，共 288 份主分辨率场，可直接评分和训练 |
| `results/` | 支撑正文与补充材料的结构级指标、统计、计时及诊断证据 |
| `historical_source/` | 高网格、Y 分支、原始训练与诊断程序存档，说明历史实验如何完成 |
| `paper/`、`docs/`、`validation/` | 当前论文、现有图片、数据字典、来源和发布检查 |

不上传重复场缓存、运行日志、虚拟环境、系统文件、SSH 密钥、优化器中间状态和全部高分辨率大场。每份精简 NPZ 内保留的 `.npy` 成员逐字节来自原文件，未重新计算数值。`data/field_manifest.json` 同时记录原文件与导出文件的哈希。文件来源与适配记录见 `docs/source_provenance.json`。

## 1. 获取与安装

推荐 Linux 或 Windows 的 WSL2；Meep 的材料初始化在 WSL 中完成。详细英文命令也见根目录 README。

```bash
git clone https://github.com/Jacksonhy/geometry-phase-sbend.git
cd geometry-phase-sbend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 按显卡与驱动安装合适的 PyTorch。下面是本次实测环境的构建：
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/requirements-gpu.txt
python -m pip install --no-deps -e .
```

使用 editable 安装并保留整个仓库。模型、材料表和数据通过源代码所在 checkout 定位，不能只复制安装后的 Python 包。权重和主场均直接在 Git 中，无需另行下载或安装 Git LFS。磁盘至少预留数 GiB，以容纳 Git、输入缓存和评估输出；新训练还需要 GPU 显存。

## 2. 先做无需 GPU 的检查

```bash
python scripts/verify_assets.py
python scripts/verify_tables.py --tables results --output outputs/table-check.json
```

第一项检查 12 个权重、288 个字段文件及图片/源码来源的哈希；第二项只重算存档统计表的算术关系，不会运行 FDTD 或模型。两项通过不等于所有科学实验已重新运行。

## 3. 复算主结果

```bash
# 先检查全部 12 个模型各预测 2 个结构
python -m phase_reference.evaluate --limit 2 --output outputs/smoke
# 独立欧拉集：72 个结构 × 12 个模型
python -m phase_reference.evaluate --dataset euler_independent72 --output outputs/euler72
# 正弦迁移：36 个结构 × 12 个模型
python -m phase_reference.evaluate --dataset sinusoidal36 --output outputs/sine36
# 原始回顾性测试的 100 个结构
python -m phase_reference.evaluate --dataset euler_original180 --split test --output outputs/original100
```

输出 `per_geometry.csv` 和 `summary.json`。后者逐种子统计；论文的“均值 ± 标准差”来自三个种子各自的结构平均误差。不要把 72 × 3 当作 216 个独立几何结构。要缩小范围可加 `--methods arc_only local_tangent --seeds 2026`。CPU 可加 `--device cpu`，但本次对照核验使用 GPU。

独立集全场 RelL2 的三种子均值应约为 Linear 0.3636、ArcOnly 0.1610、LocalTangent 0.1472、Path v2 0.1474。跨设备/库版本允许小的浮点差异。指标定义、单位和数组方向详见 `docs/DATA.md`。

## 4. 只给几何做新预测

创建带 Meep 的独立环境：

```bash
conda create -n phase-meep -c conda-forge python=3.11 pymeep=1.35.1 scipy
# 在仓库根目录，把包也以 editable 方式装入这个环境
conda run -n phase-meep python -m pip install --no-deps -e .
conda run -n phase-meep python -m phase_reference.predict --geometry-json examples/geometry.json --prepare-only --output outputs/inputs
# 回到 GPU Python 环境：
python -m phase_reference.predict --geometry-json examples/geometry.json --prepared-inputs outputs/inputs --method local_tangent --seed 2026 --output outputs/query
```

也可用 `--meep-python /实际的/Meep环境/bin/python` 代替 `--prepared-inputs`，自动调用材料预处理。两个环境都必须能够导入本仓库包。

输入格式示例：

```json
{
  "geometry_id": "query_001",
  "params": {"wg_width_um": 0.4, "lateral_offset_um": 3.5, "R_min_um": 5.85},
  "input_port": 1
}
```

欧拉参数限于本文考察范围与 0.025 μm 格点，并满足器件适配条件。`input_port` 1/2 对应左/右侧激励。批量输入可用 JSON 列表或 `{"records": [...]}`。正弦结构用 `length_um` 取代 `R_min_um`，已有示例见正弦清单。参数范围之外的准确性没有被本文证实。

预处理只初始化材料到 Meep 时间 0，不运行时域传播，不读取待预测结构真值。输入文件同时保存四种载波；更换输出方法不会改变共同输入。不要删除 Path v1 输入特征。推理前会核对几何、端口与标定哈希，避免误用旧缓存。重新准备同名输入时请选新输出目录，以免静默覆盖。

## 5. 重新训练

```bash
python -m phase_reference.train --method local_tangent --seed 2026 --output outputs/train-lt-2026
```

完整比较需要四种方法 × 三个种子全部运行。默认 2400 次更新，64/16 固定训练/验证划分，其他超参数与 README 一致。训练程序不读取测试场。`--steps 2` 仅用于运行检查，不代表完成论文训练。便携 CLI 为全新训练，不支持断点续训；原始带恢复功能的脚本在 `historical_source/` 中。

## 6. 论文与图片

正文只保留当前新版的四张图，对应原图 1、2、3、6；原图 6 在新版编号为图 4。没有补回新版删除的正文部分。补充图 S6 来自原参考诊断图，S7 来自原正弦场示例图；来源清单见 `docs/PROVENANCE.md`。

英文 Markdown 转 PDF：

```bash
python scripts/export_manuscript.py --pandoc /实际路径/pandoc --chrome /实际路径/chrome
```

图片和公式嵌入临时本地 HTML，用独立无头浏览器配置输出 A4 PDF。正文科学数值与公式按 v4.3 保留，数据可用性段落已反映当前公开仓库。作者信息、声明和目标期刊格式仍需按投稿要求补充。

## 7. 历史实验与复现边界

本包可直接运行主分辨率 S 弯训练、推理及主要场/相位指标。高网格场、直波导原始诊断场、完整 Y 分支原始数据和所有探索权重没有一并发布；相应程序与证据表已存档。`historical_source/` 中保留原绝对路径及实验之间的依赖，运行这些程序需要按原组织方式准备输入和适配路径，不能把它们当作已全部移植的入口。完整边界见 `docs/PROVENANCE.md`。

论文计时沿用历史硬件和实测边界，不是新机器上的速度承诺。本次发布核验范围记录于 `validation/`。没有把全部 FDTD、高网格或 12 次完整训练重新运行一遍。

## 8. 今后更新 GitHub

在本地仓库修改后，先查看差异，再明确暂存需要公开的文件：

```bash
git status --short
git diff
git add README.md docs/ paper/
git diff --cached --stat
git commit -m "Update manuscript and documentation"
git push origin main
```

更改权重、数据或来源文件时，要同步更新相应哈希清单并运行校验。不要直接 `git add .` 把新缓存或私有文件混入。要发布可引用版本，可创建新的版本标签；不可把软件仓库维护者账号当作论文作者清单，也不要填写尚未取得的 DOI。

保留第三方 MIT 许可证。当前没有替原创论文、数据和代码擅自选择一项统一开源许可证；授权状态见根目录 `LICENSING.md`。
