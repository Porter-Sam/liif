# Fusion-LIIF Gray Handoff

## 当前目标

在原 LIIF 框架上做红外-可见光融合实验。第一版采用灰度融合：

- `VI` 转为单通道灰度。
- `IR` 为单通道。
- 输入尺寸固定为 `128 x 128`。
- 查询生成尺寸固定为 `256 x 256`。
- 使用 LIIF 的隐式坐标查询方式，但输入从单图像改为双模态 `VI + IR`。

核心想法是：用两个 EDSR encoder 分别提取 VI/IR 特征，再通过查询点相关的特征距离计算连续融合权重，避免手工显著图和 0/1 门控。

## 方法概述

给定对齐输入：

```text
VI: 1 x 128 x 128
IR: 1 x 128 x 128
```

双 encoder 提取特征：

```math
z_{vi}=E_{vi}(VI), \quad z_{ir}=E_{ir}(IR)
```

再投影到 metric space：

```math
\tilde z_{vi}=P_{vi}(z_{vi}), \quad \tilde z_{ir}=P_{ir}(z_{ir})
```

对查询点 `q`，模型构造 query metric feature，并计算到 VI/IR 特征的距离：

```math
D_{vi}(q)=\|\psi(q)-\tilde z_{vi}(q)\|^2
```

```math
D_{ir}(q)=\|\psi(q)-\tilde z_{ir}(q)\|^2
```

连续权重：

```math
[w_{vi}, w_{ir}]
=
\operatorname{softmax}(-D_{vi}/\tau, -D_{ir}/\tau)
```

融合特征：

```math
z_f(q)=w_{vi}(q)z_{vi}(q)+w_{ir}(q)z_{ir}(q)
```

然后由 LIIF MLP decoder 输出查询点像素值：

```math
\hat F(q)=MLP(z_f(q), q, cell)
```

## 已修改文件

### `models/fusion_liif.py`

新增模型 `fusion-liif`。

主要功能：

- 双 encoder：`vi_encoder` 和 `ir_encoder`。
- 单通道灰度输入。
- `vi_metric` / `ir_metric` 做低维 metric projection。
- `query_metric` 根据局部 VI/IR metric feature 和相对坐标生成查询度量特征。
- 根据距离 softmax 得到 VI/IR 连续权重。
- 记录可视化变量：
  - `last_modality_weight`
  - `last_modality_distance`
- 支持加载原 LIIF 权重：
  - `pretrained_liif: ./edsr-baseline-liif.pth`
  - RGB 第一层卷积权重会自动平均为单通道。
  - 原 LIIF 的 MLP 输出 3 通道，加载到当前 1 通道 MLP 时会对最后输出层做均值适配。
- 支持冻结 encoder：
  - `freeze_encoder: true`

### `models/edsr.py`

EDSR 构造函数新增 `n_colors` 参数。

当前 fusion 配置使用：

```yaml
n_colors: 1
```

### `models/__init__.py`

注册新增模型文件：

```python
from . import fusion_liif
```

### `datasets/image_folder.py`

新增三文件夹数据集：

```text
triple-image-folders
```

返回：

```python
(vi, ir, gt)
```

用于监督融合训练。

### `datasets/wrappers.py`

新增：

```text
fusion-implicit-paired
```

功能：

- 接收 paired 或 triple 数据。
- 对 `VI/IR/GT` 做共享裁剪，保证严格对齐。
- 转灰度。
- 将 `VI/IR` resize 到 `128 x 128`。
- 将 `GT` 保持为 `256 x 256` 查询目标。
- 返回：

```python
{
    "inp": vi_lr,
    "vi": vi_lr,
    "ir": ir_lr,
    "coord": hr_coord,
    "cell": cell,
    "gt": hr_pixel_samples,
    "gt_img": gt_hr,
}
```

同时修复了 `utils.to_pixel_samples`，现在支持单通道，不再写死 3 通道。

### `train_liif.py`

新增 fusion 分支：

- 如果模型有 `is_fusion_liif=True`，训练时读取 `vi/ir`。
- 否则保持原 LIIF 单输入逻辑。

新增验证可视化：

每次 `epoch_val` 后保存：

```text
IR | VI | Distance | Pred | GT
```

路径：

```text
save/<实验名>/visual/epoch-xxxxxx.png
```

其中 `Distance` 当前显示归一化后的：

```math
D_{ir}-D_{vi}
```

### `test.py`

同步兼容 `fusion-liif` 的验证与 batched query。

### `configs/train-div2k/train_edsr-baseline-fusion-liif-gray.yaml`

新增/当前主配置。

关键设置：

```yaml
model:
  name: fusion-liif
  args:
    encoder_spec:
      name: edsr-baseline
      args:
        no_upsampling: true
        n_colors: 1
    imnet_spec:
      name: mlp
      args:
        out_dim: 1
        hidden_list: [256, 256, 256, 256]
    metric_dim: 32
    temperature: 0.1
    pretrained_liif: ./edsr-baseline-liif.pth
    freeze_encoder: true
```

数据设置：

```yaml
train_dataset:
  dataset:
    name: triple-image-folders
    args:
      root_path_1: ./load/fusion/train/vi
      root_path_2: ./load/fusion/train/ir
      root_path_3: ./load/fusion/train/gt
  wrapper:
    name: fusion-implicit-paired
    args:
      inp_size: 128
      out_size: 256
      sample_q: 8192
```

验证同理：

```yaml
./load/fusion/val/vi
./load/fusion/val/ir
./load/fusion/val/gt
```

## 数据目录要求

默认需要如下结构：

```text
E:/YuShihang/project/liif/load/fusion/train/vi
E:/YuShihang/project/liif/load/fusion/train/ir
E:/YuShihang/project/liif/load/fusion/train/gt

E:/YuShihang/project/liif/load/fusion/val/vi
E:/YuShihang/project/liif/load/fusion/val/ir
E:/YuShihang/project/liif/load/fusion/val/gt
```

要求：

- 三个文件夹中文件排序后一一对应。
- 图像需要已经配准。
- 可以是 RGB 或灰度，wrapper 会统一转灰度。
- 训练时会共享裁剪窗口，避免模态错位。

如果暂时没有 `gt`，可以把配置里的 dataset 改为：

```yaml
name: paired-image-folders
args:
  root_path_1: ./load/fusion/train/vi
  root_path_2: ./load/fusion/train/ir
```

此时 wrapper 会根据 `target_mode` 临时构造目标：

```yaml
target_mode: avg
```

可选：

```text
avg / max / vi / ir
```

不过正式监督训练建议使用三文件夹 `VI/IR/GT`。

## 训练命令

在 `boat` 环境中运行：

```bash
conda activate boat
cd E:/YuShihang/project/liif
python train_liif.py \
  --config configs/train-div2k/train_edsr-baseline-fusion-liif-gray.yaml \
  --name fusion_liif_gray_x2
```

Windows 下也可以：

```powershell
conda run -n boat python train_liif.py --config configs/train-div2k/train_edsr-baseline-fusion-liif-gray.yaml --name fusion_liif_gray_x2
```

## 当前训练策略

第一版建议：

```text
EDSR encoder: frozen
metric projection: train
query_metric: train
MLP decoder: train
```

原因：

- EDSR 保持 LIIF 预训练自然图像特征。
- metric/query 模块负责学习 VI/IR 距离。
- MLP decoder 需要适配融合特征分布，不能完全冻结。

## 已验证内容

使用 `boat` 环境做过 smoke test：

```text
VI input: 1 x 128 x 128
IR input: 1 x 128 x 128
GT image: 1 x 256 x 256
Pred query: 1 x N x 1
```

可视化 smoke 保存成功：

```text
IR | VI | Distance | Pred | GT
```

注意：Windows 环境下 `conda run` 会输出：

```text
& was unexpected at this time.
The value specified in an AutoRun registry key could not be parsed.
```

但命令退出码为 0，Python 逻辑正常执行。这是本机 AutoRun/conda shell 提示，不是模型错误。

## 当前 Git 状态说明

当前正式代码改动主要包括：

```text
configs/train-div2k/train_edsr-baseline-fusion-liif-gray.yaml
datasets/image_folder.py
datasets/wrappers.py
models/fusion_liif.py
models/edsr.py
models/__init__.py
test.py
train_liif.py
utils.py
```

`edsr-baseline-liif.pth` 是用户放入的预训练权重文件，当前可能显示为 untracked。

## 后续建议

1. 先用合成融合数据跑通监督训练。
2. 看 `Distance` 图是否出现有意义的区域差异。
3. 如果权重长期接近 0.5，可以：
   - 降低 `temperature`，如 `0.05`。
   - 给 `last_modality_weight` 加轻微熵约束。
   - 给 metric projection 更高学习率。
4. 如果输出过平滑：
   - 增加 `sample_q`。
   - 解冻 MLP 已经开启；必要时后期解冻 EDSR body 的后几层。
5. 如果距离图噪声大：
   - 对权重或距离加 TV/smooth loss。
   - 从逐像素距离切到 patch/token 距离。

