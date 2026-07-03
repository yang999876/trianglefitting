# 深度学习路线探索：用神经网络突破三角形拟合的局部最优

## 背景

当前主线工作流：Geometrize 贪心初始化 → pydiffvg 梯度下降优化。

核心瓶颈：pydiffvg 的梯度下降只能在 Geometrize 给出的布局附近做局部优化（颜色优化效果好，几何结构跳不出去）。Geometrize 是贪心算法——每步随机初始化 2000 个图形，爬山优化 L1 loss，选误差下降最大的一个。这导致三角形的空间分配和遮挡顺序被锁定在贪心路径上。

深度学习的核心价值：**不是"优化得更好"，而是学到一个从图像到三角形参数的直接映射（amortized inference），跳过逐图搜索。**

---

## 思路 1：预训练初始化网络，替代 Geometrize

**目标**：训练一个网络，输入图片，一次前向传播输出 N 个三角形的全部参数。

**训练信号来自可微渲染本身，不需要 ground truth 三角形标注**：

```
image → encoder → triangle params → differentiable renderer → rendered image
                                                                    ↓
                                              loss(rendered, target image)
```

仓库里的 U-Net 和 Transformer 路线已经在做单图 overfit。关键改变是**在大数据集上训练**（比如几万张动漫图片），让网络跨图学习"看到什么样的图，就该怎么摆三角形"。

### 为什么可能比 Geometrize 好

- Geometrize 无记忆——每张图从零贪心，不利用跨图经验
- 神经网络见过上万张图后，隐式学到"动漫图片通常哪里有大色块、哪里有细节边缘"，初始化布局更合理
- DETR 风格 Transformer 的 self-attention 让三角形之间协商分工，cross-attention 让它们感知图片内容

### 架构选型

仓库里的 `trianglefit.transformer`（DETR 风格）适合这个方向：每个 query 对应一个三角形 slot。

### 推荐使用方式

预训练模型的输出不是最终结果，而是 **diffvg test-time optimization 的初始化**。流程：

1. 预训练 Transformer 在数据集上训练
2. 推理时一次前向传播得到初始三角形参数
3. 灌进 diffvg，梯度下降微调几百步

### 实现要点

- 需要：动漫图片数据集（几千到几万张）、Dataset/DataLoader
- 从单图 overfit 到多图训练的改动不大——主要是数据加载和训练循环
- 损失函数：L1 + LPIPS，通过可微渲染器反传

### 难度与预期

- 实现难度：中
- 数据/算力：中（单卡可训）
- 能否超越 Geometrize：很可能

---

## 思路 2：自回归逐步放置（Autoregressive Triangle Placement）

**把三角形放置建模为序列生成问题**，用神经网络替代 Geometrize 的"随机采样 2000 个 + 爬山"。

```
观察当前残差图 → 预测下一个三角形参数 → 渲染叠加 → 更新残差 → 重复
```

### 网络结构

- 输入：当前渲染结果与目标图的残差（或拼接两者）
- 输出：下一个三角形的 8 个参数（cx, cy, base, height, theta, r, g, b）
- 架构：CNN encoder + MLP head，或 Vision Transformer

### 训练方式

三种可选路径，可以组合使用：

**A. 模仿学习（Behavioral Cloning）**
- 用 Geometrize 跑大量图，收集 (image, step_i, residual_i, best_triangle_i) 数据对
- 训练网络模仿 Geometrize 每一步的选择
- 优点：训练稳定、数据获取容易
- 缺点：受限于 Geometrize 的水平，无法超越老师

**B. 强化学习 / 策略梯度**
- 以最终图像质量为 reward
- 用 REINFORCE 或 PPO 训练放置策略
- 优点：有机会超越 Geometrize——网络可以学到"牺牲当前步、为后续步留出更好布局"的远见策略
- 缺点：RL 训练不稳定，序列长（几百步）导致 credit assignment 困难

**C. 混合：模仿学习预训练 + RL 微调**
- 先用 A 暖启动，再用 B 超越老师
- 实践中最可靠的路径

### 优势

- 天然支持可变数量的三角形（不需要预设 N）
- 每步决策基于全局残差，比贪心更有远见
- 可以学到高层策略（比如"先放大色块再放小细节"）

### 挑战

- 序列长度长（几百个三角形），训练成本高
- RL 方向需要较多调参经验

### 难度与预期

- 实现难度：中（模仿学习）/ 高（RL）
- 数据/算力：中 / 高
- 能否超越 Geometrize：可能（模仿学习持平或略好）/ 有潜力最强（RL）

---

## 思路 3：扩散模型生成三角形参数（Diffusion over Triangle Space）

**不在像素空间做 diffusion，在三角形参数空间做 diffusion。**

### 核心思路

- 定义三角形参数向量 x = flatten([N 个三角形的全部参数])
- 训练条件去噪网络：给定目标图片特征，从高斯噪声中逐步恢复出好的三角形参数
- 推理时从纯噪声采样，去噪得到三角形参数

### 为什么能跳出局部最优

- 去噪过程天然全局——不从特定初始化开始爬坡，而是从噪声空间"结晶"出解
- 多次采样可得到多个不同解，从中挑最好的
- 条件扩散学到 p(triangles | image) 的多模态分布，不坍缩到一个局部最优

### 实际做法

1. 先用 Geometrize + diffvg 生成大量 (image, optimized_triangle_params) 训练对
2. 训练条件扩散模型（如 1D DiT），条件是图片特征（CLIP / ResNet 提取）
3. 推理时采样多个候选，渲染后选最好的，再用 diffvg 微调

### 技术细节

- 三角形参数需要归一化到统一尺度（位置 [0,1]、角度 [-π,π]、颜色 [0,1]）
- 三角形之间排列不变性：可以用 set-based 扩散（如 SetDiffusion），或固定按某种规则排序
- 去噪网络：1D Transformer（把每个三角形的参数视为一个 token）

### 难度与预期

- 实现难度：高
- 数据/算力：高（需要大量预生成的 (image, triangles) 训练对）
- 能否超越 Geometrize：理论最强

---

## 思路 4：预训练 + Test-Time Optimization + 自我改进循环

这是最实用的工程路线，把上述思路串联成一个持续改进的系统。

### 三阶段流程

**Stage 1：大规模预训练**
- 收集动漫图片数据集
- 用 Transformer 架构在整个数据集上训练
- 损失：L1 + LPIPS，通过可微渲染器反传

**Stage 2：Test-Time Optimization**
- 每张新图，先用预训练模型前向推理得到初始三角形参数
- 灌进 diffvg，梯度下降微调几百步
- 初始化质量远好于 Geometrize，diffvg 的优化空间更大

**Stage 3：自我改进（Self-Improvement / Distillation）**
- 把 Stage 2 优化后的结果作为更好的"伪标注"
- 回馈训练集，distill 回预训练模型
- 形成 `预训练 → 推理 → 优化 → 更好的训练数据 → 更好的预训练` 循环

### 为什么有效

- Stage 1 给出比 Geometrize 更好的全局布局
- Stage 2 利用 diffvg 的精确梯度做局部精调
- Stage 3 让两个阶段互相增强——优化后的结果提升训练数据质量，更好的模型又给出更好的初始化供优化

---

## 优先级建议

| 优先级 | 方向 | 理由 |
|:---:|------|------|
| **1** | 预训练 Transformer + diffvg 微调（思路 1+4） | 两端代码已有，改动最小，回报最确定 |
| **2** | 自回归放置 - 模仿学习（思路 2A） | 可以用 Geometrize 廉价生成训练数据 |
| **3** | 自回归放置 - RL 微调（思路 2C） | 在 2A 基础上追加，有机会真正超越 Geometrize |
| **4** | 扩散模型（思路 3） | 理论最优但工程量大，适合前面走通后再探索 |

**推荐第一步**：把现有的 `trianglefit.transformer` 从单图 overfit 改为多图训练，接入 diffvg 做 test-time optimization，验证"神经网络初始化是否优于 Geometrize 初始化"这一核心假设。
