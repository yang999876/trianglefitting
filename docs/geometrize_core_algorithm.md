# Geometrize 核心算法总结：用基本图元拟合图片

## 源码定位

本仓库中的第三方应用代码位于 `third-party/geometrize`。真正的拟合算法不在 Qt UI 层，而在其子模块 `third-party/geometrize/lib/geometrize` 中，核心文件如下：

- `geometrize/runner/imagerunner.*`：应用层调用入口，负责把参数转换成 `Model::step` 调用。
- `geometrize/model.*`：维护目标图、当前图、当前误差，并决定是否接收新图元。
- `geometrize/core.*`：候选搜索、爬山优化、颜色估计、误差计算。
- `geometrize/shape/*`：图元类型、随机初始化、随机变异。
- `geometrize/rasterizer/*`：把图元转成 scanline，并把 scanline 绘制到 bitmap。

一句话概括：Geometrize 是一个逐步叠加图元的贪心随机搜索算法。每一步随机生成一批候选图元，对候选做局部爬山，选出让当前图像误差下降最多的一个图元，计算它的最佳颜色，把它叠加到当前图像上。

---

## 整体流程

`ImageRunner` 是外部最常用的入口。它持有一个 `Model`，每次调用 `step(options)` 只推进一轮拟合。

初始化时：

1. 输入目标图 `target`。
2. 若没有指定初始图，`Model` 用目标图的平均颜色填满整张 `current`。
3. 计算 `lastScore = differenceFull(target, current)`，作为当前全图误差。

每一轮 `step`：

```text
输入：目标图 target，当前图 current，允许的图元类型，alpha，候选数量，变异次数，线程数

1. 根据 shapeTypes 和 shapeBounds 构造 shapeCreator
2. 并行启动若干个搜索任务
3. 每个任务：
   a. 随机生成若干候选图元
   b. 对其中较好的候选做随机变异爬山
   c. 返回该线程找到的最低误差状态
4. 从所有线程返回的状态中选误差最低者
5. 重新计算该图元的最佳颜色
6. 试画到 current 上，增量计算 newScore
7. 若 newScore < lastScore，接收该图元；否则回滚
```

默认参数在 `ImageRunnerOptions` 中：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `shapeTypes` | `ELLIPSE` | 允许搜索的图元类型，位掩码，可组合 |
| `alpha` | `128` | 新图元透明度 |
| `shapeCount` | `50` | 每轮随机候选数量的控制参数 |
| `maxShapeMutations` | `100` | 局部爬山的变异耐心/上限 |
| `seed` | `9001` | 随机种子 |
| `maxThreads` | `0` | 0 表示自动使用硬件并发数 |

注意：`shapeCount` 不是每步添加的图元数。一次 `Model::step` 最多接收一个图元；如果没有找到能降低误差的图元，则返回空结果。

---

## 图元状态与贪心接收

算法里的候选解叫 `State`，包含：

- `m_shape`：一个图元对象。
- `m_alpha`：图元透明度。
- `m_score`：如果把该图元画到当前图上，新的全局误差会是多少。

`Model` 维护两个 bitmap：

- `target`：目标图。
- `current`：当前已由图元叠加出的近似图。

默认接收条件非常简单：

```text
newScore < lastScore
```

也就是说，它只接收让误差下降的图元。这一点决定了 Geometrize 的本质是贪心叠加：旧图元不会被重新优化，遮挡顺序也随着添加顺序固定下来。

---

## 候选搜索与爬山优化

核心搜索逻辑在 `core.cpp`：

1. `bestRandomState` 随机生成一批图元。
2. 每个图元先 `setup` 随机初始化参数。
3. 计算图元对应的能量/误差。
4. 选出误差最低的随机候选。
5. `hillClimb` 对这个候选做局部变异优化。

爬山过程是严格贪心的：

```text
best = 初始候选
age = 0

while age < maxAge:
    old = 当前状态副本
    mutate(当前状态)
    score = energy(当前状态)

    if score >= bestScore:
        回滚到 old
        age += 1
    else:
        best = 当前状态
        bestScore = score
        age = 0
```

代码里 `age` 在发现更优解时会重置，因此 `maxShapeMutations` 更接近“连续多少次变异失败后停止”，而不是严格的总变异次数。

多线程不是把一批候选分片后汇总，而是让每个线程独立跑一套候选搜索和爬山。线程使用 `seed + offset` 重置 thread-local RNG，最后 `Model` 从所有线程结果中选 `m_score` 最小的状态。

---

## 颜色估计

Geometrize 不把颜色也放进随机搜索，而是在给定图元覆盖区域和固定 alpha 后，直接估计一个最合适的 RGB。

对图元覆盖的 scanline 像素，代码近似求解这个 alpha blending 方程：

```text
target ~= shapeColor * alpha + current * (1 - alpha)
```

因此：

```text
shapeColor ~= (target - current) / alpha + current
```

实现上会遍历图元覆盖区域内的像素，对每个像素根据 `target` 与 `current` 的差值反推出需要的颜色，然后取平均并 clamp 到 `[0, 255]`。返回颜色的 alpha 固定为当前 `options.alpha`。

这个设计很关键：搜索空间只剩几何参数，颜色由闭式近似给出，因此随机搜索效率高很多。

---

## 误差函数

默认误差是归一化 RMSE，包含 RGBA 四个通道：

```text
score = sqrt(sum((target - current)^2) / (width * height * 4)) / 255
```

`differenceFull` 会扫完整张图，用于初始化。

每次试画一个图元时，Geometrize 不重新扫描全图，而是调用 `differencePartial` 做增量更新：

1. 从旧的全图 RMSE 反推旧的平方误差总和。
2. 对图元 scanline 覆盖的像素：
   - 减去这些像素在 `before` 中的旧平方误差。
   - 加上这些像素在 `after` 中的新平方误差。
3. 再转回归一化 RMSE。

这种增量计算依赖一个事实：本轮只有图元覆盖区域发生了变化。它是 Geometrize 能快速评估大量候选的核心优化之一。

---

## Scanline 光栅化

所有图元最终都会被转换成一组水平线段：

```text
Scanline(y, x1, x2)
```

之后颜色估计、试画、增量误差计算都只遍历这些 scanline 覆盖的像素。

各图元的光栅化方式大致如下：

| 图元 | 光栅化方式 |
|---|---|
| `RECTANGLE` | 按 y 逐行生成 `[x1, x2]` |
| `CIRCLE` | 在半径方框内检查 `x^2 + y^2 <= r^2` |
| `ELLIPSE` | 按椭圆方程生成水平 span |
| `LINE` | Bresenham 线算法，每个点是长度为 1 的 scanline |
| `POLYLINE` | 多段 Bresenham，并去重像素 |
| `QUADRATIC_BEZIER` | 采样 20 个点，再用 Bresenham 连接 |
| `TRIANGLE` | 先求边界像素，再对每个 y 取最小 x 到最大 x 填充 |
| `ROTATED_RECTANGLE` | 求旋转后的四角，当作多边形填充 |
| `ROTATED_ELLIPSE` | 采样 20 个椭圆边界点，当作多边形填充 |

最后会用 `trimScanlines` 裁剪到允许绘制的边界内。

这个光栅化是像素级离散光栅化，不做抗锯齿；柔和程度主要来自图元 alpha blending。

---

## 图元生成与变异

支持的图元类型是位掩码枚举：

```text
RECTANGLE
ROTATED_RECTANGLE
TRIANGLE
ELLIPSE
ROTATED_ELLIPSE
CIRCLE
LINE
QUADRATIC_BEZIER
POLYLINE
```

`createDefaultShapeCreator` 会根据 `shapeTypes` 从允许类型里均匀随机选一种，然后绑定该类型的：

- `setup`：随机初始化参数。
- `mutate`：随机扰动参数。
- `rasterize`：转换为 scanline。

初始化倾向于产生局部小图元。例如三角形以一个随机点为基准，另外两个点在约 `[-32, 32]` 范围内偏移；矩形、圆、椭圆的初始尺寸也通常在几十像素尺度内。

变异同样是局部扰动。例如：

- 三角形每次随机选一个顶点，坐标扰动约 `[-32, 32]`。
- 圆每次移动中心或改变半径，扰动约 `[-16, 16]`。
- 旋转矩形可能移动两个角点之一，或小幅改变角度。
- 二次贝塞尔曲线随机扰动端点或控制点。

这些启发式让搜索更像局部随机爬山，而不是全局几何优化。

---

## 可定制扩展点

Geometrize 的核心流程留了几个扩展入口：

1. **自定义 `shapeCreator`**
   可以替代默认随机图元生成逻辑，例如只生成三角形，或根据残差热区采样初始位置。

2. **自定义 `EnergyFunction`**
   默认能量是 RMSE。调用者可以换成 L1、感知损失的近似版本、带区域权重的误差，或加入图元面积/形状先验。

3. **自定义 `ShapeAcceptancePreconditionFunction`**
   默认只接收 `newScore < lastScore`。可以改成允许极小幅变差、限制透明像素覆盖、避免与已有图元过度重叠等规则。

4. **限制 `shapeBounds`**
   可以把图元搜索限制在图像的某个百分比区域，用于分块拟合或局部修复。

---

## 对三角形拟合项目的启发

Geometrize 的优势：

- 搜索空间低：颜色由覆盖区域直接估计，随机搜索主要处理几何参数。
- 单步评估快：scanline + 增量 RMSE 避免每个候选都扫全图。
- 工程简单稳定：每步只接收降低误差的图元，结果单调改善。
- 很适合作为 diffvg 等连续优化器的初始化来源。

主要局限：

- 强贪心：一旦图元被接收，后续不会回头改它。
- 遮挡顺序固定：图元添加顺序就是渲染顺序。
- 局部搜索强依赖随机初始和变异半径，容易卡在局部最优。
- 默认候选不看残差热区，只是在允许范围内随机采样。
- 固定 alpha，且每个图元只有一个平均颜色，难以表达复杂纹理。
- 默认 RMSE 只度量像素差，不理解边缘、结构或感知质量。

如果要在本项目里借鉴它，最值得复用的不是完整 C++ 实现，而是这几个思想：

1. 用 scanline/mask 只评估受影响区域。
2. 给定几何时，用闭式近似直接估计颜色。
3. 以“随机候选 + 局部变异 + 贪心接收”生成初始三角形。
4. 再交给 diffvg 做全局连续微调，弥补 Geometrize 不会回头优化旧图元的问题。

