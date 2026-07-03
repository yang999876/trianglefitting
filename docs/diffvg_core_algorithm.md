# diffvg 核心算法总结：用可微矢量图拟合图片

## 源码定位

`third-party/diffvg` 是一个可微 2D 矢量图光栅化器。它本身不是像 Geometrize 那样的贪心图元搜索器，而是提供“给定一组矢量图元，渲染成像素图，并把像素损失反传回图元参数”的能力。用它拟合图片时，通常由 Python/PyTorch 负责初始化图元、定义 loss 和优化循环。

核心文件：

- `pydiffvg/shape.py`：Python 侧的图元数据结构。
- `pydiffvg/render_pytorch.py`：PyTorch autograd 封装，连接 Python tensor 与 C++/CUDA renderer。
- `diffvg.cpp`：前向渲染、反向传播、边界采样、颜色合成等核心实现。
- `scene.*`：构建场景、BVH、边界采样分布和梯度缓存。
- `shape.h` / `color.h` / `filter.h`：C++ 侧图元、颜色、滤波器数据结构。
- `apps/painterly_rendering.py`：从随机 path/blob 拟合目标图片的主要示例。
- `apps/refine_svg.py`：从已有 SVG 出发，优化路径点和颜色以贴近目标图。

一句话概括：diffvg 把矢量图元参数变成 PyTorch 可学习变量，通过可微渲染得到图片，再用 MSE、LPIPS 等损失反向传播，用 Adam 优化图元位置、控制点、线宽、颜色和变换。

---

## 和 Geometrize 的根本区别

Geometrize 的流程是：

```text
随机候选 -> 局部爬山 -> 选一个误差下降最大的图元 -> 永久添加
```

diffvg 的流程是：

```text
初始化一批图元 -> 渲染 -> 计算 loss -> 反传梯度 -> 同时更新全部可学习参数
```

因此 diffvg 不负责“自动决定下一步添加哪个图元”。图元数量、类型、初始位置和渲染顺序通常由外部代码给定；diffvg 负责让这些连续参数能够被梯度下降优化。

---

## 支持的基本图元与场景结构

Python 侧图元定义在 `pydiffvg/shape.py`：

| Python 类 | C++ 类型 | 主要可优化参数 |
|---|---|---|
| `Circle` | `ShapeType::Circle` | `radius`, `center`, `stroke_width` |
| `Ellipse` | `ShapeType::Ellipse` | `radius`, `center`, `stroke_width` |
| `Rect` | `ShapeType::Rect` | `p_min`, `p_max`, `stroke_width` |
| `Path` | `ShapeType::Path` | Bezier/line 控制点、可选逐点 thickness、`stroke_width` |
| `Polygon` | 内部按 path 处理 | 顶点坐标、闭合规则、`stroke_width` |

颜色不直接挂在 shape 上，而是通过 `ShapeGroup` 绑定：

- `shape_ids`：这个 group 包含哪些 shape。
- `fill_color`：填充颜色，可为常量、线性渐变或径向渐变。
- `stroke_color`：描边颜色，也支持常量和渐变。
- `shape_to_canvas`：shape 到画布的 3x3 变换矩阵。
- `use_even_odd_rule`：填充规则。

渲染顺序由 `ShapeGroup` 的顺序决定。`diffvg.cpp` 中采样像素时会把命中的 fragments 按 group id 从后到前排序并 alpha blending。

---

## 拟合图片的典型流程

以 `apps/painterly_rendering.py` 为代表：

1. 读取目标图片，归一化到 `[0, 1]`。
2. 随机生成很多 path：
   - 普通模式：开放 Bezier 曲线，优化 stroke。
   - `--use_blob`：闭合 Bezier blob，优化 fill。
3. 将图元参数设为 PyTorch 可学习变量：
   - `path.points.requires_grad = True`
   - `path.stroke_width.requires_grad = True`
   - `group.fill_color/stroke_color.requires_grad = True`
4. 每轮重新 `serialize_scene`，调用 `RenderFunction.apply` 渲染。
5. 将 RGBA 结果与白底合成，得到 RGB 图。
6. 计算 loss：
   - 默认：`(img - target).pow(2).mean()`
   - 可选：`LPIPS(img, target) + mean color penalty`
7. `loss.backward()` 触发 diffvg 的反向传播。
8. Adam 更新点坐标、线宽和颜色。
9. 对颜色、线宽等参数做 clamp。

伪代码：

```text
shapes, shape_groups = 随机初始化或从 SVG 解析
params = 图元点坐标 + 颜色 + 线宽 + 变换
optim = Adam(params)

for iter in range(num_iter):
    scene_args = serialize_scene(shapes, shape_groups)
    rgba = render(width, height, samples_x, samples_y, seed, background, *scene_args)
    rgb = alpha_composite(rgba, white_background)
    loss = image_loss(rgb, target)
    loss.backward()
    optim.step()
    clamp(params)
```

`apps/refine_svg.py` 的区别是从已有 SVG 解析出 path 和颜色，再优化 path control points 与 fill color；这适合“已有矢量初稿 -> 对齐目标图片”的场景。

---

## PyTorch 封装：serialize_scene / forward / backward

`pydiffvg/render_pytorch.py` 中的 `RenderFunction` 继承 `torch.autograd.Function`。

### serialize_scene

`serialize_scene` 把 Python 对象线性展开成 PyTorch autograd 能接收的参数列表：

- canvas 宽高、图元数量、group 数量。
- 每个 shape 的类型和 tensor 参数。
- 每个 group 的 shape ids、fill/stroke 颜色、填充规则、变换矩阵。
- filter 类型和 radius。

这个步骤很重要：PyTorch autograd 只会追踪传入 `RenderFunction.apply` 的 tensor，所以可优化参数必须在这里作为参数传进去。

### forward

forward 会：

1. 从序列化参数恢复 C++ `Circle` / `Ellipse` / `Path` / `Rect` / `ShapeGroup`。
2. 构造 `diffvg.Scene`。
3. 调用 C++ `diffvg.render(...)`，输出 `H x W x 4` 的 RGBA tensor。
4. 把 scene 和参数信息保存到 `ctx`，供 backward 使用。

### backward

backward 接收上游传来的 `grad_img`，再次调用 `diffvg.render(...)`，但这次传入的是：

- `render_image = nullptr`
- `d_render_image = grad_img`

C++ 渲染器会把像素梯度累积到 `scene.d_shapes`、`scene.d_shape_groups`、`scene.d_filter` 中。随后 Python wrapper 把这些 C++ 梯度拷回 PyTorch tensor，作为 `RenderFunction` 各输入参数的梯度返回。

这就是 diffvg 能接入 PyTorch 优化器的关键。

---

## 前向渲染：采样、命中、排序与合成

核心前向逻辑在 `diffvg.cpp` 的 `render_kernel` 和 `sample_color`。

对每个像素，diffvg 会采样 `num_samples_x * num_samples_y` 个子像素点。示例中常用 `2 x 2` samples。

每个采样点：

1. 转到归一化 screen space，再映射到 canvas space。
2. 通过场景 BVH 找出可能命中的 `ShapeGroup`。
3. 对 stroke，用 `within_distance` 判断采样点是否在描边宽度范围内。
4. 对 fill，用 winding number 判断采样点是否在闭合图形内部。
5. 收集所有命中的 fragment。
6. 按 group id 从后到前排序。
7. 做 alpha blending：

```text
accum_color = prev_color * (1 - alpha) + alpha * new_color
accum_alpha = prev_alpha * (1 - alpha) + alpha
final_rgb = accum_color / accum_alpha
```

采样结果会经过 pixel filter splat 到输出像素。filter 支持：

- Box
- Tent
- RadialParabolic
- Hann

`weight_kernel` 会先统计每个像素的 filter 权重和，`render_kernel` 再按权重归一化累积颜色。

---

## 反向传播：颜色梯度与几何梯度

diffvg 的难点在几何参数梯度。颜色、alpha、渐变 stop 等参数相对直接：它们出现在颜色采样和 alpha blending 公式里，可以按链式法则反传。

形状位置、半径、控制点等参数更麻烦，因为像素覆盖关系是离散的：一个边界稍微移动，某些采样点会从外部变成内部。diffvg 主要使用两条路线处理。

### 路线 1：边界采样梯度

默认非 prefiltering 模式下，backward 结束前会额外执行 boundary sampling：

1. 根据各 shape 的边界长度构造采样 CDF。
2. 在图元边界上随机采样点。
3. 在边界法线两侧各采样一次颜色，估计边界两侧颜色差。
4. 结合上游像素梯度，得到该边界移动对 loss 的贡献。
5. 根据 Reynolds transport theorem，把边界移动速度投影到 shape 参数上。

代码里的 `accumulate_boundary_gradient` 会根据图元类型把贡献分配到不同参数：

- Circle：center 和 radius。
- Ellipse：center 和两个 radius。
- Rect：`p_min` / `p_max`。
- Path：Bezier 控制点或线段端点。
- Stroke width / path thickness。
- `shape_to_canvas` 变换矩阵。

这部分是 diffvg 能优化形状几何的核心。

### 路线 2：prefiltering / 距离软覆盖

当 `use_prefiltering=True` 时，`sample_color_prefiltered` 会用到点到图元边界的距离 `d` 和 `smoothstep`：

- fill：根据 signed distance 得到软覆盖权重。
- stroke：根据到曲线的距离和 stroke width 得到软覆盖权重。

这样像素颜色对边界距离变成连续近似，可以通过 `d_compute_distance` 把梯度传回几何参数。这个路线更像软光栅化，减少离散覆盖带来的不可导问题，但会改变边缘模型。

---

## 距离、内部判断与加速结构

内部/描边判断用到几个关键模块：

- `winding_number.h`：判断 fill 是否覆盖某点。
- `within_distance.h`：判断 stroke 是否覆盖某点。
- `compute_distance.h`：计算点到 shape 的最近距离和最近点，并实现距离对 shape 参数的反向传播。
- `sample_boundary.h`：在 circle、ellipse、rect、path 边界上采样点，并返回法线、PDF 和 path 局部信息。
- `scene.cpp`：构建 BVH 和边界长度 CDF。

`Scene` 会为 shape group 和 path 构建 BVH，加速“某个采样点可能命中哪些图元”的查询；也会为边界采样准备按长度归一化的 CDF/PMF。

---

## 可优化参数与常见约束

diffvg 能反传的参数包括：

- circle / ellipse 的中心和半径。
- rect 的两个角点。
- path / polygon 的控制点。
- path stroke width 或逐点 thickness。
- constant fill/stroke RGBA。
- linear/radial gradient 的端点、中心、半径、stop offset 和 stop color。
- shape group 的 `shape_to_canvas` 变换矩阵。
- pixel filter radius。
- 可选背景图像。

示例里经常在 optimizer step 后手动约束参数：

- 颜色 clamp 到 `[0, 1]`。
- stroke width clamp 到 `[1.0, max_width]`。
- 某些示例把归一化坐标乘回画布尺寸，以便 learning rate 更稳定。

这些约束不是 diffvg 自动做的，需要外部优化代码负责。

---

## 用基本图元拟合图片时的实际算法形态

如果目标是用三角形、矩形、曲线等基本图元拟合图片，diffvg 的“核心算法”可以理解为：

```text
1. 选定图元数量 N、类型和渲染顺序
2. 初始化图元几何参数和颜色
3. 将所有参数设为 requires_grad
4. 用 diffvg 前向渲染得到当前图像
5. 用像素 MSE / L1 / LPIPS / 其他 loss 对比目标图
6. 通过 diffvg backward 得到几何和颜色梯度
7. 用 Adam/SGD 更新所有参数
8. clamp 或正则化参数
9. 重复直到收敛
```

这是一种连续优化。它可以同时调整所有图元，但它不会自动改变：

- 图元数量。
- 图元拓扑类型。
- Path 的段数。
- shape group 的遮挡顺序。
- 已经被遮挡到没有梯度的图元。

所以 diffvg 通常需要一个好的初始化。Geometrize、随机多起点、残差热区采样、神经网络初始化都可以作为 diffvg 的前端。

---

## 优势与局限

优势：

- 可以直接接入 PyTorch，和任意可微 loss 组合。
- 能同时优化几何、颜色、线宽、渐变和变换。
- 比 Geometrize 的单步贪心更适合全局联合微调。
- 支持 SVG/path/Bezier，比单纯三角形或圆更通用。
- C++/CUDA 并行渲染，适合大量迭代。

局限：

- 只处理连续参数优化，不能直接解决图元数量、拓扑、排序等离散决策。
- 对初始化敏感；坏初始化容易陷入局部最优。
- 遮挡会让底层图元梯度变弱甚至消失。
- 几何梯度来自边界附近，对大范围结构重排能力有限。
- 参数需要外部 clamp/正则化，否则可能出现颜色越界、线宽异常或图元漂移。
- 大量图元和高采样率会显著增加显存/时间开销。

---

## 对本项目的启发

diffvg 最适合放在拟合流程的后半段：

1. 前端用 Geometrize、随机搜索或神经网络给出图元初始布局。
2. diffvg 接管连续参数：
   - 顶点坐标
   - 颜色/alpha
   - 线宽
   - 可选变换
3. 用 MSE/L1/LPIPS 等目标函数联合微调。
4. 周期性结合非梯度方法处理离散问题：
   - 删除低贡献图元。
   - 重新初始化被遮挡或无效的图元。
   - 优化渲染顺序。
   - 在残差热区补充新图元。

一句实用判断：Geometrize 擅长“找图元放哪里”，diffvg 擅长“把已经放下的图元连续调到更准”。两者组合，比单独使用任意一边更符合基本图元拟合图片的工程需求。

