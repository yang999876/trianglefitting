#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kValues = 8;

__global__ void score_triangles_kernel(
    const float* __restrict__ target,
    const float* __restrict__ current,
    const float* __restrict__ centers,
    const float* __restrict__ half_base,
    const float* __restrict__ height,
    const float* __restrict__ theta,
    const float current_sse,
    const int n,
    const int h,
    const int w,
    float* __restrict__ scores,
    float* __restrict__ colors,
    float* __restrict__ counts) {
  const int candidate = blockIdx.x;
  if (candidate >= n) {
    return;
  }

  const float cx = centers[candidate * 2 + 0];
  const float cy = centers[candidate * 2 + 1];
  const float hb = fmaxf(half_base[candidate], 1.0e-6f);
  const float th = fmaxf(height[candidate], 1.0e-6f);
  const float angle = theta[candidate];
  const float cos_theta = cosf(angle);
  const float sin_theta = sinf(angle);
  const int pixels = h * w;

  float local_count = 0.0f;
  float local_sum_r = 0.0f;
  float local_sum_g = 0.0f;
  float local_sum_b = 0.0f;
  float local_sq_r = 0.0f;
  float local_sq_g = 0.0f;
  float local_sq_b = 0.0f;
  float local_old_sse = 0.0f;

  for (int index = threadIdx.x; index < pixels; index += blockDim.x) {
    const int y = index / w;
    const int x = index - y * w;
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;
    const float dx = px - cx;
    const float dy = py - cy;
    const float x_rot = cos_theta * dx + sin_theta * dy;
    const float y_rot = -sin_theta * dx + cos_theta * dy;
    const float y_from_top = y_rot + th * 0.5f;
    const float half_base_at_y = y_from_top * (hb / th);
    const bool inside =
        (y_from_top >= 0.0f) &&
        (y_rot <= th * 0.5f) &&
        (x_rot >= -half_base_at_y) &&
        (x_rot <= half_base_at_y);
    if (!inside) {
      continue;
    }

    const int offset_r = index;
    const int offset_g = pixels + index;
    const int offset_b = pixels * 2 + index;
    const float tr = target[offset_r];
    const float tg = target[offset_g];
    const float tb = target[offset_b];
    const float cr = current[offset_r];
    const float cg = current[offset_g];
    const float cb = current[offset_b];
    const float er = tr - cr;
    const float eg = tg - cg;
    const float eb = tb - cb;

    local_count += 1.0f;
    local_sum_r += tr;
    local_sum_g += tg;
    local_sum_b += tb;
    local_sq_r += tr * tr;
    local_sq_g += tg * tg;
    local_sq_b += tb * tb;
    local_old_sse += er * er + eg * eg + eb * eb;
  }

  __shared__ float shared[kThreads * kValues];
  const int base = threadIdx.x * kValues;
  shared[base + 0] = local_count;
  shared[base + 1] = local_sum_r;
  shared[base + 2] = local_sum_g;
  shared[base + 3] = local_sum_b;
  shared[base + 4] = local_sq_r;
  shared[base + 5] = local_sq_g;
  shared[base + 6] = local_sq_b;
  shared[base + 7] = local_old_sse;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int lhs = threadIdx.x * kValues;
      const int rhs = (threadIdx.x + stride) * kValues;
      #pragma unroll
      for (int i = 0; i < kValues; ++i) {
        shared[lhs + i] += shared[rhs + i];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float count = shared[0];
    counts[candidate] = count;
    if (count <= 0.0f) {
      scores[candidate] = __int_as_float(0x7f800000);
      colors[candidate * 3 + 0] = 0.0f;
      colors[candidate * 3 + 1] = 0.0f;
      colors[candidate * 3 + 2] = 0.0f;
      return;
    }

    const float r = fminf(fmaxf(shared[1] / count, 0.0f), 1.0f);
    const float g = fminf(fmaxf(shared[2] / count, 0.0f), 1.0f);
    const float b = fminf(fmaxf(shared[3] / count, 0.0f), 1.0f);
    colors[candidate * 3 + 0] = r;
    colors[candidate * 3 + 1] = g;
    colors[candidate * 3 + 2] = b;

    const float target_sq_sum = shared[4] + shared[5] + shared[6];
    const float target_dot_color = r * shared[1] + g * shared[2] + b * shared[3];
    const float color_sq_sum = r * r + g * g + b * b;
    const float new_sse_inside = target_sq_sum - 2.0f * target_dot_color + color_sq_sum * count;
    scores[candidate] = current_sse - shared[7] + new_sse_inside;
  }
}

}  // namespace

std::vector<at::Tensor> score_triangles_cuda(
    at::Tensor target,
    at::Tensor current,
    at::Tensor centers,
    at::Tensor half_base,
    at::Tensor height,
    at::Tensor theta,
    double current_sse) {
  const int n = static_cast<int>(centers.size(0));
  const int h = static_cast<int>(target.size(1));
  const int w = static_cast<int>(target.size(2));
  auto options = target.options();
  at::Tensor scores = at::empty({n}, options);
  at::Tensor colors = at::empty({n, 3}, options);
  at::Tensor counts = at::empty({n}, options);
  if (n == 0) {
    return {scores, colors, counts};
  }

  score_triangles_kernel<<<n, kThreads>>>(
      target.data_ptr<float>(),
      current.data_ptr<float>(),
      centers.data_ptr<float>(),
      half_base.data_ptr<float>(),
      height.data_ptr<float>(),
      theta.data_ptr<float>(),
      static_cast<float>(current_sse),
      n,
      h,
      w,
      scores.data_ptr<float>(),
      colors.data_ptr<float>(),
      counts.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {scores, colors, counts};
}
