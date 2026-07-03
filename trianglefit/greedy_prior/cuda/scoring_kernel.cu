#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kValues = 8;
constexpr float kPi = 3.14159265358979323846f;

__device__ __forceinline__ uint32_t rng_next(uint32_t* state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

__device__ __forceinline__ float rng_uniform(uint32_t* state) {
  return (static_cast<float>(rng_next(state) & 0x00ffffff) + 0.5f) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ uint32_t mix_seed(uint64_t seed, int candidate, int round) {
  uint64_t x = seed;
  x ^= static_cast<uint64_t>(candidate + 1) * 0x9E3779B97F4A7C15ull;
  x ^= static_cast<uint64_t>(round + 1) * 0xBF58476D1CE4E5B9ull;
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  uint32_t y = static_cast<uint32_t>(x ^ (x >> 32));
  return y == 0 ? 0x1234567u : y;
}

__device__ __forceinline__ void isosceles_vertices(
    const float cx,
    const float cy,
    const float half_base,
    const float tri_height,
    const float theta,
    float* x0,
    float* y0,
    float* x1,
    float* y1,
    float* x2,
    float* y2) {
  const float hb = fmaxf(half_base, 1.0e-6f);
  const float th = fmaxf(tri_height, 1.0e-6f);
  const float ct = cosf(theta);
  const float st = sinf(theta);
  const float apex_y = -th * 0.5f;
  const float base_y = th * 0.5f;
  *x0 = cx - st * apex_y;
  *y0 = cy + ct * apex_y;
  *x1 = cx + ct * -hb - st * base_y;
  *y1 = cy + st * -hb + ct * base_y;
  *x2 = cx + ct * hb - st * base_y;
  *y2 = cy + st * hb + ct * base_y;
}

__device__ __forceinline__ bool add_edge_intersection(
    const float px_y,
    const float ax,
    const float ay,
    const float bx,
    const float by,
    float* xs,
    int* count) {
  const float min_y = fminf(ay, by);
  const float max_y = fmaxf(ay, by);
  if (fabsf(by - ay) < 1.0e-6f || px_y < min_y || px_y >= max_y || *count >= 3) {
    return false;
  }
  const float t = (px_y - ay) / (by - ay);
  xs[*count] = ax + t * (bx - ax);
  *count += 1;
  return true;
}

__device__ void score_triangle_block(
    const float* __restrict__ target,
    const float* __restrict__ current,
    const float current_sse,
    const int h,
    const int w,
    const float cx,
    const float cy,
    const float half_base,
    const float tri_height,
    const float theta,
    float* __restrict__ shared,
    float* __restrict__ out_score,
    float* __restrict__ out_r,
    float* __restrict__ out_g,
    float* __restrict__ out_b,
    float* __restrict__ out_count) {
  const float hb = fmaxf(half_base, 1.0e-6f);
  const float th = fmaxf(tri_height, 1.0e-6f);
  const float cos_theta = cosf(theta);
  const float sin_theta = sinf(theta);
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
    *out_count = count;
    if (count <= 0.0f) {
      *out_score = __int_as_float(0x7f800000);
      *out_r = 0.0f;
      *out_g = 0.0f;
      *out_b = 0.0f;
    } else {
      const float r = fminf(fmaxf(shared[1] / count, 0.0f), 1.0f);
      const float g = fminf(fmaxf(shared[2] / count, 0.0f), 1.0f);
      const float b = fminf(fmaxf(shared[3] / count, 0.0f), 1.0f);
      const float target_sq_sum = shared[4] + shared[5] + shared[6];
      const float target_dot_color = r * shared[1] + g * shared[2] + b * shared[3];
      const float color_sq_sum = r * r + g * g + b * b;
      const float new_sse_inside = target_sq_sum - 2.0f * target_dot_color + color_sq_sum * count;
      *out_score = current_sse - shared[7] + new_sse_inside;
      *out_r = r;
      *out_g = g;
      *out_b = b;
    }
  }
  __syncthreads();
}

__device__ void score_triangle_block_scanline(
    const float* __restrict__ target,
    const float* __restrict__ current,
    const float current_sse,
    const int h,
    const int w,
    const float cx,
    const float cy,
    const float half_base,
    const float tri_height,
    const float theta,
    float* __restrict__ shared,
    float* __restrict__ out_score,
    float* __restrict__ out_r,
    float* __restrict__ out_g,
    float* __restrict__ out_b,
    float* __restrict__ out_count) {
  float vx0, vy0, vx1, vy1, vx2, vy2;
  isosceles_vertices(cx, cy, half_base, tri_height, theta, &vx0, &vy0, &vx1, &vy1, &vx2, &vy2);

  const float min_y_f = fminf(vy0, fminf(vy1, vy2));
  const float max_y_f = fmaxf(vy0, fmaxf(vy1, vy2));
  const int y_min = max(0, static_cast<int>(ceilf(min_y_f - 0.5f)));
  const int y_max = min(h - 1, static_cast<int>(floorf(max_y_f - 0.5f)));
  const int pixels = h * w;

  float local_count = 0.0f;
  float local_sum_r = 0.0f;
  float local_sum_g = 0.0f;
  float local_sum_b = 0.0f;
  float local_sq_r = 0.0f;
  float local_sq_g = 0.0f;
  float local_sq_b = 0.0f;
  float local_old_sse = 0.0f;

  if (y_min <= y_max) {
    for (int y = y_min + threadIdx.x; y <= y_max; y += blockDim.x) {
      const float py = static_cast<float>(y) + 0.5f;
      float xs[3];
      int count = 0;
      add_edge_intersection(py, vx0, vy0, vx1, vy1, xs, &count);
      add_edge_intersection(py, vx1, vy1, vx2, vy2, xs, &count);
      add_edge_intersection(py, vx2, vy2, vx0, vy0, xs, &count);
      if (count < 2) {
        continue;
      }
      float x_left = fminf(xs[0], xs[1]);
      float x_right = fmaxf(xs[0], xs[1]);
      if (count == 3) {
        x_left = fminf(x_left, xs[2]);
        x_right = fmaxf(x_right, xs[2]);
      }
      const int x_min = max(0, static_cast<int>(ceilf(x_left - 0.5f)));
      const int x_max = min(w - 1, static_cast<int>(floorf(x_right - 0.5f)));
      for (int x = x_min; x <= x_max; ++x) {
        const int index = y * w + x;
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
    }
  }

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
    *out_count = count;
    if (count <= 0.0f) {
      *out_score = __int_as_float(0x7f800000);
      *out_r = 0.0f;
      *out_g = 0.0f;
      *out_b = 0.0f;
    } else {
      const float r = fminf(fmaxf(shared[1] / count, 0.0f), 1.0f);
      const float g = fminf(fmaxf(shared[2] / count, 0.0f), 1.0f);
      const float b = fminf(fmaxf(shared[3] / count, 0.0f), 1.0f);
      const float target_sq_sum = shared[4] + shared[5] + shared[6];
      const float target_dot_color = r * shared[1] + g * shared[2] + b * shared[3];
      const float color_sq_sum = r * r + g * g + b * b;
      const float new_sse_inside = target_sq_sum - 2.0f * target_dot_color + color_sq_sum * count;
      *out_score = current_sse - shared[7] + new_sse_inside;
      *out_r = r;
      *out_g = g;
      *out_b = b;
    }
  }
  __syncthreads();
}

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

__global__ void search_triangles_kernel(
    const float* __restrict__ target,
    const float* __restrict__ current,
    const float current_sse,
    const int n,
    const int h,
    const int w,
    const float bounds_min_x,
    const float bounds_min_y,
    const float bounds_max_x,
    const float bounds_max_y,
    const float min_half_base,
    const float max_half_base,
    const float min_height,
    const float max_height,
    const float center_step_x,
    const float center_step_y,
    const float half_base_step,
    const float height_step,
    const float angle_step,
    const int mutation_count,
    const uint64_t seed,
    float* __restrict__ out_params,
    float* __restrict__ out_colors,
    float* __restrict__ out_scores) {
  const int candidate = blockIdx.x;
  if (candidate >= n) {
    return;
  }

  __shared__ float scratch[kThreads * kValues];
  __shared__ float best_cx;
  __shared__ float best_cy;
  __shared__ float best_half_base;
  __shared__ float best_height;
  __shared__ float best_theta;
  __shared__ float best_score;
  __shared__ float best_r;
  __shared__ float best_g;
  __shared__ float best_b;
  __shared__ float proposal_cx;
  __shared__ float proposal_cy;
  __shared__ float proposal_half_base;
  __shared__ float proposal_height;
  __shared__ float proposal_theta;
  __shared__ float proposal_score;
  __shared__ float proposal_r;
  __shared__ float proposal_g;
  __shared__ float proposal_b;
  __shared__ float proposal_count;

  uint32_t rng = mix_seed(seed, candidate, 0);
  if (threadIdx.x == 0) {
    best_cx = bounds_min_x + rng_uniform(&rng) * (bounds_max_x - bounds_min_x);
    best_cy = bounds_min_y + rng_uniform(&rng) * (bounds_max_y - bounds_min_y);
    best_half_base = min_half_base + rng_uniform(&rng) * (max_half_base - min_half_base);
    best_height = min_height + rng_uniform(&rng) * (max_height - min_height);
    best_theta = rng_uniform(&rng) * (2.0f * kPi);
  }
  __syncthreads();

  score_triangle_block_scanline(
      target,
      current,
      current_sse,
      h,
      w,
      best_cx,
      best_cy,
      best_half_base,
      best_height,
      best_theta,
      scratch,
      &best_score,
      &best_r,
      &best_g,
      &best_b,
      &proposal_count);

  for (int mutation = 0; mutation < mutation_count; ++mutation) {
    if (threadIdx.x == 0) {
      rng = mix_seed(seed, candidate, mutation + 1);
      proposal_cx = best_cx;
      proposal_cy = best_cy;
      proposal_half_base = best_half_base;
      proposal_height = best_height;
      proposal_theta = best_theta;
      const int choice = static_cast<int>(rng_next(&rng) % 4u);
      if (choice == 0) {
        proposal_cx += (rng_uniform(&rng) * 2.0f - 1.0f) * center_step_x;
        proposal_cy += (rng_uniform(&rng) * 2.0f - 1.0f) * center_step_y;
        proposal_cx = fminf(fmaxf(proposal_cx, bounds_min_x), bounds_max_x);
        proposal_cy = fminf(fmaxf(proposal_cy, bounds_min_y), bounds_max_y);
      } else if (choice == 1) {
        proposal_half_base += (rng_uniform(&rng) * 2.0f - 1.0f) * half_base_step;
        proposal_half_base = fminf(fmaxf(proposal_half_base, min_half_base), max_half_base);
      } else if (choice == 2) {
        proposal_height += (rng_uniform(&rng) * 2.0f - 1.0f) * height_step;
        proposal_height = fminf(fmaxf(proposal_height, min_height), max_height);
      } else {
        proposal_theta += (rng_uniform(&rng) * 2.0f - 1.0f) * angle_step;
        proposal_theta = fmodf(proposal_theta, 2.0f * kPi);
        if (proposal_theta < 0.0f) {
          proposal_theta += 2.0f * kPi;
        }
      }
    }
    __syncthreads();

    score_triangle_block_scanline(
        target,
        current,
        current_sse,
        h,
        w,
        proposal_cx,
        proposal_cy,
        proposal_half_base,
        proposal_height,
        proposal_theta,
        scratch,
        &proposal_score,
        &proposal_r,
        &proposal_g,
        &proposal_b,
        &proposal_count);

    if (threadIdx.x == 0 && proposal_score < best_score) {
      best_cx = proposal_cx;
      best_cy = proposal_cy;
      best_half_base = proposal_half_base;
      best_height = proposal_height;
      best_theta = proposal_theta;
      best_score = proposal_score;
      best_r = proposal_r;
      best_g = proposal_g;
      best_b = proposal_b;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out_params[candidate * 5 + 0] = best_cx;
    out_params[candidate * 5 + 1] = best_cy;
    out_params[candidate * 5 + 2] = best_half_base;
    out_params[candidate * 5 + 3] = best_height;
    out_params[candidate * 5 + 4] = best_theta;
    out_colors[candidate * 3 + 0] = best_r;
    out_colors[candidate * 3 + 1] = best_g;
    out_colors[candidate * 3 + 2] = best_b;
    out_scores[candidate] = best_score;
  }
}

__global__ void reduce_best_kernel(
    const float* __restrict__ scores,
    const int n,
    int* __restrict__ best_index) {
  __shared__ float shared_scores[kThreads];
  __shared__ int shared_indices[kThreads];
  float local_score = __int_as_float(0x7f800000);
  int local_index = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float score = scores[i];
    if (score < local_score) {
      local_score = score;
      local_index = i;
    }
  }
  shared_scores[threadIdx.x] = local_score;
  shared_indices[threadIdx.x] = local_index;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float rhs_score = shared_scores[threadIdx.x + stride];
      if (rhs_score < shared_scores[threadIdx.x]) {
        shared_scores[threadIdx.x] = rhs_score;
        shared_indices[threadIdx.x] = shared_indices[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    best_index[0] = shared_indices[0];
  }
}

__global__ void gather_best_kernel(
    const float* __restrict__ params,
    const float* __restrict__ colors,
    const float* __restrict__ scores,
    const int* __restrict__ best_index,
    float* __restrict__ best_params,
    float* __restrict__ best_color,
    float* __restrict__ best_score) {
  const int index = best_index[0];
  if (threadIdx.x < 5) {
    best_params[threadIdx.x] = params[index * 5 + threadIdx.x];
  }
  if (threadIdx.x < 3) {
    best_color[threadIdx.x] = colors[index * 3 + threadIdx.x];
  }
  if (threadIdx.x == 0) {
    best_score[0] = scores[index];
  }
}

__global__ void apply_triangle_kernel(
    float* __restrict__ current,
    const float* __restrict__ params,
    const float* __restrict__ color,
    const int h,
    const int w) {
  const int pixels = h * w;
  const int total = pixels * 3;
  const float cx = params[0];
  const float cy = params[1];
  const float hb = fmaxf(params[2], 1.0e-6f);
  const float th = fmaxf(params[3], 1.0e-6f);
  const float theta = params[4];
  const float cos_theta = cosf(theta);
  const float sin_theta = sinf(theta);

  for (int offset = blockIdx.x * blockDim.x + threadIdx.x; offset < total; offset += blockDim.x * gridDim.x) {
    const int channel = offset / pixels;
    const int index = offset - channel * pixels;
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
    if (inside) {
      current[offset] = color[channel];
    }
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

std::vector<at::Tensor> search_and_apply_cuda(
    at::Tensor target,
    at::Tensor current,
    double current_sse,
    int64_t candidate_count,
    int64_t mutation_count,
    double bounds_min_x,
    double bounds_min_y,
    double bounds_max_x,
    double bounds_max_y,
    double min_half_base,
    double max_half_base,
    double min_height,
    double max_height,
    double center_step_x,
    double center_step_y,
    double half_base_step,
    double height_step,
    double angle_step,
    int64_t seed,
    int64_t round_index) {
  const int n = static_cast<int>(candidate_count);
  const int h = static_cast<int>(target.size(1));
  const int w = static_cast<int>(target.size(2));
  auto options = target.options();
  at::Tensor params = at::empty({n, 5}, options);
  at::Tensor colors = at::empty({n, 3}, options);
  at::Tensor scores = at::empty({n}, options);
  at::Tensor best_index = at::empty({1}, target.options().dtype(at::kInt));
  at::Tensor best_params = at::empty({5}, options);
  at::Tensor best_color = at::empty({3}, options);
  at::Tensor best_score = at::empty({1}, options);

  const uint64_t mixed_seed = static_cast<uint64_t>(seed) ^ (static_cast<uint64_t>(round_index + 1) * 0xD2B74407B1CE6E93ull);
  search_triangles_kernel<<<n, kThreads>>>(
      target.data_ptr<float>(),
      current.data_ptr<float>(),
      static_cast<float>(current_sse),
      n,
      h,
      w,
      static_cast<float>(bounds_min_x),
      static_cast<float>(bounds_min_y),
      static_cast<float>(bounds_max_x),
      static_cast<float>(bounds_max_y),
      static_cast<float>(min_half_base),
      static_cast<float>(max_half_base),
      static_cast<float>(min_height),
      static_cast<float>(max_height),
      static_cast<float>(center_step_x),
      static_cast<float>(center_step_y),
      static_cast<float>(half_base_step),
      static_cast<float>(height_step),
      static_cast<float>(angle_step),
      static_cast<int>(mutation_count),
      mixed_seed,
      params.data_ptr<float>(),
      colors.data_ptr<float>(),
      scores.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  reduce_best_kernel<<<1, kThreads>>>(scores.data_ptr<float>(), n, best_index.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gather_best_kernel<<<1, kThreads>>>(
      params.data_ptr<float>(),
      colors.data_ptr<float>(),
      scores.data_ptr<float>(),
      best_index.data_ptr<int>(),
      best_params.data_ptr<float>(),
      best_color.data_ptr<float>(),
      best_score.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int apply_blocks = std::min(1024, std::max(1, (h * w * 3 + kThreads - 1) / kThreads));
  apply_triangle_kernel<<<apply_blocks, kThreads>>>(
      current.data_ptr<float>(),
      best_params.data_ptr<float>(),
      best_color.data_ptr<float>(),
      h,
      w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {best_params, best_color, best_score};
}
