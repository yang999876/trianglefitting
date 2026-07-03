#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> score_triangles_cuda(
    torch::Tensor target,
    torch::Tensor current,
    torch::Tensor centers,
    torch::Tensor half_base,
    torch::Tensor height,
    torch::Tensor theta,
    double current_sse);

std::vector<torch::Tensor> search_and_apply_cuda(
    torch::Tensor target,
    torch::Tensor current,
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
    int64_t round_index);

std::vector<torch::Tensor> score_triangles(
    torch::Tensor target,
    torch::Tensor current,
    torch::Tensor centers,
    torch::Tensor half_base,
    torch::Tensor height,
    torch::Tensor theta,
    double current_sse) {
  TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
  TORCH_CHECK(current.is_cuda(), "current must be a CUDA tensor");
  TORCH_CHECK(centers.is_cuda(), "centers must be a CUDA tensor");
  TORCH_CHECK(half_base.is_cuda(), "half_base must be a CUDA tensor");
  TORCH_CHECK(height.is_cuda(), "height must be a CUDA tensor");
  TORCH_CHECK(theta.is_cuda(), "theta must be a CUDA tensor");
  TORCH_CHECK(target.scalar_type() == torch::kFloat32, "target must be float32");
  TORCH_CHECK(current.scalar_type() == torch::kFloat32, "current must be float32");
  TORCH_CHECK(centers.scalar_type() == torch::kFloat32, "centers must be float32");
  TORCH_CHECK(half_base.scalar_type() == torch::kFloat32, "half_base must be float32");
  TORCH_CHECK(height.scalar_type() == torch::kFloat32, "height must be float32");
  TORCH_CHECK(theta.scalar_type() == torch::kFloat32, "theta must be float32");
  TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
  TORCH_CHECK(current.is_contiguous(), "current must be contiguous");
  TORCH_CHECK(centers.is_contiguous(), "centers must be contiguous");
  TORCH_CHECK(half_base.is_contiguous(), "half_base must be contiguous");
  TORCH_CHECK(height.is_contiguous(), "height must be contiguous");
  TORCH_CHECK(theta.is_contiguous(), "theta must be contiguous");
  TORCH_CHECK(target.dim() == 3 && target.size(0) == 3, "target must have shape [3, H, W]");
  TORCH_CHECK(current.sizes() == target.sizes(), "current must match target shape");
  TORCH_CHECK(centers.dim() == 2 && centers.size(1) == 2, "centers must have shape [N, 2]");
  TORCH_CHECK(half_base.dim() == 2 && half_base.size(1) == 1, "half_base must have shape [N, 1]");
  TORCH_CHECK(height.dim() == 2 && height.size(1) == 1, "height must have shape [N, 1]");
  TORCH_CHECK(theta.dim() == 2 && theta.size(1) == 1, "theta must have shape [N, 1]");
  TORCH_CHECK(half_base.size(0) == centers.size(0), "half_base count must match centers");
  TORCH_CHECK(height.size(0) == centers.size(0), "height count must match centers");
  TORCH_CHECK(theta.size(0) == centers.size(0), "theta count must match centers");
  return score_triangles_cuda(target, current, centers, half_base, height, theta, current_sse);
}

std::vector<torch::Tensor> search_and_apply(
    torch::Tensor target,
    torch::Tensor current,
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
  TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
  TORCH_CHECK(current.is_cuda(), "current must be a CUDA tensor");
  TORCH_CHECK(target.scalar_type() == torch::kFloat32, "target must be float32");
  TORCH_CHECK(current.scalar_type() == torch::kFloat32, "current must be float32");
  TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
  TORCH_CHECK(current.is_contiguous(), "current must be contiguous");
  TORCH_CHECK(target.dim() == 3 && target.size(0) == 3, "target must have shape [3, H, W]");
  TORCH_CHECK(current.sizes() == target.sizes(), "current must match target shape");
  TORCH_CHECK(candidate_count > 0, "candidate_count must be positive");
  TORCH_CHECK(mutation_count >= 0, "mutation_count must be non-negative");
  return search_and_apply_cuda(
      target,
      current,
      current_sse,
      candidate_count,
      mutation_count,
      bounds_min_x,
      bounds_min_y,
      bounds_max_x,
      bounds_max_y,
      min_half_base,
      max_half_base,
      min_height,
      max_height,
      center_step_x,
      center_step_y,
      half_base_step,
      height_step,
      angle_step,
      seed,
      round_index);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("score_triangles", &score_triangles, "Score opaque isosceles triangle candidates (CUDA)");
  m.def("search_and_apply", &search_and_apply, "Search, select, and apply the best opaque isosceles triangle (CUDA)");
}
