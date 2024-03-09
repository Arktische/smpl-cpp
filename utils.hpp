#ifndef SMPLX_UTILS_HPP
#define SMPLX_UTILS_HPP
#include <optional>
#include "torch/torch.h"

namespace smplx {
struct ModelOutput {};

struct SMPLOutput {
    std::optional<torch::Tensor> vertices;
    std::optional<torch::Tensor> joints;
    std::optional<torch::Tensor> full_pose;
    std::optional<torch::Tensor> global_orient;
    std::optional<torch::Tensor> betas;
    std::optional<torch::Tensor> body_pose;
    std::optional<torch::Tensor> transl;
    std::optional<torch::Tensor> v_shaped;
};
} // namespace smplx
#endif