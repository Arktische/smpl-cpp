#ifndef SMPLX_LBS_HPP
#define SMPLX_LBS_HPP
#include "ATen/TensorIndexing.h"
#include "ATen/ops/sqrt.h"
#include "common.hpp"
#include "torch/torch.h"

namespace smplx::lbs {
inline auto vertices2joints(Tensor &J_regressor, Tensor &vertices) -> Tensor {
    return torch::einsum("bik,ji->bjk", {vertices, J_regressor});
}
inline auto blend_shape(Tensor &betas, Tensor &shape_disps) -> Tensor {
    return torch::einsum("bl,mkl->bmk", {betas, shape_disps});
}

inline auto rot_mat_to_euler(Tensor &rot_mats) -> Tensor {
    auto sy = torch::sqrt(rot_mats.index({Slice(None), 0, 0}) *
                              rot_mats.index({Slice(None), 0, 0}) +
                          rot_mats.index({Slice(None), 1, 0}) *
                              rot_mats.index({Slice(None), 1, 0}));

    return torch::atan2(-rot_mats.index({Slice(None), 2, 0}), sy);
}

inline auto transform_mat(Tensor &&R, Tensor &&t) -> Tensor {
    return torch::cat(
        {torch::pad(R, {0, 0, 0, 1}), torch::pad(t, {0, 0, 0, 1},"constant",1)},2);
}
auto batch_rodrigues(Tensor &&rot_vecs, float epsilon = 1e-8) -> Tensor;
auto batch_rigid_transform(Tensor &rot_mats, Tensor &joints, Tensor &parents,
                           torch::Dtype dtype = torch::kFloat32)
    -> std::tuple<Tensor, Tensor>;

auto lbs(Tensor &betas, Tensor &pose, Tensor &v_template, Tensor &shapedirs,
         Tensor &posedirs, Tensor &J_regressor, Tensor &parents,
         Tensor &lbs_weights, bool pose2rot) -> std::tuple<Tensor, Tensor>;

auto vertices2landmarks(Tensor &vertices, Tensor &faces, Tensor &lmk_faces_idx,
                        Tensor &lmk_bary_coords) -> Tensor;

auto find_dynamic_lmk_idx_and_bcoords(
    const Tensor &vertices, const Tensor &pose,
    const Tensor &dynamic_lmk_faces_idx, const Tensor &dynamic_lmk_b_coords,
    std::vector<int> &neck_kin_chain,
    bool pose2rot) -> std::tuple<Tensor, Tensor>;
} // namespace smplx::lbs
#endif
