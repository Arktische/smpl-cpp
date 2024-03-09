#include "lbs.hpp"
#include <cmath>
#include <tuple>
#include <vector>
#include "ATen/TensorIndexing.h"
#include "ATen/ops/arange.h"
#include "ATen/ops/bmm.h"
#include "ATen/ops/cat.h"
#include "ATen/ops/clamp.h"
#include "ATen/ops/einsum.h"
#include "ATen/ops/from_blob.h"
#include "ATen/ops/index_select.h"
#include "ATen/ops/matmul.h"
#include "ATen/ops/norm.h"
#include "ATen/ops/pad.h"
#include "ATen/ops/round.h"
#include "ATen/ops/split.h"
#include "ATen/ops/unsqueeze.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "torch/types.h"

namespace smplx::lbs {
auto batch_rodrigues(Tensor &&rot_vecs, float epsilon) -> Tensor {
    auto batch_size = rot_vecs.size(0);

    auto angle = torch::norm(rot_vecs + 1e-8, 2, 1, true);

    auto rot_dir = rot_vecs / angle;

    auto cos = torch::unsqueeze(torch::cos(angle), 1);
    auto sin = torch::unsqueeze(torch::sin(angle), 1);

    auto r = torch::split(rot_dir, 1, 1);

    auto K = torch::zeros(
        {batch_size, 3, 3},
        torch::device(rot_vecs.device()).dtype(rot_vecs.dtype()));
    auto zeros = torch::zeros(
        {batch_size, 1},
        torch::device(rot_vecs.device()).dtype(rot_vecs.dtype()));

    K = torch::cat({zeros, -r[2], r[1], r[2], zeros, -r[0], -r[1], r[0], zeros},
                   1)
            .view({batch_size, 3, 3});

    auto ident =
        torch::eye(3,
                   torch::device(rot_vecs.device()).dtype(rot_vecs.dtype()))
            .unsqueeze(0);

    auto rot_mat = ident + sin * K + (1 - cos) * torch::bmm(K, K);

    return rot_mat;
}

auto batch_rigid_transform(Tensor &rot_mats, Tensor &joints, Tensor &parents,
                           torch::Dtype) -> std::tuple<Tensor, Tensor> {

    joints = torch::unsqueeze(joints, -1);

    auto rel_joints = joints.clone();
    rel_joints.index_put_(
        {Slice(None), Slice(1, None)},
        rel_joints.index({Slice(None), Slice(1, None)}) -
            joints.index({Slice(None), parents.index({Slice(1, None)})}));

    auto transforms_mat = transform_mat(rot_mats.reshape({-1, 3, 3}),
                                        rel_joints.reshape({-1, 3, 1}))
                              .reshape({-1, joints.size(1), 4, 4});

    // auto transform_chain = [transforms_mat[:, 0]]
    std::vector<Tensor> transform_chain{transforms_mat.index({Slice(None), 0})};
    transform_chain.reserve(parents.size(0));
    for (auto i = 1; i < parents.size(0); ++i) {
        auto x = parents[i];
        auto curr_res = torch::matmul(transform_chain[parents[i].item<int>()],
                                      transforms_mat.index({Slice(None), i}));
        transform_chain.emplace_back(curr_res);
    }

    auto transforms = torch::stack(transform_chain, 1);

    auto posed_joints =
        transforms.index({Slice(None), Slice(None), Slice(None, 3), 3});

    auto joints_homogen = torch::pad(joints, {0, 0, 0, 1});

    auto rel_transforms =
        transforms - torch::pad(torch::matmul(transforms, joints_homogen),
                                {3, 0, 0, 0, 0, 0, 0, 0});

    return std::make_tuple(posed_joints, rel_transforms);
}

auto lbs(Tensor &betas, Tensor &pose, Tensor &v_template, Tensor &shapedirs,
         Tensor &posedirs, Tensor &J_regressor, Tensor &parents,
         Tensor &lbs_weights, bool pose2rot) -> std::tuple<Tensor, Tensor> {
    auto batch_size = std::max(betas.size(0), pose.size(0));

    auto v_shaped = v_template + blend_shape(betas, shapedirs);

    auto J = vertices2joints(J_regressor, v_shaped);

    auto ident =
        torch::eye(3, torch::device(betas.device()).dtype(betas.dtype()));

    Tensor pose_offsets, rot_mats;
    if (pose2rot) {
        rot_mats =
            batch_rodrigues(pose.view({-1, 3})).view({batch_size, -1, 3, 3});

        auto pose_feature = (rot_mats.index({Slice(None), Slice(1, None),
                                             Slice(None), Slice(None)}) -
                             ident)
                                .view({batch_size, -1});

        pose_offsets =
            torch::matmul(pose_feature, posedirs).view({batch_size, -1, 3});
    } else {
        auto pose_feature = pose.index({Slice(None), Slice(1, None)})
                                .view({batch_size, -1, 3, 3}) -
                            ident;

        rot_mats = pose.view({batch_size, -1, 3, 3});
        pose_offsets =
            torch::matmul(pose_feature.view({batch_size, -1}), posedirs)
                .view({batch_size, -1, 3});
    }

    auto v_posed = pose_offsets + v_shaped;

    auto [J_transformed, A] = batch_rigid_transform(rot_mats, J, parents);

    auto W = lbs_weights.unsqueeze(0).expand({batch_size, -1, -1});

    auto num_joints = J_regressor.size(0);

    auto T = torch::matmul(W, A.view({batch_size, num_joints, 16}))
                 .view({batch_size, -1, 4, 4});

    auto homogen_coord =
        torch::ones({batch_size, v_posed.size(1), 1},
                    torch::device(betas.device()).dtype(betas.dtype()));

    auto v_posed_homo = torch::cat({v_posed, homogen_coord}, 2);

    auto v_homo = torch::matmul(T, torch::unsqueeze(v_posed_homo, -1));

    return std::make_tuple(
        v_homo.index({Slice(None), Slice(None), Slice(None, 3), 0}),
        J_transformed);
}

auto vertices2landmarks(Tensor &vertices, Tensor &faces, Tensor &lmk_faces_idx,
                        Tensor &lmk_bary_coords) -> Tensor {

    auto batch_size{vertices.size(0)}, num_verts{vertices.size(1)};

    auto device = vertices.device();

    auto lmk_faces = torch::index_select(faces, 0, lmk_faces_idx.view({-1}))
                         .view({batch_size, -1, 3});

    lmk_faces +=
        torch::arange(batch_size, torch::device(device).dtype(torch::kLong))
            .view({-1, 1, 1}) *
        num_verts;

    auto lmk_vertices = vertices.view({-1, 3})
                            .index_select(0, lmk_faces)
                            .view({batch_size, -1, 3, 3});

    return torch::einsum("blfi,blf->bli", {lmk_vertices, lmk_bary_coords});
}

auto find_dynamic_lmk_idx_and_bcoords(
    const Tensor &vertices, const Tensor &pose,
    const Tensor &dynamic_lmk_faces_idx, const Tensor &dynamic_lmk_b_coords,
    std::vector<int> &neck_kin_chain,
    bool pose2rot) -> std::tuple<Tensor, Tensor> {
    auto dtype = vertices.dtype();

    auto batch_size = vertices.size(0);

    Tensor rot_mats;
    if (pose2rot) {
        auto aa_pose = torch::index_select(
            pose.view({batch_size, -1, 3}), 1,
            torch::from_blob(neck_kin_chain.data(),
                             {static_cast<long long>(neck_kin_chain.size())},
                             torch::dtype(torch::kInt)));
        rot_mats =
            batch_rodrigues(aa_pose.view({-1, 3})).view({batch_size, -1, 3, 3});
    } else {
        rot_mats = torch::index_select(
            pose.view({batch_size, -1, 3, 3}), 1,
            torch::from_blob(neck_kin_chain.data(),
                             {static_cast<long long>(neck_kin_chain.size())},
                             torch::dtype(torch::kInt)));
    }

    auto rel_rot_mat =
        torch::eye(3, torch::device(vertices.device()).dtype(dtype))
            .unsqueeze_(0)
            .repeat({batch_size, 1, 1});

    auto size = neck_kin_chain.size();
    for (auto i = 0; i < size; ++i) {
        rel_rot_mat = torch::bmm(rot_mats.index({Slice(None), i}), rel_rot_mat);
    }

    auto y_rot_angle =
        torch::round(
            torch::clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / M_PI, {}, 39))
            .to(torch::dtype(torch::kLong));

    auto neg_mask = y_rot_angle.lt(0).to(torch::dtype(torch::kLong));
    auto mask = y_rot_angle.lt(-39).to(torch::dtype(torch::kLong));
    auto neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle);

    y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle);

    auto dyn_lmk_faces_idx =
        torch::index_select(dynamic_lmk_faces_idx, 0, y_rot_angle);
    auto dyn_lmk_b_coords =
        torch::index_select(dynamic_lmk_b_coords, 0, y_rot_angle);

    return std::make_tuple(dyn_lmk_faces_idx, dyn_lmk_b_coords);
}

} // namespace smplx::lbs