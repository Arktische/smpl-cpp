#include "bodymodel.hpp"
#include <optional>
#include "common.hpp"
#include "lbs.hpp"
#include "torch/nn/module.h"
#include "utils.hpp"
#include "vertex_joint_selector.hpp"

namespace smplx {
SMPL::SMPL(const char *model_path, smpl_option &&opt)
    : vertex_joint_selector_(opt.vertex_ids_),
      joint_mapper_(opt.joint_mapper_) {
    ASSERT_MSG(std::filesystem::exists(model_path), "%s not exist", model_path);

    ASSERT_MSG(check_file_ext(model_path, "npz"), "invalid extension %s",
               std::filesystem::path(model_path).extension().string().c_str());

    data_struct_ = io::npz_load(model_path);
    shapedirs_ = torch::from_blob(data_struct_["shapedirs"].data<float>(),
                                  data_struct_["shapedirs"].shape,
                                  TensorOpt().dtype(opt.dtype_));

    auto num_betas = shapedirs_.size(-1) < SHAPE_SPACE_DIM
                         ? std::min(opt.num_betas_, 10)
                         : std::min(opt.num_betas_, SHAPE_SPACE_DIM);

    if (opt.age_ == "kid" && opt.kid_template_path_.has_value()) {
        auto x = io::npy_load(opt.kid_template_path_.value());
        auto v_template =
            torch::from_blob(data_struct_["v_template"].data<float>(),
                             data_struct_["v_template"].shape);
        auto v_template_smil = torch::from_blob(x.data<float>(), x.shape);
        v_template_smil -= torch::mean(v_template_smil, 0);
        auto v_template_diff =
            torch::unsqueeze(v_template_smil - v_template, 2);
        shapedirs_ = torch::cat({shapedirs_, v_template_diff}, 2);
        ++num_betas;
    }

    num_betas_ = num_betas;

    register_buffer("shapedirs", shapedirs_);

    faces_ = torch::from_blob(data_struct_["f"].data<float>(),
                              data_struct_["f"].shape,
                              TensorOpt().dtype(torch::kLong));

    register_buffer("faces_tensor", faces_);

    betas_ = torch::zeros({opt.batch_size_, num_betas_},
                          TensorOpt().dtype(opt.dtype_).requires_grad(true));
    if (opt.create_betas_ && opt.betas_.has_value()) {
        betas_ = std::move(opt.betas_.value());
    }

    global_orient_ =
        torch::zeros({opt.batch_size_, 3}, TensorOpt().dtype(opt.dtype_));

    if (opt.create_global_orient_ && opt.global_orient_.has_value()) {
        global_orient_ = std::move(opt.global_orient_.value());
    }

    body_pose_ = torch::zeros({opt.batch_size_, NUM_BODY_JOINTS * 3},
                              TensorOpt().dtype(opt.dtype_));

    if (opt.create_body_pose_ && opt.body_pose_.has_value()) {
        body_pose_ = std::move(opt.body_pose_.value());
    }

    transl_ = torch::zeros({opt.batch_size_, 3}, TensorOpt().dtype(opt.dtype_));
    if (opt.create_transl_ && opt.transl_.has_value()) {
        transl_ = std::move(opt.transl_.value());
    }

    v_template_ = torch::from_blob(data_struct_.at("v_template").data<float>(),
                                   data_struct_.at("v_template").shape,
                                   TensorOpt().dtype(opt.dtype_));
    if (opt.v_template_.has_value()) {
        v_template_ = std::move(opt.v_template_.value());
    }

    J_regressor_ = torch::from_blob(
        data_struct_.at("J_regressor").data<float>(),
        data_struct_.at("J_regressor").shape, TensorOpt().dtype(opt.dtype_));

    posedirs_ = torch::from_blob(data_struct_.at("posedirs").data<float>(),
                                 data_struct_.at("posedirs").shape,
                                 TensorOpt().dtype(opt.dtype_));

    auto num_pose_basis = posedirs_.size(-1);

    posedirs_ = posedirs_.reshape({-1, num_pose_basis}).t();

    lbs_weights_ = torch::from_blob(data_struct_.at("weights").data<float>(),
                                    data_struct_.at("weights").shape,
                                    TensorOpt().dtype(opt.dtype_));
    parents_ = torch::from_blob((void *)parents,
                                {sizeof(parents) / sizeof(parents[0])},
                                TensorOpt().dtype(torch::kLong));

    register_buffer("parents", parents_);
    register_buffer("lbs_weights", lbs_weights_);
    register_buffer("v_template", v_template_);
    register_buffer("J_regressor", J_regressor_);
    register_buffer("posedirs", posedirs_);
    register_parameter("betas", betas_.requires_grad_(true));
    register_parameter("global_orient", global_orient_.requires_grad_(true));
    register_parameter("body_pose", body_pose_.requires_grad_(true));
    register_parameter("transl", transl_.requires_grad_(true));
}

auto SMPL::forward(fwd_option &&opt) -> SMPLOutput {
    if (opt.global_orient_.has_value()) {
        global_orient_ = std::move(opt.global_orient_.value());
    }

    if (opt.body_pose_.has_value()) {
        body_pose_ = std::move(opt.body_pose_.value());
    }

    if (opt.betas_.has_value()) {
        betas_ = std::move(opt.betas_.value());
    }

    if (opt.transl_.has_value()) {
        transl_ = std::move(opt.transl_.value());
    }

    auto full_pose = torch::cat({global_orient_, body_pose_}, 1);

    auto batch_size =
        mmax(betas_.size(0), global_orient_.size(0), body_pose_.size(0));

    if (betas_.size(0) != batch_size) {
        betas_ = betas_.expand({int(batch_size / betas_.size(0)), -1});
    }

    auto [vertices, joints] =
        lbs::lbs(betas_, full_pose, v_template_, shapedirs_, posedirs_,
                 J_regressor_, parents_, lbs_weights_, opt.pose2rot_);

    joints = vertex_joint_selector_.forward(vertices, joints);

    if (opt.joint_mapper_.has_value() && !joint_mapper_.has_value()) {
        joints = opt.joint_mapper_.value()(joints);
    }

    if (!opt.joint_mapper_.has_value() && joint_mapper_.has_value()) {
        joints = joint_mapper_.value()(joints);
    }

    joints += transl_.unsqueeze(1);
    vertices += transl_.unsqueeze(1);

    return SMPLOutput{
        .vertices =
            opt.return_verts_ ? std::make_optional(vertices) : std::nullopt,
        .joints = joints,
        .full_pose = opt.return_full_pose_ ? std::make_optional(full_pose)
                                           : std::nullopt,
        .global_orient = global_orient_,
        .betas = betas_,
        .body_pose = body_pose_,
    };
}

} // namespace smplx
