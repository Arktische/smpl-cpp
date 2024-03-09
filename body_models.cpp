#include "body_models.hpp"
#include <memory>
#include "common.hpp"
#include "lbs.hpp"
#include "utils.hpp"
#include "vertex_joint_selector.hpp"

namespace smplx {

auto SMPL::construct(const char *model_path) -> void {
    vertex_joint_selector_ = std::make_unique<VertexJointSelector>(vars_.vertex_ids);
    ASSERT_MSG(std::filesystem::exists(model_path), "%s not exist", model_path);

    ASSERT_MSG(check_file_ext(model_path, "npz"), "invalid extension %s",
               std::filesystem::path(model_path).extension().string().c_str());

    data_struct_ = io::npz_load(model_path);

    shapedirs_ = torch::from_blob(data_struct_["shapedirs"].data<float>(),
                                  data_struct_["shapedirs"].shape,
                                  torch::dtype(vars_.dtype));

    auto num_betas = shapedirs_.size(-1) < SHAPE_SPACE_DIM
                         ? std::min(vars_.num_betas, 10)
                         : std::min(vars_.num_betas, SHAPE_SPACE_DIM);

    if (vars_.age == "kid" && vars_.kid_template_path.has_value()) {
        auto x = io::npy_load(vars_.kid_template_path.value());
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

    vars_.num_betas = num_betas;

    register_buffer("shapedirs", shapedirs_);

    faces_ =
        torch::from_blob(data_struct_["f"].data<float>(),
                         data_struct_["f"].shape, torch::dtype(torch::kLong));

    register_buffer("faces_tensor", faces_);

    if (!vars_.betas.has_value()) {
        vars_.betas.emplace(
            torch::zeros({vars_.batch_size, vars_.num_betas},
                         torch::dtype(vars_.dtype).requires_grad(true)));
    }

    if (!vars_.global_orient.has_value()) {
        vars_.global_orient.emplace(
            torch::zeros({vars_.batch_size, 3}, torch::dtype(vars_.dtype)));
    }

    if (!vars_.body_pose.has_value()) {
        vars_.body_pose.emplace(
            torch::zeros({vars_.batch_size, NUM_BODY_JOINTS * 3},
                         torch::dtype(vars_.dtype)));
    }

    if (!vars_.transl.has_value()) {
        vars_.transl.emplace(
            torch::zeros({vars_.batch_size, 3}, torch::dtype(vars_.dtype)));
    }

    v_template_ = torch::from_blob(data_struct_.at("v_template").data<float>(),
                                   data_struct_.at("v_template").shape,
                                   torch::dtype(vars_.dtype));
    if (vars_.v_template.has_value()) {
        v_template_ = std::move(vars_.v_template.value());
    }

    J_regressor_ = torch::from_blob(
        data_struct_.at("J_regressor").data<float>(),
        data_struct_.at("J_regressor").shape, torch::dtype(vars_.dtype));

    posedirs_ = torch::from_blob(data_struct_.at("posedirs").data<float>(),
                                 data_struct_.at("posedirs").shape,
                                 torch::dtype(vars_.dtype));

    auto num_pose_basis = posedirs_.size(-1);

    posedirs_ = posedirs_.reshape({-1, num_pose_basis}).t();

    lbs_weights_ = torch::from_blob(data_struct_.at("weights").data<float>(),
                                    data_struct_.at("weights").shape,
                                    torch::dtype(vars_.dtype));
    parents_ = torch::from_blob((void *)parents,
                                {sizeof(parents) / sizeof(parents[0])},
                                torch::dtype(torch::kLong));

    register_buffer("parents", parents_);
    register_buffer("lbs_weights", lbs_weights_);
    register_buffer("v_template", v_template_);
    register_buffer("J_regressor", J_regressor_);
    register_buffer("posedirs", posedirs_);
    register_parameter("betas", vars_.betas.value().requires_grad_(true));
    register_parameter("global_orient",
                       vars_.global_orient.value().requires_grad_(true));
    register_parameter("body_pose",
                       vars_.body_pose.value().requires_grad_(true));
    register_parameter("transl", vars_.transl.value().requires_grad_(true));
}

auto SMPL::forward_impl() -> SMPLOutput {
    auto full_pose =
        torch::cat({vars_.global_orient.value(), vars_.body_pose.value()}, 1);

    auto batch_size =
        mmax(vars_.betas.value().size(0), vars_.global_orient.value().size(0),
             vars_.body_pose.value().size(0));

    if (vars_.betas.value().size(0) != batch_size) {
        vars_.betas = vars_.betas.value().expand(
            {int(batch_size / vars_.betas.value().size(0)), -1});
    }

    auto [vertices, joints] = lbs::lbs(
        vars_.betas.value(), full_pose, v_template_, shapedirs_, posedirs_,
        J_regressor_, parents_, lbs_weights_, vars_.pose2rot);

    joints = vertex_joint_selector_->forward(vertices, joints);

    if (vars_.joint_mapper.has_value() && !vars_.joint_mapper.has_value()) {
        joints = vars_.joint_mapper.value()(joints);
    }

    if (!vars_.joint_mapper.has_value() && vars_.joint_mapper.has_value()) {
        joints = vars_.joint_mapper.value()(joints);
    }

    joints += vars_.transl.value().unsqueeze(1);
    vertices += vars_.transl.value().unsqueeze(1);

    return {vars_.return_verts ? std::make_optional(vertices) : std::nullopt,
            joints,
            vars_.return_full_pose ? std::make_optional(full_pose)
                                   : std::nullopt,
            vars_.global_orient,
            vars_.betas,
            vars_.body_pose};
}

} // namespace smplx
