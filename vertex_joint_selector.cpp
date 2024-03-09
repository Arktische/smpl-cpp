#include "vertex_joint_selector.hpp"
#include "common.hpp"

namespace smplx {
VertexJointSelector::VertexJointSelector(const VertexIDsT &vertex_ids,
                                         bool use_hands,
                                         bool use_feet_keypoints) {
    Tensor extra_joints_idxs;

    auto face_keyp_idxs = torch::tensor(
        {vertex_ids.at("nose"), vertex_ids.at("reye"), vertex_ids.at("leye"),
         vertex_ids.at("rear"), vertex_ids.at("lear")},
        torch::dtype(torch::kInt64));

    extra_joints_idxs = torch::cat({extra_joints_idxs, face_keyp_idxs});

    if (use_feet_keypoints) {
        auto feet_keyp_idxs =
            torch::tensor({vertex_ids.at("LBigToe"), vertex_ids.at("LSmallToe"),
                           vertex_ids.at("LHeel"), vertex_ids.at("RBigToe"),
                           vertex_ids.at("RSmallToe"), vertex_ids.at("RHeel")},
                          torch::dtype(torch::kInt32));
        extra_joints_idxs = torch::cat({extra_joints_idxs, feet_keyp_idxs});
    }

    if (use_hands) {
        std::vector<int> tip_idxs(10);
        for (auto str : tip_names) {
            tip_idxs.emplace_back(vertex_ids.at(str));
        }

        extra_joints_idxs = torch::cat(
            {extra_joints_idxs, torch::from_blob(tip_idxs.data(), {10})});
    }

    extra_joints_idxs_ =
        register_parameter("extra_joints_idxs", extra_joints_idxs);
}

auto VertexJointSelector::forward(Tensor &vertices, Tensor &joints) -> Tensor {
    auto extra_joints = torch::index_select(vertices, 1, extra_joints_idxs_);

    return torch::cat({joints, extra_joints}, 1);
}
} // namespace smplx