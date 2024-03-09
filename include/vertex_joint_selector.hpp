#ifndef SMPLX_VERTEX_JOINT_SELECTOR_HPP
#define SMPLX_VERTEX_JOINT_SELECTOR_HPP
#include "ATen/core/TensorBody.h"
#include "ATen/ops/cat.h"
#include "ATen/ops/from_blob.h"
#include "ATen/ops/index_select.h"
#include "ATen/ops/tensor.h"
#include "common.hpp"
#include "torch/nn/module.h"
#include "torch/types.h"

namespace smplx {
class VertexJointSelector : public torch::nn::Module {
  private:
    constexpr static char tip_names[10][8]{
        "lthumb", "lindex", "lmiddle", "lring", "lpinky",
        "rthumb", "rindex", "rmiddle", "rring", "rpinky"};

    Tensor extra_joints_idxs_;

  public:
    VertexJointSelector() = default;
    VertexJointSelector(const VertexIDsT &vertex_ids, bool use_hands = true,
                        bool use_feet_keypoints = true);

    auto forward(Tensor &vertices, Tensor &joints) -> Tensor;
};
} // namespace smplx
#endif