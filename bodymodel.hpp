#ifndef SMPLX_BODYMODEL_HPP
#define SMPLX_BODYMODEL_HPP
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include "ATen/ops/from_blob.h"
#include "ATen/ops/mean.h"
#include "ATen/ops/transpose.h"
#include "ATen/ops/unsqueeze.h"
#include "c10/core/ScalarType.h"
#include "common.hpp"
#include "npyio.hpp"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/torch.h"
#include "utils.hpp"
#include "vertex_joint_selector.hpp"

namespace smplx {
struct option {
    std::optional<Tensor> betas_ = std::nullopt;
    std::optional<Tensor> global_orient_ = std::nullopt;

    std::optional<Tensor> body_pose_ = std::nullopt;

    std::optional<Tensor> transl_ = std::nullopt;

    std::optional<Tensor> v_template_ = std::nullopt;
    
    std::optional<std::function<Tensor(Tensor &)>> joint_mapper_{std::nullopt};

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, Tensor>>>
    option &betas(T &&betas) {
        betas_.emplace(std::forward<T>(betas));
        return *this;
    }

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, Tensor>>>
    option &global_orient(T &&global_orient) {
        global_orient_.emplace(global_orient);
        return *this;
    }

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, Tensor>>>
    option &body_pose(T &&body_pose) {
        body_pose_.emplace(body_pose);
        return *this;
    }

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, Tensor>>>
    option &transl(T &&transl) {
        transl_.emplace(transl);
        return *this;
    }

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, Tensor>>>
    option &v_template(T &&v_template) {
        v_template_.emplace(v_template);
        return *this;
    }
};

struct smpl_option:public option {
    bool create_betas_ = true;
    bool create_global_orient_ = true;
    bool create_body_pose_ = true;
    bool create_transl_ = true;
    int num_betas_ = 10;
    int batch_size_ = 1;
    torch::ScalarType dtype_ = torch::kFloat32;
    std::string gender_ = "neutral";
    std::string age_ = "adult";
        option &kid_template_path();
    option &create_betas(bool x) {
        create_betas_ = x;
        return *this;
    }
    std::map<std::string, int> vertex_ids_ = kVertexIds.at("smplh");

    std::optional<std::string> kid_template_path_{std::nullopt};
};

struct fwd_option:public option {
    bool pose2rot_;
    bool return_verts_;
    bool return_full_pose_;
};

inline auto check_file_ext(const char *path, const char *extension) -> bool {
    std::string fileExt = std::filesystem::path(path).extension().string();
    std::string ext(extension);

    // trim front .
    if (!fileExt.empty() && fileExt.front() == '.') {
        fileExt.erase(fileExt.begin());
    }

    return std::equal(
        fileExt.begin(), fileExt.end(), ext.begin(), ext.end(),
        [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

class SMPL : public torch::nn::Module {
  public:
    const static int NUM_JOINTS{23};
    const static int NUM_BODY_JOINTS{23};
    const static int SHAPE_SPACE_DIM{300};
    static constexpr size_t parents[] = {0,  0,  0,  0,  1,  2,  3,  4,
                                         5,  6,  7,  8,  9,  9,  9,  12,
                                         13, 14, 16, 17, 18, 19, 20, 21};
    SMPL() = delete;

    SMPL(const char *model_path, smpl_option &&opt);

    auto num_betas() -> int { return num_betas_; }

    auto num_verts() -> int { return v_template_.size(0); }

    auto num_faces() -> int { return faces_.size(0); }

    auto forward(fwd_option&&opt) -> SMPLOutput;

  private:
    std::string gender_;
    std::string age_;
    int num_betas_;

    Tensor betas_;
    Tensor global_orient_;
    Tensor body_pose_;
    Tensor transl_;

    io::npz_t data_struct_;
    Tensor faces_;
    Tensor shapedirs_;
    Tensor v_template_;
    Tensor J_regressor_;
    Tensor posedirs_;
    Tensor lbs_weights_;
    Tensor parents_;

    VertexJointSelector vertex_joint_selector_;
    std::optional<std::function<Tensor(Tensor &)>> joint_mapper_;
};
} // namespace smplx

#endif