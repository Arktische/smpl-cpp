#ifndef SMPLX_BODYMODEL_HPP
#define SMPLX_BODYMODEL_HPP
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include "c10/core/ScalarType.h"
#include "common.hpp"
#include "npyio.hpp"
#include "utils.hpp"
#include "vertex_joint_selector.hpp"

namespace smplx {

namespace internal {
struct option {
    std::optional<Tensor> betas;
    std::optional<Tensor> global_orient{std::nullopt};

    std::optional<Tensor> body_pose{std::nullopt};

    std::optional<Tensor> transl{std::nullopt};

    std::optional<Tensor> v_template{std::nullopt};

    std::optional<std::function<Tensor(Tensor &)>> joint_mapper{std::nullopt};

    int num_betas = 10;
    int batch_size = 1;
    torch::ScalarType dtype = torch::kFloat32;
    std::string gender = "neutral";
    std::string age = "adult";
    const std::map<std::string, int>& vertex_ids = kSmplhVertexIds;

    std::optional<std::string> kid_template_path{std::nullopt};

    bool pose2rot;
    bool return_verts;
    bool return_full_pose;
};
} // namespace internal

// template<typename T>
// auto gender(T&& gender ) {
//     return [&gender](internal::option& opt) {
//         opt.gender
//     };
// }


template <typename T> auto transl(T &&transl) {
    return [&transl](internal::option &opt) { opt.transl.emplace(transl); };
}

template <typename T> auto body_pose(T &&body_pose) {
    return [&body_pose](internal::option &opt) {
        opt.body_pose.emplace(std::forward<T>(body_pose));
    };
}

template <typename T> auto v_template(T &&v_template) {
    return [&v_template](internal::option &opt) {
        opt.v_template.emplace(std::forward<T>(v_template));
    };
}

template <typename T> auto global_orient(T &&global_orient) {
    return [&global_orient](internal::option &opt) {
        opt.global_orient.emplace(std::forward<T>(global_orient));
    };
}

template <typename T> auto betas(T &&betas) {
    return [&betas](internal::option &opt) {
        opt.betas.emplace(std::forward<T>(betas));
    };
}

inline auto batch_size(int batch_size) {
    return [batch_size](internal::option &opt) {
        opt.batch_size = batch_size;
    };
}



using option = std::function<void(internal::option &)>;

template <typename T, typename... Args>
inline void apply_option(internal::option &opt, T&& f, Args&&... args) {
    f(opt);
    if constexpr (sizeof...(Args) > 0) {
        apply_option(opt, args...);
    }
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

    template <typename... Args> SMPL(const char *model_path, Args &&...args) {
        if constexpr (sizeof...(Args) > 0) {
            apply_option(vars_, args...);
        }
        construct(model_path);
    }

    auto construct(const char *model) -> void;

    auto num_betas() -> int { return vars_.num_betas; }

    auto num_verts() -> int { return v_template_.size(0); }

    auto num_faces() -> int { return faces_.size(0); }

    template <typename... Args> auto forward(Args &&...args) -> SMPLOutput {
        if constexpr (sizeof...(Args) > 0) {
            apply_option(vars_, args...);
        }
        return forward_impl();
    }

    auto forward_impl() -> SMPLOutput;

  private:
    internal::option vars_;

    io::npz_t data_struct_;
    Tensor faces_;
    Tensor shapedirs_;
    Tensor v_template_;
    Tensor J_regressor_;
    Tensor posedirs_;
    Tensor lbs_weights_;
    Tensor parents_;

    std::unique_ptr<VertexJointSelector> vertex_joint_selector_;
};
} // namespace smplx

#endif