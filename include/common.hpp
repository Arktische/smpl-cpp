#ifndef SMPLX_COMMON_HPP
#define SMPLX_COMMON_HPP
#include <map>
#include <string>
#include <vector>
#include "torch/torch.h"
namespace smplx {
using Tensor = torch::Tensor;
using namespace torch::indexing;

extern const std::map<std::string, std::map<std::string, int>> kVertexIds;
using VertexIDsT = std::map<std::string, int>;
extern const std::vector<std::string> JOINT_NAMES;

extern const std::vector<std::string> SMPLH_JOINT_NAMES;
} // namespace smplx

#define ASSERT_MSG(cond, format, ...)                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("Assertion failure, condition=%s,msg=", #cond);             \
            printf(format, #__VA_ARGS__);                                      \
            abort();                                                           \
        }                                                                      \
    } while (0)

template <typename T> T mmax(T v) { return v; }

template <typename T, typename... Args> T mmax(T v, Args... args) {
    return std::max(v, mmax(args...));
}
#endif