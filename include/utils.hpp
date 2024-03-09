#ifndef SMPLX_UTILS_HPP
#define SMPLX_UTILS_HPP
#include <filesystem>
#include <optional>
#include "common.hpp"

namespace smplx {

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

struct ModelOutput {};

struct SMPLOutput {
    std::optional<Tensor> vertices;
    std::optional<Tensor> joints;
    std::optional<Tensor> full_pose;
    std::optional<Tensor> global_orient;
    std::optional<Tensor> betas;
    std::optional<Tensor> body_pose;
    std::optional<Tensor> transl;
    std::optional<Tensor> v_shaped;
};
} // namespace smplx
#endif