#ifndef SMPLX_VERTEX_IDS_HPP
#define SMPLX_VERTEX_IDS_HPP
#include "common.hpp"
namespace smplx {
const std::map<std::string, int> kSmplhVertexIds{
    {"nose", 332},     {"reye", 6260},      {"leye", 2800},
    {"rear", 4071},    {"lear", 583},       {"rthumb", 6191},
    {"rindex", 5782},  {"rmiddle", 5905},   {"rring", 6016},
    {"rpinky", 6133},  {"lthumb", 2746},    {"lindex", 2319},
    {"lmiddle", 2445}, {"lring", 2556},     {"lpinky", 2673},
    {"LBigToe", 3216}, {"LSmallToe", 3226}, {"LHeel", 3387},
    {"RBigToe", 6617}, {"RSmallToe", 6624}, {"RHeel", 6787}};

const std::map<std::string, int> kSmplxVertexIds{
    {"nose", 9120},    {"reye", 9929},      {"leye", 9448},
    {"rear", 616},     {"lear", 6},         {"rthumb", 8079},
    {"rindex", 7669},  {"rmiddle", 7794},   {"rring", 7905},
    {"rpinky", 8022},  {"lthumb", 5361},    {"lindex", 4933},
    {"lmiddle", 5058}, {"lring", 5169},     {"lpinky", 5286},
    {"LBigToe", 5770}, {"LSmallToe", 5780}, {"LHeel", 8846},
    {"RBigToe", 8463}, {"RSmallToe", 8474}, {"RHeel", 8635}};

const std::map<std::string, int> kManoVertexIds {
    {"thumb", 744}, {"index", 320}, {"middle", 443}, {"ring", 554},
        {"pinky", 671},
};
}
#endif