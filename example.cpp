#include "body_models.hpp"
#include "common.hpp"
#include "smplx.hpp"


int main() {
    auto t = torch::rand({689,3,10});
    smplx::SMPL smpl("/path/to/npz",smplx::betas(t));

    smpl.eval();

    smpl.forward(smplx::body_pose(t),smplx::betas(t));
}