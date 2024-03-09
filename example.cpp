#include "body_models.hpp"
#include "common.hpp"
#include "smplx.hpp"


int main(int argc,char*argv[]) {
    // auto t = torch::rand({689,3,10});

    std::cout << argv[1] << std::endl;
    smplx::SMPL smpl(argv[1]);

    smpl.eval();

    smpl.forward();
}