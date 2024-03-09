#include "body_models.hpp"
#include "common.hpp"
#include "smplx.hpp"


int main(int argc,char*argv[]) {
    ASSERT_MSG(argc == 2, "please specify model file path");
    std::cout << argv[1] << std::endl;
    smplx::SMPL smpl(argv[1]);

    smpl.eval();

    auto x = smpl.forward();

    std::cout << x.betas.value() << "\n" << x.joints.value()<<"\n" << x.global_orient.value()<<std::endl;
    return 0;
}