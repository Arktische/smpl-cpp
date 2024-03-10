# smpl-cpp

A 100% compatiable C++ implemention for python package [smplx](https://github.com/vchoutas/smplx). Easy to use and integrate into any pytorch/libtorch workflow. Original SMPL project can be found on [this website](https://smpl-x.is.tue.mpg.de/)

# Fetch SMPL,SMPL-H,SMPL-X model files
See [here](models.md) for more instructions.
1. if you are using SMPL, you should convert `SMPL_*.pkl` to `.npz` format by using provided python script.
```bash
python pkl2npz.py /path/to/SMPL_*.pkl ...
```

# Dependencies
* `libtorch` provides tensor computation and network definition. if you prefer running on Nvidia GPU, please download CUDA version liborch
  you can choose libtorch version according to your environgment from https://pytorch.org/get-started/locally/
* `zlib` for read `.npz` file
```bash
apt install zlib1g-dev
# or you can install by conda
conda install zlib
```

* compiler supports C++17 standards
  * mingw64 on Windows not supported due to incompatiable ABI
  * vc141, vc142, vc143 are tested (correspoding to Visual Studio 2017/2019/2022)
  * gcc > 8.0 tested
  * clang > 5.0 in theory (not tested yet)



# Build
If you have installed libtorch and zlib in other paths, please modify corresponding path in commands below 

## Compile with source files
Make sure your current work directory is project root.
**CUDA version**

```bash
g++ -o a.out -std=c++17 <your_source_file_here> \
body_models.cpp joint_names.cpp lbs.cpp npyio.cpp \
vertex_ids.cpp vertex_joint_selector.cpp -Iinclude\
-Ilibtorch/include -Ilibtorch/include/torchcsrc/api/include -Izlib/include \
-Llibtorch/lib -Lzlib/lib \
-lz -lc10  -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:libtorch/lib:zlib/lib
```

**CPU version**

```bash
g++ -o a.out -std=c++17 <your_source_file_here> \
body_models.cpp joint_names.cpp lbs.cpp npyio.cpp \
vertex_ids.cpp vertex_joint_selector.cpp -Iinclude\
-Ilibtorch/include -Ilibtorch/include/torchcsrc/api/include -Izlib/include \
-Llibtorch/lib -Lzlib/lib \
-lz -lc10 -ltorch -ltorch_cpu

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:libtorch/lib:zlib/lib
```
## Static and dynamic library
Modify `CMakeLists.txt` and change the `TORCH_DIR` to `/path/to/your/libtorch/share/cmake/Torch`.
```bash
mkdir build && cd build
cmake ../
make
```

# Usage(*WIP*)
It has the same api with python package smplx.
