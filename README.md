# smplx-cpp

A 100% compatiable C++ implemention for python package smplx. Easy to use and integrate into any pytorch/libtorch workflow.

# dependencies
* `libtorch` provides tensor computation and network definition. if you prefer running on Nvidia GPU, please download CUDA version liborch
  you can choose libtorch version according to your environgment from https://pytorch.org/get-started/locally/
* `zlib` for read `.npz` file
```bash
apt install zlib1g-dev
# or you can install by conda
conda install zlib
```

* compiler supports C++17 standards.



# build
`smplx-cpp` is easy to build and integrate, just add zlib and libtorch include directories and link libraries. 

if you have installed libtorch and zlib in other paths. plz modify corresponding path in commands below 

**CUDA version**

```bash
g++ -o a.out -std=c++17 <your_source_file_here> bodymodel.cpp joint_names.cpp lbs.cpp npyio.cpp vertex_ids.cpp \
-Ilibtorch/include -Ilibtorch/include/torchcsrc/api/include -Izlib/include \
-Llibtorch/lib -Lzlib/lib \
-lz -lc10  -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda

export LD_LIBRARY_PARH=$LD_LIBRARY_PATH:libtorch/lib:zlib/lib
```

**CPU version**

```bash
g++ -o a.out -std=c++17 <your_source_file_here> bodymodel.cpp joint_names.cpp lbs.cpp npyio.cpp vertex_ids.cpp \
-Ilibtorch/include -Ilibtorch/include/torchcsrc/api/include -Izlib/include \
-Llibtorch/lib -Lzlib/lib \
-lz -lc10 -ltorch -ltorch_cpu

export LD_LIBRARY_PARH=$LD_LIBRARY_PATH:libtorch/lib:zlib/lib
```

