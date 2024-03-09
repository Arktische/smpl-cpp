cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(smplx)
set(TORCH_DIR /path/to/your/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(PROJECT_SOURCES body_models.cpp joint_names.cpp lbs.cpp npyio.cpp vertex_ids.cpp vertex_joint_selector.cpp)

add_library(smplx SHARED ${PROJECT_SOURCES})
add_library(smplx_static STATIC ${PROJECT_SOURCES})
target_link_libraries(smplx "${TORCH_LIBRARIES}" ZLIB::ZLIB)
target_link_libraries(smplx_static "${TORCH_LIBRARIES}" ZLIB::ZLIB)
set_property(TARGET smplx PROPERTY CXX_STANDARD 17)
set_property(TARGET smplx_staic PROPERTY CXX_STANDARD 17)
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
#if (MSVC)
#  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#  add_custom_command(TARGET example-app
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:example-app>)
#endif (MSVC)