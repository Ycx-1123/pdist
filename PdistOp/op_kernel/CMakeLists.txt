# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()
# 添加这行，让 kernel 能找到 ../op_host 下的头文件


add_kernels_compile()
