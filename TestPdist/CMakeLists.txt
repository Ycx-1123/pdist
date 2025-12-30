cmake_minimum_required(VERSION 3.16)
project(pdist_verify)

set(ASCEND_HOME /usr/local/Ascend/ascend-toolkit/latest)

include_directories(
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/include/aclnn
    ${ASCEND_HOME}/opp/vendors/customize/op_api/include
)

link_directories(
    ${ASCEND_HOME}/lib64
    ${ASCEND_HOME}/opp/vendors/customize/op_api/lib
)

add_executable(main main.cpp)

target_link_libraries(main 
    ascendcl 
    nnopbase 
    cust_opapi 
    pthread
)