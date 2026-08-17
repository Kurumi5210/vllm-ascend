# AlltoAllAttnUpdateAllGather op_host CMake
# dev/vllm-ascend csrc 风格 (对齐邻居 dispatch_ffn_combine/CMakeLists.txt 的
# target_sources 自动发现模式; op_add_subdirectory 在 csrc/CMakeLists.txt 中
# 已 glob **/op_host/CMakeLists.txt 自动纳入).

set(_ALLTOALL_ATTN_INC_OPTS)
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    list(APPEND _ALLTOALL_ATTN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    list(APPEND _ALLTOALL_ATTN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    list(APPEND _ALLTOALL_ATTN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
endif()

# --cce-auto-sync=off: MC2 peermem 算子 kernel 自管 SetFlag/WaitFlag 同步
#   (与 newest 仓 mask_all_to_all_v2 同款; 区别于 dispatch_ffn_combine=on,
#    后者走 HCCL 集体通信不自管同步). 不加 -DHCCL_COMM (kernel 不依赖).
# -DAICORE_EXCEPTION_RESTART: 仅 910_93 (A3), AI Core 硬件异常自动重启.
add_ops_compile_options(
        OP_NAME AlltoAllAttnUpdateAllGather
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                ${_ALLTOALL_ATTN_INC_OPTS}
)

if ("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend910_93")
    add_ops_compile_options(
            OP_NAME AlltoAllAttnUpdateAllGather
            OPTIONS -DAICORE_EXCEPTION_RESTART
    )
endif()

target_sources(op_host_aclnnInner PRIVATE
        allto_all_attn_update_all_gather_def.cpp
)

target_sources(opapi PRIVATE
        aclnn_allto_all_attn_update_all_gather.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
            aclnn_allto_all_attn_update_all_gather.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
            aclnn_allto_all_attn_update_all_gather.cpp
    )
endif ()

target_sources(optiling PRIVATE
        allto_all_attn_update_all_gather_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_sources(opsproto PRIVATE
        allto_all_attn_update_all_gather_proto.cpp
)

file(GLOB _ALLTOALL_ATTN_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_allto_all_attn_update_all_gather.h")

install(FILES ${_ALLTOALL_ATTN_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
