/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather ACLNN Two-Stage Interface Declaration
 *
 * In-place fused {alltoall + cross-cp LSE-weighted attn update + head-AllGather
 * + permute} for CP (context parallel). The OpDef _ref inplace rename (attn_ref
 * only) makes opbuild emit NnopbaseSetRef for attn, so the inner API drops the
 * duplicate attn Output param. lse is a pure input (no SetRef, no lse output —
 * the fused op only needs lse for Phase B weighting, not as an output).
 *
 *   attn               : [T·cp, n/cp · D]  bf16   (Input & Output, same tensor)
 *   lse                : [T·cp, n/cp]      fp32   (Input only — no output)
 *   mask_num           : []  int32  (0-d; per-rank active token count)
 *   group, group_size  : HCCL group descriptor (group_size = cp ∈ {1,2,4,8,16,32})
 *
 * Active rows [0, mask_num × cp): undergo full three-phase fusion in place.
 * Inactive rows [mask_num × cp, T·cp): pass-through (kernel does not touch).
 */

#pragma once

#include "aclnn/aclnn_base.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Stage 1: Compute workspace size and validate parameters.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllAttnUpdateAllGatherGetWorkspaceSize(
    const aclTensor *attn,
    const aclTensor *lse,
    const aclTensor *mask_num,
    char *group,
    int64_t group_size,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * Stage 2: Execute.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllAttnUpdateAllGather(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif
