/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather InferShape & InferDataType
 *
 * Inplace: Input("attn_ref") and Output("attn_ref") share the same name.
 * Only attn_ref is an output (lse is now a pure input — no lse output).
 * Output shape = input shape — the kernel only rearranges rows + does
 * LSE-weighted reduce + head-AllGather, so tensor shape is unchanged.
 */

#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShapeAlltoAllAttnUpdateAllGather(gert::InferShapeContext* context) {
    auto attnShape = context->GetOutputShape(0);    // attn_ref [totalT, n_per_cp · D]

    if (context->GetInputShape(0) == nullptr || attnShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Output shape matches input shape exactly (in-place, no shape change).
    *attnShape = *context->GetInputShape(0);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAlltoAllAttnUpdateAllGather(gert::InferDataTypeContext* context) {
    // attn_ref: BF16 (same as Input attn_ref)
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllAttnUpdateAllGather)
    .InferShape(InferShapeAlltoAllAttnUpdateAllGather)
    .InferDataType(InferDataTypeAlltoAllAttnUpdateAllGather);

}  // namespace ops
