/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather OpDef Registration
 *
 * Fused {AlltoAll + LSE-weighted attention update + head AllGather}, in-place on
 * attn along the CP (context parallel) communication group.
 *
 *   attn_ref [totalT, n_per_cp · D] bf16   (Input & Output, same GM address)
 *   lse     [totalT, n_per_cp]     fp32   (Input only — read for Phase B weighting,
 *                                         no output: downstream does not consume it)
 *   mask_num []                     int32  (mask_per_rank; b0_total = mask_per_rank · cp_size)
 *
 * Active rows [0, b0_total): Phase A peermem-pack → Phase B LSE-weighted reduce
 * → Phase C peermem head-AllGather (reverse stride) back to attn.
 * Inactive rows [b0_total, totalT): inplace pass-through (kernel does not touch;
 * opbuild SetRef guarantees input/output share GM address).
 *
 * Inplace contract: Input("attn_ref") + Output("attn_ref") same-name + _ref
 * suffix → opbuild emits NnopbaseSetRef. lse has no Output (pure input, no SetRef).
 */

#include "register/op_def_registry.h"

namespace ops {

class AlltoAllAttnUpdateAllGather : public OpDef {
public:
    explicit AlltoAllAttnUpdateAllGather(const char *name) : OpDef(name) {
        // ===== Inputs =====
        // attn_ref: 同名 Input + Output + _ref 后缀 = inplace SetRef gate
        // lse:      纯 input (无 Output, 无 SetRef) — 仅 Phase B 加权读取
        this->Input("attn_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ===== Outputs (inplace: attn_ref same name as Input, _ref suffix) =====
        this->Output("attn_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ===== Attributes =====
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("group_size").AttrType(REQUIRED).Int();

        // ===== AICore Configuration =====
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        // ===== MC2 Communication Domain Declaration =====
        this->MC2().HcclGroup("group");
    }
};

OP_ADD(AlltoAllAttnUpdateAllGather);

}  // namespace ops
