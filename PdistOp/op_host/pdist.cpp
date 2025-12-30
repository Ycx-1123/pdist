/**
 * @file pdist.cpp
 * @brief Host-side tiling implementation for Pdist operator (Cyclic Tiling Optimized)
 */

#include "pdist_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// 辅助函数：计算 Ceil(a, b)
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    PdistTilingData tiling;
    
    // 1. 获取输入参数
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const float* p_ptr = attrs->GetAttrPointer<float>(0);
    float p = (p_ptr != nullptr) ? *p_ptr : 2.0f;

    const gert::StorageShape* x_shape = context->GetInputShape(0);
    uint32_t n = x_shape->GetStorageShape().GetDim(0);
    uint32_t m = x_shape->GetStorageShape().GetDim(1);

    // 2. 获取平台信息
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    
    // 3. 计算对齐后的 m (tileLength)
    // 硬件要求 DataCopy 地址 32 字节对齐
    // FP16: 32 bytes = 16 elements
    // FP32: 32 bytes = 8 elements
    // 为了稳妥，统一按 32 字节对齐向上取整
    uint32_t align = 32; 
    uint32_t typeSize = 0;
    auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype == ge::DT_FLOAT) {
        typeSize = 4;
    } else {
        typeSize = 2; // FP16
    }
    
    // 计算每行占用的字节数，并向上取整到 32 字节倍数
    uint32_t rowSize = m * typeSize;
    uint32_t alignedRowSize = (rowSize + align - 1) / align * align;
    uint32_t tileLength = alignedRowSize / typeSize; // 对齐后的元素个数

    // 4. 决定核数 (BlockDim)
    uint32_t aicoreNum = ascendcPlatform.GetCoreNumAic();
    uint32_t usedCoreNum = aicoreNum;

    // 小数据量优化：如果 N 很小，没必要用多核，避免通信开销
    if (n < aicoreNum) {
        usedCoreNum = 1;
    }
    
    // 设置使用的核数
    context->SetBlockDim(usedCoreNum);

    // 5. 关键优化：Cyclic Tiling (循环分配) 参数计算
    // 我们不再算每个核处理多少行 (rowsPerBlock)，而是让每个核跳着处理
    // 策略：Core i 处理行索引为: i, i + usedCoreNum, i + 2*usedCoreNum ...
    // 
    // 这种模式下，Host 端只需要传基础参数，具体的循环逻辑由 Kernel 自己算
    // 为了兼容之前的结构体，我们复用字段：
    // usedCoreNum -> 传给 kernel，作为 stride (步长)
    // 
    // 注意：这里需要稍微调整一下 Kernel 的逻辑，所以 Host 这边传参要改一下含义
    // 现在的逻辑变得极其简单：Host 只需要把 N, M, P, CoreNum 传下去就行了
    
    tiling.set_n(n);
    tiling.set_m(m);
    tiling.set_p(p);
    tiling.set_tileLength(tileLength);
    tiling.set_usedCoreNum(usedCoreNum); // 新增：告诉 Kernel 总共有多少个核在跑

    // 计算 TilingKey (保持默认 1)
    tiling.set_tilingKey(1);

    // 6. 序列化数据
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    int64_t n = x1_shape->GetDim(0);
    int64_t outputSize = n * (n - 1) / 2;
    
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, outputSize);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Pdist : public OpDef {
public:
    explicit Pdist(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("p")
            .AttrType(OPTIONAL)
            .Float(2.0);

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Pdist);
} // namespace ops