#ifndef PDIST_TILING_H
#define PDIST_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PdistTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(float, p);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
END_TILING_DATA_DEF;

// 注意这里第一个参数是算子类型名，必须是 Pdist
REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)
}
#endif // PDIST_TILING_H