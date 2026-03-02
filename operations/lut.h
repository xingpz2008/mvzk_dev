#ifndef MVZK_LUT_H__
#define MVZK_LUT_H__

#include <vector>
#include <utility>
#include <cstdint>
#include "emp-zk/emp-mvzk/data_type/PolyTensor.h"

class MVZKExec;

/**
 * @brief 查找表句柄 (Lightweight Handle)
 * * 这是一个极其轻量的对象，内部仅持有一个 int table_id。
 * 它的生命周期与底层的表数据无关。你可以随时创建它，随时销毁它。
 * 底层数据和累积的 Buffer 托管在全局 MVZKExec 中，直到最后统一检查。
 */
class PublicTable {
private:
    int table_id = -1; // 指向 MVZKExec 中数据的句柄

public:
    // 默认构造 (无效句柄)
    PublicTable() = default;

    /**
     * @brief 构造并注册表
     * 这会在 MVZKExec 中分配持久化存储，并返回 ID。
     */
    PublicTable(const std::vector<std::pair<uint64_t, uint64_t>>& table_data);

    /**
     * @brief 提交查找请求 (Non-blocking)
     * 请求会被放入底层的 Buffer 中，不会立即执行。
     */
    void lookup(const PolyTensor& keys, const PolyTensor& vals);

    // 获取 ID
    int get_id() const { return table_id; }
};

class RangeCheckTable{
private:
    int table_id = -1;

public:
    RangeCheckTable() = default;
    RangeCheckTable(const std::vector<uint64_t>& table_data);
    void range_check(const PolyTensor& x, const std::string& tag = "Unknown");
    int get_id() const {return table_id;}
};

/**
 * @brief 强制执行所有表的检查
 * 通常不需要手动调用，MVZKExec 析构时会自动执行。
 * It seems that the function should not be invoked by the user.
 */
void LUT_FlushAll();
void Range_Check_FlushAll();

#endif