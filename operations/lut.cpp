#include "lut.h"
#include "emp-zk/emp-mvzk/exec/MVZKExec.h"

PublicTable::PublicTable(const std::vector<std::pair<uint64_t, uint64_t>>& table_data) {
    if (MVZKExec::mvzk_exec) {
        // 向全局执行器注册，获得永久 ID
        this->table_id = MVZKExec::mvzk_exec->register_lut_table(table_data);
    } else {
        LOG_ERROR(" MVZKExec not initialized!");
        exit(-1);
    }
}

void PublicTable::lookup(const PolyTensor& keys, const PolyTensor& vals) {
    if (table_id == -1 || !MVZKExec::mvzk_exec) {
        LOG_ERROR("Unregistered Public Table!");
        return;
    }
    // 转发请求，带上 ID
    MVZKExec::mvzk_exec->submit_lut_check(keys, vals, table_id);
}

void LUT_FlushAll() {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->flush_all_luts();
    }
}

RangeCheckTable::RangeCheckTable(const std::vector<uint64_t>& table_data){
    if (MVZKExec::mvzk_exec) {
        // 向全局执行器注册，获得永久 ID
        this->table_id = MVZKExec::mvzk_exec->register_range_check_table(table_data);
    } else {
        LOG_ERROR(" MVZKExec not initialized!");
        exit(-1);
    }
}

void RangeCheckTable::range_check(const PolyTensor& x, const std::string& tag) {
    if (table_id == -1 || !MVZKExec::mvzk_exec) {
        LOG_ERROR("Unregistered Range Check Table!");
        return;
    }
    // 转发请求，带上 ID
    MVZKExec::mvzk_exec->stat_range_checks[tag] += x.total_elements;
    MVZKExec::mvzk_exec->submit_range_check(x, table_id);
}

void Range_Check_FlushAll(){
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->flush_all_range_checks();
    }
}