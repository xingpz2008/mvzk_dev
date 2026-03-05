#include "PolyDelta.h"
#include "../exec/MVZKExec.h" // 【关键】在这里包含 Exec 的定义

// ==========================================
// 析构函数实现
// ==========================================
PolyDelta::~PolyDelta() {
    // 此时 MVZKExec 已完全定义，可以访问 mvzk_exec 静态成员
    if (!is_consumed && degree > 0 && MVZKExec::mvzk_exec && !is_pre_generated) {
        MVZKExec::mvzk_exec->submit_to_buffer(std::move(*this));
    }
}

PolyDelta PolyDelta::clone() const {
    PolyDelta res;

    // 1. 数据属性：完全复制 (Deep Copy)
    res.degree = this->degree;
    res.coeffs = this->coeffs;      // std::vector 的赋值是深拷贝
    res.key = this->key;
    res.is_pre_generated = this->is_pre_generated; // 保留数据来源属性

    // 2. 生命周期属性：重置 (Reset)
    // 克隆出来的新对象应当是“活”的，且默认不作为约束（除非后续显式指定）
    res.is_consumed = false;   
    res.is_constraint = false; 
    
    return res;
}
// ==========================================
// 1. PolyDelta 与 PolyDelta 的运算
// ==========================================

PolyDelta PolyDelta::operator+(const PolyDelta& rhs) const {
    return MVZKExec::mvzk_exec->add(*this, rhs);
}

PolyDelta PolyDelta::operator-(const PolyDelta& rhs) const {
    return MVZKExec::mvzk_exec->sub(*this, rhs);
}

PolyDelta PolyDelta::operator*(const PolyDelta& rhs) const {
    return MVZKExec::mvzk_exec->mul(*this, rhs);
}

// ==========================================
// 2. PolyDelta 与 常数 的运算
// ==========================================

PolyDelta PolyDelta::operator+(uint64_t scalar) const {
    return MVZKExec::mvzk_exec->add_const(*this, scalar);
}

PolyDelta PolyDelta::operator-(uint64_t scalar) const {
    return MVZKExec::mvzk_exec->sub_const(*this, scalar);
}

PolyDelta PolyDelta::operator*(uint64_t scalar) const {
    return MVZKExec::mvzk_exec->mul_const(*this, scalar);
}

// ==========================================
// 3. 友元函数实现 (全局函数)
// ==========================================

// c + poly
PolyDelta operator+(uint64_t scalar, const PolyDelta& rhs) {
    // 调用 add_const (交换律)
    return MVZKExec::mvzk_exec->add_const(rhs, scalar);
}

// c * poly
PolyDelta operator*(uint64_t scalar, const PolyDelta& rhs) {
    // 调用 mul_const (交换律)
    return MVZKExec::mvzk_exec->mul_const(rhs, scalar);
}

// c - poly
PolyDelta operator-(uint64_t scalar, const PolyDelta& rhs) {
    // 这是一个特殊情况，需要专门的 sub_const_rev 接口
    return MVZKExec::mvzk_exec->sub_const_rev(scalar, rhs);
}

void PolyDelta::store_relation(const PolyDelta& lhs, const PolyDelta& rhs){
    // Store the relation and submit to buffer. Making it a constraint.

    PolyDelta constraint = lhs - rhs;
    constraint.is_constraint = true;
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->submit_to_buffer(std::move(constraint));
    }
    return;
}

void PolyDelta::store_relation(const PolyDelta& lhs, const uint64_t rhs){
    // Store the relation and submit to buffer. Making it a constraint.

    PolyDelta constraint = lhs - rhs;
    constraint.is_constraint = true;
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->submit_to_buffer(std::move(constraint));
    }
    return;
}

PolyDelta& PolyDelta::operator+=(const PolyDelta& rhs) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->add_assign(*this, rhs);
    }
    return *this;
}

PolyDelta& PolyDelta::operator*=(uint64_t scalar) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->mul_assign(*this, scalar);
    }
    return *this;
}

PolyDelta& PolyDelta::operator-=(const PolyDelta& rhs) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->sub_assign(*this, rhs);
    }
    return *this;
}

PolyDelta& PolyDelta::operator+=(uint64_t scalar) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->add_assign_const(*this, scalar);
    }
    return *this;
}

PolyDelta& PolyDelta::operator-=(uint64_t scalar) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->sub_assign_const(*this, scalar);
    }
    return *this;
}