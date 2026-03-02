#ifndef POLY_DELTA_H__
#define POLY_DELTA_H__

#include <vector>
#include <cstdint>
#include <utility> // std::move

// 【关键】前向声明 MVZKExec，不 include 头文件
class MVZKExec; 

class PolyDelta {
public:
    // ==========================================
    // 数据成员 (Public 方便访问)
    // ==========================================
    int degree;
    std::vector<uint64_t> coeffs; // Prover
    uint64_t real_val;            // Prover
    uint64_t key;                 // Verifier              
    mutable bool is_consumed; 
    bool is_pre_generated;
    bool is_constraint;

    // ==========================================
    // 构造函数
    // ==========================================
    PolyDelta() : degree(0), key(0), real_val(0), is_consumed(false), is_pre_generated(false), is_constraint(false) {}

    // 移动构造
    PolyDelta(PolyDelta&& other) noexcept 
        : degree(other.degree), coeffs(std::move(other.coeffs)), real_val(other.real_val),
          key(other.key), is_consumed(other.is_consumed), is_pre_generated(other.is_pre_generated), 
          is_constraint(other.is_constraint) {
        other.is_consumed = true;
        other.degree = 0;
    }

    // 移动赋值
    PolyDelta& operator=(PolyDelta&& other) noexcept {
        if (this != &other) {
            degree = other.degree;
            coeffs = std::move(other.coeffs);
            key = other.key;
            real_val = other.real_val;
            is_consumed = other.is_consumed;
            other.is_consumed = true;
            other.degree = 0;
            is_pre_generated = other.is_pre_generated;
            is_constraint = other.is_constraint;
        }
        return *this;
    }

    // 禁止拷贝
    PolyDelta(const PolyDelta&) = delete;
    PolyDelta& operator=(const PolyDelta&) = delete;
    PolyDelta clone() const;

    // 【关键】析构函数只声明，去 .cpp 里实现
    ~PolyDelta();

    // ==========================================
    // 运算符重载声明 (只声明)
    // ==========================================
    
    // 1. Poly vs Poly
    PolyDelta operator+(const PolyDelta& rhs) const;
    PolyDelta operator-(const PolyDelta& rhs) const;
    PolyDelta operator*(const PolyDelta& rhs) const;

    // 2. Poly vs Scalar
    PolyDelta operator+(uint64_t scalar) const;
    PolyDelta operator-(uint64_t scalar) const;
    PolyDelta operator*(uint64_t scalar) const;

    // 3. Scalar vs Poly (友元函数声明)
    friend PolyDelta operator+(uint64_t scalar, const PolyDelta& rhs);
    friend PolyDelta operator*(uint64_t scalar, const PolyDelta& rhs);
    friend PolyDelta operator-(uint64_t scalar, const PolyDelta& rhs);

    // 4. Check helper
    static void store_relation(const PolyDelta& lhs, const PolyDelta& rhs);
    static void store_relation(const PolyDelta& lhs, const uint64_t rhs);

    // 5. In-place function
    PolyDelta& operator+=(const PolyDelta& rhs);
    PolyDelta& operator-=(const PolyDelta& rhs); // 【新增】
    PolyDelta& operator*=(uint64_t scalar);
    PolyDelta& operator+=(uint64_t scalar);
    PolyDelta& operator-=(uint64_t scalar);
};

#endif