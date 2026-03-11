#ifndef POLY_TENSOR_H__
#define POLY_TENSOR_H__

#include <vector>
#include <cstdint>
#include <iostream>
#include <numeric> 
#include <utility> 
#include <cassert>
#include <cstring> // 加上这个，因为 from_public 实现时需要 memcpy

// 前向声明执行器
class MVZKExec;

class PolyTensor {
public:
    // ==========================================
    // 1. 元数据 (Metadata)
    // ==========================================
    std::vector<int> shape;     
    size_t total_elements;      
    int degree;                 
    
    // ==========================================
    // 2. 扁平化数据存储 (SoA Layout)
    // ==========================================
    std::vector<uint64_t> flat_coeffs; // Prover Only
    // std::vector<uint64_t> flat_real_vals; // Prover Only
    std::vector<uint64_t> flat_keys;      // Verifier Only

    // ==========================================
    // 3. 状态标志位
    // ==========================================
    mutable bool is_consumed; 
    bool is_constraint; 
    mutable bool is_from_fresh_matmul; // NOT IN USE, UNABLE TO PERFORM DIM REDUCTION

    // ==========================================
    // 4. 构造与析构
    // ==========================================
    PolyTensor() : total_elements(0), degree(0), is_consumed(false), is_constraint(false), is_from_fresh_matmul(false) {}

    // 标准构造函数 (实现放在这里是没问题的，也可以移到 .cpp)
    PolyTensor(const std::vector<int>& s, int d) 
        : shape(s), degree(d), is_consumed(false), is_constraint(false), is_from_fresh_matmul(false) {
        if (shape.empty()) {
            total_elements = 0;
        } else {
            total_elements = 1;
            for(int dim : shape) total_elements *= (size_t)dim;
        }
        flat_coeffs.resize(total_elements * (d + 1), 0);
        flat_keys.resize(total_elements, 0);
    }

    // 移动构造
    PolyTensor(PolyTensor&& other) noexcept 
        : shape(std::move(other.shape)),
          total_elements(other.total_elements),
          degree(other.degree),
          flat_coeffs(std::move(other.flat_coeffs)),
          flat_keys(std::move(other.flat_keys)),
          is_consumed(other.is_consumed),
          is_constraint(other.is_constraint),
          is_from_fresh_matmul(other.is_from_fresh_matmul) {
        other.is_consumed = true; 
        other.total_elements = 0;
        other.degree = 0;
    }

    // 移动赋值
    PolyTensor& operator=(PolyTensor&& other) noexcept {
        if (this != &other) {
            shape = std::move(other.shape);
            total_elements = other.total_elements;
            degree = other.degree;
            flat_coeffs = std::move(other.flat_coeffs);
            flat_keys = std::move(other.flat_keys);
            is_consumed = other.is_consumed;
            is_constraint = other.is_constraint;
            other.is_consumed = true;
            other.total_elements = 0;
            is_from_fresh_matmul = other.is_from_fresh_matmul;
        }
        return *this;
    }

    // 禁止拷贝
    PolyTensor(const PolyTensor&) = delete;
    PolyTensor& operator=(const PolyTensor&) = delete;

    ~PolyTensor();

    // ==========================================
    // 5. 高性能访问助手 & 工具
    // ==========================================

    // 获取第 deg 阶系数块的起始指针
    uint64_t* get_coeffs_ptr(int deg) {
        assert(deg <= degree);
        return &flat_coeffs[deg * total_elements];
    }
    const uint64_t* get_coeffs_ptr(int deg) const {
        assert(deg <= degree);
        return &flat_coeffs[deg * total_elements];
    }

    uint64_t* get_real_vals_ptr() { 
        return get_coeffs_ptr(degree); 
    }
    const uint64_t* get_real_vals_ptr() const { 
        return get_coeffs_ptr(degree); 
    }

    std::vector<uint64_t> get_real_vals_vector() const {
        // 1. 拿到最高阶系数块的起始指针
        const uint64_t* start_ptr = get_coeffs_ptr(degree);
        
        // 2. 直接利用内存连续性，一键拷贝成全新的 vector 并返回
        return std::vector<uint64_t>(start_ptr, start_ptr + total_elements);
    }

    uint64_t* get_keys_ptr() { return flat_keys.data(); }
    const uint64_t* get_keys_ptr() const { return flat_keys.data(); }

    void mark_consumed() const { is_consumed = true; }

    // 深拷贝函数
    PolyTensor clone() const;

    void copy_from(const PolyTensor& src, size_t dest_offset);

    // 工厂函数
    static PolyTensor from_public(const std::vector<int>& shape, const std::vector<uint64_t>& data);

    // Check helper
    static void store_relation(const PolyTensor& lhs, const PolyTensor& rhs, const std::string& tag = "Unknown");
    static void store_zero_relation(PolyTensor& lhs, const std::string& tag = "Unknown");
    // This function stores a self valid polytensor to buffer.
    static void store_self_relation(const PolyTensor& lhs, const std::string& tag = "Unknown");
    //static void store_relation(const PolyTensor& lhs, const std::vector<uint64_t> rhs);

    // Shape Manipulation
    PolyTensor flatten() const;
    PolyTensor reshape(const std::vector<int>& new_shape) const;

    // Degree Manipulation
    PolyTensor refresh_degree(const std::string& check_name = "Refreshed Tensor");

    // ==========================================
    // 6. 运算符重载 - In-Place (修改自身)
    // ==========================================
    
    // Tensor vs Tensor
    PolyTensor& operator+=(const PolyTensor& rhs);
    PolyTensor& operator-=(const PolyTensor& rhs);
    PolyTensor& operator*=(const PolyTensor& rhs);
    
    // Tensor vs Scalar
    PolyTensor& operator+=(uint64_t scalar);
    PolyTensor& operator-=(uint64_t scalar);
    PolyTensor& operator*=(uint64_t scalar);

    // ==========================================
    // 7. 运算符重载 - 生成新对象
    // ==========================================

    // Tensor op Tensor
    PolyTensor operator+(const PolyTensor& rhs) const;
    PolyTensor operator-(const PolyTensor& rhs) const;
    PolyTensor operator*(const PolyTensor& rhs) const;

    // Tensor op Scalar
    PolyTensor operator+(uint64_t scalar) const;
    PolyTensor operator-(uint64_t scalar) const;
    PolyTensor operator*(uint64_t scalar) const;

    // ==========================================
    // 8. 友元函数 (Scalar在左侧)
    // ==========================================
    
    friend PolyTensor operator+(uint64_t scalar, const PolyTensor& rhs);
    friend PolyTensor operator*(uint64_t scalar, const PolyTensor& rhs);
    friend PolyTensor operator-(uint64_t scalar, const PolyTensor& rhs);

    // ==========================================
    // 9. Helper Function
    // ==========================================
    // void PolyTensor::accumulate_with_random(PolyDelta& accumulator, const std::vector<uint64_t>& chi) const;

    // ==========================================
    // 10. Matrix Multiplication API
    // ==========================================
    PolyTensor MatMul(const PolyTensor& rhs) const;
    friend PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs);
};

#endif