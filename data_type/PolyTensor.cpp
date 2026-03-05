// PolyTensor.cpp
#include "PolyTensor.h"
#include "../exec/MVZKExec.h"
#include <cstring>

// ==========================================
// 1. 析构与辅助函数
// ==========================================
PolyTensor::~PolyTensor() {
    // 1. 如果对象已经被 Move 走了（total_elements == 0），或者是空 Tensor，直接返回
    // 这种情况下它是“空壳”，不需要检查
    if (total_elements == 0) {
        return;
    }

    // 2. 检查 "Dangling Wire" (悬空导线)
    // 如果一个 Tensor：
    //   a. 还没有被消耗 (is_consumed == false)
    //   b. 且不是显式的约束项 (is_constraint == false)
    //   c. 且不是用作临时计算的 (通常我们只关心那些主要的 Tensor)
    // 那么这可能是一个 Bug。
    
    
    if (!is_consumed && !is_constraint) {
        // 输出警告信息，帮助定位是哪个 Tensor 出了问题
        std::cerr << "[WARNING] PolyTensor destroyed but NOT consumed! "
                  << "Potential circuit logic error (dangling wire)." 
                  << " Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cerr << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        std::cerr << "], Degree: " << degree << std::endl;
        
        // 在严格模式下，你甚至可以直接 exit(-1) 强迫自己修 bug
    }
    
    // 3. 内存清理
    // std::vector 会自动析构，不需要手动释放 flat_coeffs 等
}

PolyTensor PolyTensor::clone() const {
    PolyTensor res;
    // 复制元数据
    res.shape = this->shape;
    res.total_elements = this->total_elements;
    res.degree = this->degree;
    
    // 复制核心数据 (Deep Copy)
    res.flat_coeffs = this->flat_coeffs;
    res.flat_keys = this->flat_keys;
    
    // 重置状态
    res.is_consumed = false;
    res.is_constraint = false;
    
    return res;
}

PolyTensor PolyTensor::from_public(const std::vector<int>& shape, const std::vector<uint64_t>& data) {
    // 创建 0 阶 Tensor
    PolyTensor res(shape, 0); 
    
    if (data.size() != res.total_elements) {
        std::cerr << "[ERROR] Size mismatch in PolyTensor::from_public" << std::endl;
        exit(-1);
    }

    // 填充 0 阶系数 (对于公开常数，MAC/Coeffs[0] 就是数值本身)
    std::memcpy(res.get_coeffs_ptr(0), data.data(), data.size() * sizeof(uint64_t));
    
    // 填充 Real Values
    std::memcpy(res.get_real_vals_ptr(), data.data(), data.size() * sizeof(uint64_t));

    std::memcpy(res.get_keys_ptr(), data.data(), data.size() * sizeof(uint64_t));

    return res;
}

void PolyTensor::copy_from(const PolyTensor& src, size_t dest_offset) {
    // 1. 基础越界检查
    if (dest_offset + src.total_elements > this->total_elements) {
        std::cerr << "[ERROR] Buffer overflow in copy_from!" << std::endl;
        exit(-1);
    }
    
    // 2. 阶数检查 (这是为了配合 concat 的逻辑)
    // 目标 Tensor (this) 的阶数必须足够大，容纳源数据
    if (this->degree < src.degree) {
        std::cerr << "[ERROR] Destination tensor degree too small in copy_from!" << std::endl;
        exit(-1);
    }

    size_t copy_bytes = src.total_elements * sizeof(uint64_t);

    // 3. 拷贝系数 (Coeffs)
    // 【关键】只循环到 src.degree。
    // 如果 this->degree 更大，那么高阶部分本身就是初始化为 0 的，不需要额外操作。
    for (int d = 0; d <= src.degree; ++d) {
        std::memcpy(this->get_coeffs_ptr(d) + dest_offset, 
                    src.get_coeffs_ptr(d), 
                    copy_bytes);
    }

    // 4. 拷贝 Real Values
    std::memcpy(this->get_real_vals_ptr() + dest_offset, 
                src.get_real_vals_ptr(), 
                copy_bytes);

    // 5. 拷贝 Keys (Verifier Only)
    // 注意：Verifier 的 Keys 只有一层，不随阶数变化，直接拷
    if (!this->flat_keys.empty() && !src.flat_keys.empty()) {
        std::memcpy(this->get_keys_ptr() + dest_offset, 
                    src.get_keys_ptr(), 
                    copy_bytes);
    }
    
    src.mark_consumed();
}

// ==========================================
// 2. Tensor vs Tensor (In-Place)
// ==========================================

PolyTensor& PolyTensor::operator+=(const PolyTensor& rhs) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->add_assign(*this, rhs);
    }
    return *this;
}

PolyTensor& PolyTensor::operator-=(const PolyTensor& rhs) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->sub_assign(*this, rhs);
    }
    return *this;
}

PolyTensor& PolyTensor::operator*=(const PolyTensor& rhs) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->mul_assign(*this, rhs);
    }
    return *this;
}

// ==========================================
// 3. Tensor vs Scalar (In-Place)
// ==========================================

PolyTensor& PolyTensor::operator+=(uint64_t scalar) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->add_assign_const(*this, scalar);
    }
    return *this;
}

PolyTensor& PolyTensor::operator-=(uint64_t scalar) {
    // 优化：利用模运算性质 T -= c  <=>  T += (PR - c)
    // 这样不需要在 Exec 里实现 sub_assign_const
    uint64_t neg_scalar = (scalar == 0) ? 0 : (PR - scalar);
    return this->operator+=(neg_scalar);
}

PolyTensor& PolyTensor::operator*=(uint64_t scalar) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->mul_assign(*this, scalar);
    }
    return *this;
}

// ==========================================
// 4. Tensor vs Tensor (生成新对象)
// ==========================================

PolyTensor PolyTensor::operator+(const PolyTensor& rhs) const {
    if (MVZKExec::mvzk_exec) {
        // 直接调用 Exec，利用 RVO 优化，无多余拷贝
        return MVZKExec::mvzk_exec->add(*this, rhs);
    }
    return PolyTensor();
}

PolyTensor PolyTensor::operator-(const PolyTensor& rhs) const {
    if (MVZKExec::mvzk_exec) {
        // 调用我们刚刚在 Exec 里补齐的 sub 接口
        return MVZKExec::mvzk_exec->sub(*this, rhs);
    }
    return PolyTensor();
}

PolyTensor PolyTensor::operator*(const PolyTensor& rhs) const {
    if (MVZKExec::mvzk_exec) {
        return MVZKExec::mvzk_exec->mul(*this, rhs);
    }
    return PolyTensor();
}

// ==========================================
// 5. Tensor vs Scalar (生成新对象)
// ==========================================

PolyTensor PolyTensor::operator+(uint64_t scalar) const {
    // 策略：Clone + In-Place (最稳健)
    PolyTensor res = this->clone();
    res += scalar;
    this->is_consumed = true;
    return res;
}

PolyTensor PolyTensor::operator-(uint64_t scalar) const {
    PolyTensor res = this->clone();
    res -= scalar; // 会调用 += (PR-scalar)
    this->is_consumed = true;
    return res;
}

PolyTensor PolyTensor::operator*(uint64_t scalar) const {
    PolyTensor res = this->clone();
    res *= scalar;
    this->is_consumed = true;
    return res;
}

// ==========================================
// 6. 友元函数 (Scalar vs Tensor)
// ==========================================

// c + T -> T + c
PolyTensor operator+(uint64_t scalar, const PolyTensor& rhs) {
    return rhs + scalar;
}

// c * T -> T * c
PolyTensor operator*(uint64_t scalar, const PolyTensor& rhs) {
    return rhs * scalar;
}

// c - T -> -T + c -> T*(-1) + c
PolyTensor operator-(uint64_t scalar, const PolyTensor& rhs) {
    // 1. 克隆 rhs
    PolyTensor res = rhs.clone();
    
    // 2. 变为 -T (乘 PR-1)
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->mul_assign(res, PR - 1);
    }
    
    // 3. 加 c
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->add_assign_const(res, scalar);
    }
    
    return res;
}

PolyTensor PolyTensor::MatMul(const PolyTensor& rhs) const {
    if (MVZKExec::mvzk_exec) {
        // *this 就是 lhs
        return MVZKExec::mvzk_exec->MatMul(*this, rhs);
    }
    return PolyTensor();
}

PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs) {
    if (MVZKExec::mvzk_exec) {
        // 直接透传两个参数
        return MVZKExec::mvzk_exec->MatMul(lhs, rhs);
    }
    return PolyTensor();
}

// ==========================================
// 7. Check Helper
// ==========================================

// PolyTensor.cpp

void PolyTensor::store_relation(const PolyTensor& lhs, const PolyTensor& rhs, const std::string& tag) {
    // 0. Statistical 
    int max_degree = std::max(lhs.degree, rhs.degree);
    MVZKExec::mvzk_exec->stat_constraints[tag][max_degree] += lhs.total_elements;
    
    // 1. 计算差值 Diff = LHS - RHS
    // 这里会自动调用 Exec::sub，处理阶数对齐和模减法
    // 结果 Diff 是一个全零张量 (如果验证通过的话)
    PolyTensor diff = lhs - rhs;

    // 2. 标记为约束 (Constraint)
    // 这是一个特殊的 Flag，告诉 Exec "这个 Tensor 的值必须为 0"
    diff.is_constraint = true;

    // 3. 提交到 Tensor 专用 Buffer
    // 使用 std::move 避免拷贝，直接把 Diff 的内存所有权移交给 Exec
    // Add matmul verification specification
    /* [Dim reduction is not applicable in high level VOLE design, so we use uniform tensor buffer for check.]
    if (lhs.is_from_fresh_matmul or rhs.is_from_fresh_matmul){
        MVZKExec::mvzk_exec->submit_matmul_tensor_to_buffer(std::move(diff));
        lhs.is_from_fresh_matmul = false;
        rhs.is_from_fresh_matmul = false;
    }else{
        MVZKExec::mvzk_exec->submit_tensor_to_buffer(std::move(diff));
    }
    */ 
    MVZKExec::mvzk_exec->submit_tensor_to_buffer(std::move(diff));
    
    // 4. 标记输入对象已消耗
    // (可选) 如果你的析构函数里有 "未消耗对象报警" 机制，这一步是必须的
    // 这表示 lhs 和 rhs 的生命周期/用途在此处完结
    lhs.mark_consumed();
    rhs.mark_consumed();
}

void PolyTensor::store_zero_relation(PolyTensor& lhs, const std::string& tag){
    MVZKExec::mvzk_exec->stat_constraints[tag][lhs.degree] += lhs.total_elements;
    lhs.is_constraint = true;
    MVZKExec::mvzk_exec->submit_tensor_to_buffer(std::move(lhs));
}

void PolyTensor::store_self_relation(const PolyTensor& lhs, const std::string& tag){
    // In this step, we check if a non-zero polytensor is valid. 
    // Considering that our current check_all only handles zero PolyTensors, we "forge" a another polytensor for verification with plus 1 deg.
    MVZKExec::mvzk_exec->stat_constraints[tag][lhs.degree] += lhs.total_elements;
    lhs.mark_consumed();
    MVZKExec::mvzk_exec->submit_non_zero_tensor_to_buffer(lhs);
}


// ==========================================
// Shape Manipulation
// ==========================================

PolyTensor PolyTensor::flatten() const {
    if (this->shape.size() < 2) return this->clone();

    // 1. 获取 Batch Size (第 0 维)
    int batch_size = this->shape[0];

    // 2. 计算剩余维度的乘积 (C * H * W)
    size_t flattened_dim = 1;
    for (size_t i = 1; i < this->shape.size(); ++i) {
        flattened_dim *= (size_t)this->shape[i];
    }

    // 3. 克隆当前张量 (深拷贝数据)
    PolyTensor res = this->clone(); 

    // 4. 修改新张量的 Shape
    res.shape.clear();
    res.shape.push_back(batch_size);
    res.shape.push_back((int)flattened_dim);

    // 5. 标记原张量已消耗
    this->is_consumed = true;

    return res;
}

PolyTensor PolyTensor::reshape(const std::vector<int>& new_shape) const {
    size_t old_count = this->total_elements;
    size_t new_count = 1;
    for(int s : new_shape) new_count *= (size_t)s;

    if (old_count != new_count) {
        LOG_ERROR("Reshape size mismatch! Old: " << old_count << ", New: " << new_count);
        exit(-1);
    }

    PolyTensor res = this->clone();
    res.shape = new_shape;
    this->is_consumed = true;
    return res;
}