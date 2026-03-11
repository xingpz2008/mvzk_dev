#ifndef MVZK_EXECUTION_H__
#define MVZK_EXECUTION_H__

#include <vector>
#include <cstdint>
#include <map>
#include <memory>
#include "emp-zk/emp-vole/utility.h"
#include "../utility.h"

class PolyDelta; 
class PolyTensor;

class MVZKExec {
protected:
    virtual void accumulate_delta_buffer(PolyDelta& final_item) = 0;
    virtual void accumulate_tensor_buffer(PolyDelta& final_item) = 0;

    static PolyTensor tree_product(std::vector<PolyTensor>& items) {
        if (items.empty()) return PolyTensor::from_public({1}, {1});
        if (items.size() == 1) return std::move(items[0]);

        size_t mid = items.size() / 2;
        std::vector<PolyTensor> left(std::make_move_iterator(items.begin()), 
                                     std::make_move_iterator(items.begin() + mid));
        std::vector<PolyTensor> right(std::make_move_iterator(items.begin() + mid), 
                                      std::make_move_iterator(items.end()));
        
        return tree_product(left) * tree_product(right);
    }
    
    // 辅助：将大 Tensor 拆分成 1x1 的小 Tensor 列表 (用于 Tree Product)
    // 这是一个比较重的操作，但为了复用 tree_product 逻辑暂时必须这么做
    // 优化方向：实现专门的 reduce_mul kernel
    std::vector<PolyTensor> split_to_scalars(const PolyTensor& big_tensor) {
        size_t len = big_tensor.total_elements;

        // =========================================================
        // Path A: 串行极速路径 (针对小 Tensor)
        // =========================================================
        // 阈值设为 8192 (经验值)。
        // 如果数据量太小，或者当前已经处于并行区域内 (避免嵌套并行)，则直接串行处理。
        // 这能彻底消除 perf 中看到的 libgomp 调度开销。
        if (len < MVZK_OMP_SIZE_THRESHOLD * 2 || omp_in_parallel()) {
            std::vector<PolyTensor> res;
            res.reserve(len);
            
            for(size_t i = 0; i < len; ++i) {
                // 直接在 vector 尾部构造，无锁，无开销
                res.emplace_back(std::vector<int>{1}, big_tensor.degree);
                PolyTensor& t = res.back();
                
                // Copy Coeffs
                if (!big_tensor.flat_coeffs.empty()) {
                    for(int d = 0; d <= big_tensor.degree; ++d) {
                        t.flat_coeffs[d] = big_tensor.flat_coeffs[d * len + i]; 
                    }
                }

                // Copy Keys
                if(!big_tensor.flat_keys.empty()) {
                    t.flat_keys[0] = big_tensor.flat_keys[i];
                }

                t.is_consumed = false;
            }
            
            big_tensor.is_consumed = true;
            return res;
        }

        // =========================================================
        // Path B: 并行 TLS 路径 (针对大 Tensor)
        // =========================================================
        // 只有当任务足够重时，才付出启动线程的代价
        
        int max_threads = omp_get_max_threads();
        // 这里的 vector 初始化开销相比于下面的计算可以忽略不计
        std::vector<std::vector<PolyTensor>> thread_buffers(max_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_res = thread_buffers[tid];
            
            // 预估容量：避免 resize 带来的多次内存搬运
            local_res.reserve(len / max_threads + 16); 

            // 使用 nowait 移除隐式同步，各跑各的，最大化吞吐
            #pragma omp for schedule(static) nowait
            for(size_t i = 0; i < len; ++i) {
                local_res.emplace_back(std::vector<int>{1}, big_tensor.degree);
                PolyTensor& t = local_res.back(); 

                // --- 拷贝逻辑 (与串行路径一致) ---
                if (!big_tensor.flat_coeffs.empty()) {
                    for(int d = 0; d <= big_tensor.degree; ++d) {
                        t.flat_coeffs[d] = big_tensor.flat_coeffs[d * len + i]; 
                    }
                }
                if(!big_tensor.flat_keys.empty()) {
                    t.flat_keys[0] = big_tensor.flat_keys[i];
                }
                t.is_consumed = false;
            }
        }

        // 3. 结果合并 (Merge)
        // 使用 Move Semantics，只拷贝指针，极快
        std::vector<PolyTensor> res;
        res.reserve(len);
        
        for(auto& buf : thread_buffers) {
            res.insert(res.end(), 
                       std::make_move_iterator(buf.begin()), 
                       std::make_move_iterator(buf.end()));
        }

        big_tensor.is_consumed = true;
        return res;
    }

    struct LUTData {
        // LUTData is the meta LUT instance, containing public table data and all its related lookup requests.
        // 原始表数据
        std::vector<std::pair<uint64_t, uint64_t>> data;
        
        // Prover 索引: 记录 (k,v) 当前的版本号
        std::map<std::pair<uint64_t, uint64_t>, uint64_t> version_tracker;

        // 待处理请求
        struct Request {
            // Request struct may involve a bunch of x-y-v pairs.
            PolyTensor keys;
            PolyTensor vals;
            std::vector<uint64_t> v_j_hint; 
            // Version number: We use this plaintext format when submiting lookup reqests. At verification, before obtaining challenge, 
            // we will convert it to commited value.
        };
        // The buffer contains a lot of requests.
        std::vector<Request> buffer;
        
        size_t current_buffer_size = 0; // 积压的总查询数

        size_t data_hash;

        LUTData(const std::vector<std::pair<uint64_t, uint64_t>>& _data) 
            : data(_data) {
            data_hash = compute_vector_hash(_data);
        }
    };

    struct RangeCheckData {
        // LUTData is the meta LUT instance, containing public table data and all its related lookup requests.
        // 原始表数据
        std::vector<uint64_t> data;
        
        // Prover 索引: 记录 (k,v) 当前的版本号
        // std::map<uint64_t, uint64_t> version_tracker;
        // Here we use vector to replace general map.
        // If the table is not start from zero, considering offset
        // TODO: Implement offset version tracker for a non-zero range check.
        std::vector<uint64_t> version_tracker;

        // 待处理请求
        struct Request {
            // Request struct may involve a bunch of x-y-v pairs.
            PolyTensor vals;
            std::vector<uint64_t> v_j_hint; 
            // Version number: We use this plaintext format when submiting lookup reqests. At verification, before obtaining challenge, 
            // we will convert it to commited value.
        };
        // The buffer contains a lot of requests.
        std::vector<Request> buffer;
        
        size_t current_buffer_size = 0; // 积压的总查询数

        size_t data_hash;

        RangeCheckData(const std::vector<uint64_t>& _data) 
            : data(_data) {
            data_hash = compute_vector_hash(_data);
            version_tracker.resize(_data.size(), 0);
        }
    };

    std::pair<PolyTensor, PolyTensor> concat_buffer_tensors(const LUTData* table) {
        // 1. 基础检查
        if (table->buffer.empty()) {
            return {PolyTensor(), PolyTensor()};
        }

        size_t N_Query = table->current_buffer_size;

        // =========================================================
        // Step 1: 预扫描 (Pre-scan) - 确定最大阶数并检查一致性
        // =========================================================
        
        int max_deg_key = 0;
        int max_deg_val = 0;
        
        // 记录第一个元素的阶数，用于比对是否一致
        int first_deg_key = table->buffer[0].keys.degree;
        int first_deg_val = table->buffer[0].vals.degree;
        
        bool inconsistent_key = false;
        bool inconsistent_val = false;

        for (const auto& req : table->buffer) {
            // 更新最大阶数 (用于分配内存)
            max_deg_key = std::max(max_deg_key, req.keys.degree);
            max_deg_val = std::max(max_deg_val, req.vals.degree);

            // Updating flag
            req.keys.is_consumed = true;
            req.vals.is_consumed = true;
            
            // 检查一致性 (用于报警)
            if (req.keys.degree != first_deg_key) inconsistent_key = true;
            if (req.vals.degree != first_deg_val) inconsistent_val = true;
        }

        // 发出警告
        if (inconsistent_key) {
            LOG_WARN("LUT Keys have mixed degrees! Promoting to max degree: " << max_deg_key);
        }
        if (inconsistent_val) {
            LOG_WARN("Values have mixed degrees! Promoting to max degree: " << max_deg_val);
        }

        // =========================================================
        // Step 2: 分配大块内存 (Big Tensor)
        // =========================================================
        // 使用 max_deg，确保能容纳所有请求的数据
        PolyTensor Big_Keys({(int)N_Query, 1}, max_deg_key);
        PolyTensor Big_Vals({(int)N_Query, 1}, max_deg_val);

        // =========================================================
        // Step 3: 内存拼接 (Copy with Padding)
        // =========================================================
        size_t offset = 0;
        
        for (const auto& req : table->buffer) {
            // Big_Keys 的阶数可能 >= req.keys 的阶数
            // copy_from 会自动处理：只拷贝 req 拥有的阶，Big_Keys 多出来的阶保持为 0 (Padding)
            Big_Keys.copy_from(req.keys, offset);
            Big_Vals.copy_from(req.vals, offset);

            offset += req.keys.total_elements;
        }
    
    // 安全检查
        if (offset != N_Query) {
            LOG_ERROR("Buffer size mismatch in concat! Expected " << N_Query << ", got " << offset);
            exit(-1);
        }

        return {std::move(Big_Keys), std::move(Big_Vals)};
    }

    PolyTensor concat_buffer_tensors(const RangeCheckData* table) {
        // 1. 基础检查
        if (table->buffer.empty()) {
            return PolyTensor();
        }

        size_t N_Query = table->current_buffer_size;

        // =========================================================
        // Step 1: 预扫描 (Pre-scan) - 确定最大阶数并检查一致性
        // =========================================================
        
        int max_deg_key = 0;
        int max_deg_val = 0;
        
        // 记录第一个元素的阶数，用于比对是否一致
        int first_deg_val = table->buffer[0].vals.degree;
        
        bool inconsistent_val = false;

        for (const auto& req : table->buffer) {
            // 更新最大阶数 (用于分配内存)
            max_deg_val = std::max(max_deg_val, req.vals.degree);

            // Updating flag
            req.vals.is_consumed = true;
            
            // 检查一致性 (用于报警)
            if (req.vals.degree != first_deg_val) inconsistent_val = true;
        }

        // 发出警告
        if (inconsistent_val) {
            LOG_WARN("LUT Values have mixed degrees! Promoting to max degree: " << max_deg_val);
        }

        // =========================================================
        // Step 2: 分配大块内存 (Big Tensor)
        // =========================================================
        // 使用 max_deg，确保能容纳所有请求的数据
        PolyTensor Big_Vals({(int)N_Query, 1}, max_deg_val);

        // =========================================================
        // Step 3: 内存拼接 (Copy with Padding)
        // =========================================================
        size_t offset = 0;
        
        for (const auto& req : table->buffer) {
            // Big_Keys 的阶数可能 >= req.keys 的阶数
            // copy_from 会自动处理：只拷贝 req 拥有的阶，Big_Keys 多出来的阶保持为 0 (Padding)
            Big_Vals.copy_from(req.vals, offset);

            offset += req.vals.total_elements;
        }
    
    // 安全检查
        if (offset != N_Query) {
            LOG_ERROR("Buffer size mismatch in concat! Expected " << N_Query << ", got " << offset);
            exit(-1);
        }

        return std::move(Big_Vals);
    }

    // Helper 1: 遍历 PolyTensor 执行 Im2Col
    void helper_im2col_polytensor(
        const PolyTensor& src, PolyTensor& dst,
        int N, int C, int H, int W,
        int kH, int kW, 
        int padding, int stride, int dilation, // 接收新参数
        int H_out, int W_out
    ) {
        // 1. Verifier Keys
        if (!src.flat_keys.empty()) {
            im2col_kernel(src.get_keys_ptr(), dst.get_keys_ptr(), 
                          N, C, H, W, kH, kW, padding, stride, dilation, H_out, W_out);
        }
        // 2. Prover Real Values
        // Removed

        // 3. Prover Coefficients (遍历每一阶)
        if (!src.flat_coeffs.empty()) {
            for (int d = 0; d <= src.degree; ++d) {
                im2col_kernel(src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d), 
                              N, C, H, W, kH, kW, padding, stride, dilation, H_out, W_out);
            }
        }
    }

    // Helper 2: 遍历 PolyTensor 执行权重转置
    void helper_transpose_weight_polytensor(
        const PolyTensor& src, PolyTensor& dst,
        int C_out, int C_in, int kH, int kW
    ) {
        if (!src.flat_keys.empty()) {
            transpose_weight_kernel(src.get_keys_ptr(), dst.get_keys_ptr(), C_out, C_in, kH, kW);
        }
        if (!src.flat_coeffs.empty()) {
            for (int d = 0; d <= src.degree; ++d) {
                transpose_weight_kernel(src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d), C_out, C_in, kH, kW);
            }
        }
    }

    // Helper 3: 遍历 PolyTensor 执行重排加 Bias
    virtual void permute_and_add_bias(
        const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
        int N, int H_out, int W_out, int C_out
    ) = 0;

    void helper_transpose_matrix_polytensor(
        const PolyTensor& src, PolyTensor& dst, int Rows, int Cols
    ) {
        if (!src.flat_keys.empty()) {
            transpose_matrix_kernel(src.get_keys_ptr(), dst.get_keys_ptr(), Rows, Cols);
        }
        if (!src.flat_coeffs.empty()) {
            for (int d = 0; d <= src.degree; ++d) {
                transpose_matrix_kernel(src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d), Rows, Cols);
            }
        }
    }

    virtual void helper_linear_add_bias(
        PolyTensor& data, const PolyTensor& bias, int Rows, int Cols
    ) = 0;
    
public:
    // 全局单例指针，用于 PolyDelta/PolyTensor 内部访问执行环境
    static MVZKExec *mvzk_exec; 
    
    int party; // ALICE(Prover) or BOB(Verifier)

    MVZKExec() { mvzk_exec = this; }
    virtual ~MVZKExec() = default;

    // =========================================================================
    // Part 1: PolyDelta 接口 (单体多项式)
    // =========================================================================
    
    // --- 1.1 Buffer 管理 ---
    virtual void submit_to_buffer(PolyDelta&& pd) = 0;
    virtual void check_all() = 0;
    virtual void submit_tensor_to_buffer(PolyTensor&& pt) = 0;
    virtual void submit_non_zero_tensor_to_buffer(const PolyTensor& target) = 0;
    virtual void finalize_protocol() = 0;
    //virtual void submit_matmul_tensor_to_buffer(PolyTensor&& pt) = 0;

    // --- 1.2 输入与转换 ---
    virtual void input(std::vector<PolyDelta>& pdList, const std::vector<uint64_t>& raw_data) = 0;
    virtual PolyDelta input(uint64_t raw_data) = 0;
    virtual std::vector<PolyDelta> vole2pd(std::vector<__uint128_t>& vole_data, int size) = 0;
    virtual std::vector<PolyDelta> extend_vope_from_vole(uint64_t deg, int size) = 0;
    virtual PolyTensor refresh_tensor_degree(const PolyTensor& high_degree_tensor, const std::string& check_name) = 0;

    // --- 1.3 基础运算 (生成新对象) ---
    // Poly vs Poly
    virtual PolyDelta add(const PolyDelta& lhs, const PolyDelta& rhs) = 0;
    virtual PolyDelta sub(const PolyDelta& lhs, const PolyDelta& rhs) = 0;
    virtual PolyDelta mul(const PolyDelta& lhs, const PolyDelta& rhs) = 0;

    // Poly vs Const
    virtual PolyDelta add_const(const PolyDelta& lhs, uint64_t val) = 0;
    virtual PolyDelta sub_const(const PolyDelta& lhs, uint64_t val) = 0;
    virtual PolyDelta sub_const_rev(uint64_t val, const PolyDelta& rhs) = 0; // val - poly
    virtual PolyDelta mul_const(const PolyDelta& lhs, uint64_t val) = 0;

    // --- 1.4 In-Place 运算 (修改自身) ---
    // Poly vs Poly
    virtual void add_assign(PolyDelta& lhs, const PolyDelta& rhs) = 0;
    virtual void sub_assign(PolyDelta& lhs, const PolyDelta& rhs) = 0;
    
    // Poly vs Const
    virtual void add_assign_const(PolyDelta& lhs, uint64_t val) = 0;
    virtual void sub_assign_const(PolyDelta& lhs, uint64_t val) = 0;
    virtual void mul_assign(PolyDelta& lhs, uint64_t val) = 0;

    // virtual bool verify_poly(PolyDelta& mine, PolyDelta& hers) = 0;

    // =========================================================================
    // Part 2: PolyTensor 接口 
    // =========================================================================
    
    // --- 2.1 输入接口 ---
    virtual PolyTensor input(const std::vector<int>& shape, const std::vector<uint64_t>& raw_data) = 0;

    // --- 2.2 基础运算 (生成新对象) ---
    // Tensor vs Tensor
    virtual PolyTensor add(const PolyTensor& lhs, const PolyTensor& rhs) = 0;
    virtual PolyTensor sub(const PolyTensor& lhs, const PolyTensor& rhs) = 0;
    virtual PolyTensor mul(const PolyTensor& lhs, const PolyTensor& rhs) = 0;

    // --- 2.3 In-Place 运算 (修改自身) ---
    // Tensor vs Tensor
    virtual void add_assign(PolyTensor& lhs, const PolyTensor& rhs) = 0;
    virtual void sub_assign(PolyTensor& lhs, const PolyTensor& rhs) = 0;
    virtual void mul_assign(PolyTensor& lhs, const PolyTensor& rhs) = 0;

    // Tensor vs Scalar (Const)
    virtual void add_assign_const(PolyTensor& lhs, uint64_t val) = 0; 
    
    // 【新增】乘常数 (用于 *= c)
    virtual void mul_assign(PolyTensor& lhs, uint64_t val) = 0;       

    // Note: sub_assign_const (Tensor -= c) 不需要接口
    // 因为在 PolyTensor.cpp 中通过 add_assign_const(PR - val) 实现，减少虚函数数量。

    // Matrix Multiplication
    virtual PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs) = 0;

    virtual std::vector<uint64_t> reveal(const PolyTensor& pt) = 0;

    // =========================================================================
    // Part 3: LUT Section 
    // =========================================================================

    // Stored place for LUT table instance. This instance contains the LUT public table itself, and the related requests.
    std::vector<std::unique_ptr<LUTData>> lut_tables;
    // LUT registration, return the table ID
    virtual int register_lut_table(const std::vector<std::pair<uint64_t, uint64_t>>& data) = 0;
    virtual void submit_lut_check(const PolyTensor& keys, const PolyTensor& vals, int table_id) = 0;
    // This function process one public table with all its lookup requests.
    // For security and performance consideration, we only choose to check one public table each time.
    virtual void process_lut_batch(LUTData* table) = 0;
    
    void flush_all_luts(bool require_immediate_check = false) {
        for (auto& table_ptr : lut_tables) {
            if (table_ptr->current_buffer_size > 0) {
                process_lut_batch(table_ptr.get());
            }
        }
        for (auto& range_check_ptr : range_check_tables) {
            if (range_check_ptr->current_buffer_size > 0) {
                process_range_check_batch(range_check_ptr.get());
            }
        }
        // 确保生成的 Z 约束被提交检查，will check in descructor.
        if (require_immediate_check){
            check_all(); 
        }
    }

    // Range Check Section
    std::vector<std::unique_ptr<RangeCheckData>> range_check_tables;
    virtual int register_range_check_table(const std::vector<uint64_t>& data) = 0;
    virtual void submit_range_check(const PolyTensor& x, int table_id) = 0;
    virtual void process_range_check_batch(RangeCheckData* table) = 0;
    
    void flush_all_range_checks(bool require_immediate_check = false) {
        for (auto& table_ptr : range_check_tables) {
            if (table_ptr->current_buffer_size > 0) {
                process_range_check_batch(table_ptr.get());
            }
        }
        // 确保生成的 Z 约束被提交检查，will check in descructor.
        if (require_immediate_check){
            check_all(); 
        }
    }

    // =========================================================================
    // Part 4: Linear Operation Section
    // =========================================================================


    virtual PolyTensor conv2d(
        const PolyTensor& input, 
        const PolyTensor& weight, 
        const PolyTensor& bias, 
        int stride, int padding, int dilation
    ) {
        // 1. 获取输入维度 (N, C, H, W)
        int N = input.shape[0];
        int C_in = input.shape[1];
        int H_in = input.shape[2];
        int W_in = input.shape[3];

        // 2. 获取权重维度 (Out, In, kH, kW)
        int C_out = weight.shape[0];
        // weight.shape[1] 应该是 C_in，这里可以加个 assert 检查
        int kH = weight.shape[2];
        int kW = weight.shape[3];

        if (input.shape[1] != weight.shape[1]) {
            LOG_ERROR("Conv2D Channel Mismatch! Input: " << C_in << ", Weight: " << weight.shape[1]);
            exit(-1);
        }

        // 3. 计算输出维度 (标准 PyTorch/TensorFlow 公式)
        // H_out = floor((H_in + 2*padding - dilation*(kH-1) - 1) / stride + 1)
        int H_out = (H_in + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
        int W_out = (W_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

        if (H_out <= 0 || W_out <= 0) {
            LOG_ERROR("Conv2D output dimension <= 0. Check your padding/stride settings!");
            exit(-1);
        }

        // =================================================
        // Step 1: Im2Col (Input -> Matrix)
        // =================================================
        // 目标矩阵行数 M = 所有输出像素的总数
        int im2col_M = N * H_out * W_out;
        // 目标矩阵列数 K = 卷积核体积 (Channel * kH * kW)
        int im2col_K = C_in * kH * kW;
        
        PolyTensor col_matrix({im2col_M, im2col_K}, input.degree);
        
        // 调用 Helper，传入所有参数
        helper_im2col_polytensor(input, col_matrix, 
                                 N, C_in, H_in, W_in, kH, kW, 
                                 padding, stride, dilation, 
                                 H_out, W_out);

        // =================================================
        // Step 2: Weight Transform (转置)
        // =================================================
        // Weight: (C_out, K) -> Weight_T: (K, C_out)
        PolyTensor weight_matrix({im2col_K, C_out}, weight.degree);
        helper_transpose_weight_polytensor(weight, weight_matrix, C_out, C_in, kH, kW);


        // =================================================
        // Step 3: GEMM (MatMul)
        // =================================================
        // (M x K) * (K x N) -> (M x N)
        // 结果形状: (N * H_out * W_out) x C_out
        PolyTensor output_matrix = col_matrix.MatMul(weight_matrix);

        //debug_instant_check(output_matrix);

        // =================================================
        // Step 4: Reshape & Bias Add
        // =================================================
        // 将 Row-Major 的 (Pixels, Channels) 重排回 (N, Channels, H, W)
        PolyTensor final_res({N, C_out, H_out, W_out}, output_matrix.degree);
        
        //helper_permute_and_add_bias(output_matrix, final_res,  bias, N, H_out, W_out, C_out);
        this->permute_and_add_bias(output_matrix, final_res, bias, N, H_out, W_out, C_out);

        // 资源标记清理
        input.is_consumed = true;
        weight.is_consumed = true;
        if (bias.total_elements > 0) bias.is_consumed = true;


        return final_res;
    }

    // --- Linear Layer ---
    // Input: (N, *, In_Features) -> 展平为 (M, In)
    // Weight: (Out_Features, In_Features)
    // Bias: (Out_Features)
    virtual PolyTensor linear(
        const PolyTensor& input, 
        const PolyTensor& weight, 
        const PolyTensor& bias
    ) {
        // 1. 维度检查与 Reshape
        if (weight.shape.size() != 2) {
            LOG_ERROR("Linear weight must be 2D (Out, In)");
            exit(-1);
        }
        int Out_Features = weight.shape[0];
        int In_Features = weight.shape[1];

        // 兼容 NLP 的 (N, ..., In_Features) 和 CV 的 (N, In_Features, 1, 1)
        bool is_valid_shape = false;
        if (input.shape.back() == In_Features) {
            is_valid_shape = true; // NLP 范式
        } else if (input.shape.size() >= 2 && input.shape[1] == In_Features && input.total_elements % In_Features == 0) {
            is_valid_shape = true; // CV NCHW 范式 (如 N, C, 1, 1)
        }

        if (!is_valid_shape && (GlobalLogLevel() <= LEVEL_ERROR)) {
            std::cerr << "[ERROR] Linear input/weight mismatch! Input shape: (";
            for(int s : input.shape) std::cerr << s << ",";
            std::cerr << ") Weight in_features: " << In_Features << std::endl;
            exit(-1);
        }

        // 将 Input 视为 (M, K) 矩阵
        size_t K = In_Features;
        size_t M = input.total_elements / K;
        
        // 构造 Flatten 后的 Input View (不拷贝数据，只是逻辑形状)
        // 注意：MatMul 内部只看 shape[0] 和 shape[1]，所以这里临时改 shape 是安全的
        PolyTensor input_mat = input.clone(); // 浅拷贝? 不，clone 是深拷贝。
        // 为了性能，最好不要深拷贝。但 PolyTensor 目前没有 View 机制。
        // 鉴于 Linear 的输入通常是上一层的输出（会被 consumed），直接修改 shape 最快。
        // 但 input 是 const引用... 所以我们不得不 clone 或者 cast。
        // 为了接口安全，我们假定 input_mat 是临时对象。
        input_mat.shape = {(int)M, (int)K}; 

        // 2. Weight Transpose: (Out, In) -> (In, Out)
        // 目标形状 (K, N)
        PolyTensor weight_T({(int)K, Out_Features}, weight.degree);
        helper_transpose_matrix_polytensor(weight, weight_T, Out_Features, In_Features);

        // 3. MatMul
        // (M, K) * (K, Out) -> (M, Out)
        PolyTensor output_mat = input_mat.MatMul(weight_T);

        // 4. Bias Add (Row Broadcast)
        if (bias.total_elements > 0) {
            // 调用多态 Helper，执行高阶对齐加法
            this->helper_linear_add_bias(output_mat, bias, M, Out_Features);
        }

        // 5. Restore Shape
        if (input.shape.back() == In_Features) {
            // 模式 A: NLP 范式 (..., In_Features) -> (..., Out_Features)
            std::vector<int> out_shape = input.shape;
            out_shape.back() = Out_Features;
            output_mat.shape = out_shape;
        } else {
            // 模式 B: CV 范式 (N, In_Features, 1, 1) -> (N, Out_Features)
            // 因为逻辑上已经执行了 Flatten，所以输出直接变为标准的 2D 全连接结果
            output_mat.shape = {(int)M, Out_Features}; 
        }

        // Cleanup
        input.is_consumed = true; 
        weight.is_consumed = true;
        if (bias.total_elements > 0) bias.is_consumed = true;

        return output_mat;
    }

    // --- Conv1D Layer ---
    virtual PolyTensor conv1d(
        const PolyTensor& input, 
        const PolyTensor& weight, 
        const PolyTensor& bias, 
        int stride, int padding, int dilation
    ) {
        // 1. Shapes
        int N = input.shape[0];
        int C_in = input.shape[1];
        int L_in = input.shape[2];

        int C_out = weight.shape[0];
        // weight.shape[1] == C_in
        int K_size = weight.shape[2];

        // 2. Calculate Output Length
        int L_out = (L_in + 2 * padding - dilation * (K_size - 1) - 1) / stride + 1;
        if (L_out <= 0) {
            LOG_ERROR("[ERROR] Conv1D output length <= 0!");
            exit(-1);
        }

        // =================================================
        // Step 1: Im2Col 1D (Input -> Matrix)
        // =================================================
        // M = N * L_out
        // K = C_in * K_size
        int im2col_M = N * L_out;
        int im2col_K = C_in * K_size;

        PolyTensor col_matrix({im2col_M, im2col_K}, input.degree);

        // Helper 1D Im2Col (Need to define this specific helper below or use generic one)
        // 为了代码整洁，我们直接在这里展开 helper 调用，或者写个 helper_im2col_1d
        // 这里直接调用，逻辑更清晰
        if (!input.flat_keys.empty()) {
            im2col_1d_kernel(input.get_keys_ptr(), col_matrix.get_keys_ptr(), 
                             N, C_in, L_in, K_size, padding, stride, dilation, L_out);
        }
        if (!input.flat_coeffs.empty()) {
            for (int d = 0; d <= input.degree; ++d) {
                im2col_1d_kernel(input.get_coeffs_ptr(d), col_matrix.get_coeffs_ptr(d), 
                                 N, C_in, L_in, K_size, padding, stride, dilation, L_out);
            }
        }

        // =================================================
        // Step 2: Weight Transpose
        // =================================================
        // Weight (C_out, K_dim) -> (K_dim, C_out)
        // 这里我们可以复用 utility.h 中的 transpose_matrix_kernel
        // 因为 Weight 已经是 [C_out, C_in, K]，内存上就是 [C_out, K_dim] 的二维数组
        PolyTensor weight_matrix({im2col_K, C_out}, weight.degree);
        
        // 调用我们给 Linear 写过的 transpose helper
        helper_transpose_matrix_polytensor(weight, weight_matrix, C_out, im2col_K);

        // =================================================
        // Step 3: GEMM
        // =================================================
        PolyTensor output_matrix = col_matrix.MatMul(weight_matrix);

        // =================================================
        // Step 4: Reshape & Bias Add (1D Specific)
        // =================================================
        PolyTensor final_res({N, C_out, L_out}, output_matrix.degree);

        this->helper_permute_and_add_bias_1d(output_matrix, final_res, bias, N, L_out, C_out);

        // Cleanup
        input.is_consumed = true;
        weight.is_consumed = true;
        if (bias.total_elements > 0) bias.is_consumed = true;

        return final_res;
    }

    // 【新增虚函数】Conv1D Bias 对齐
    virtual void helper_permute_and_add_bias_1d(
        const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
        int N, int L_out, int C_out
    ) = 0;

    [[deprecated("WARNING: DO NOT USE Standalone BatchNorm2D. Please use offline Conv-BN folding instead.")]]
    virtual PolyTensor batchnorm2d(PolyTensor& pt_in, const std::vector<uint64_t>& A, const std::vector<uint64_t>& B) = 0;

    virtual PolyTensor avgpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding) = 0;

    // =========================================================================
    // Part 5: Non-linear Operation Section
    // =========================================================================

    virtual PolyTensor relu(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) = 0;

    virtual PolyTensor maxpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, uint64_t scale) = 0;

    virtual PolyTensor integrated_nl(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) = 0;

    // =========================================================================
    // Part -1: Debug Helper Section 
    // =========================================================================
    virtual void debug_print(const PolyDelta& pd, std::string name = "") {
        std::cout << "[Base] " << name << " (Not implemented)" << std::endl;
    }

    virtual void debug_print(const PolyTensor& pt, std::string name, int limit = 4) {
        std::cout << "[Base] " << name << " (Not implemented)" << std::endl;
    }

    virtual bool debug_instant_check(const PolyDelta& pd) = 0;
    virtual bool debug_instant_check(const PolyTensor& pt) = 0;

    std::map<std::string, std::map<int, size_t>> stat_constraints; 
    std::map<std::string, size_t> stat_range_checks; // Range Check 一般都是 1 阶查表，保持一维即可

    void print_profiler_report() {
        std::cout << "\n===========================================================" << std::endl;
        std::cout << "          ZK Constraint Profiler Report (Degree-Aware)     " << std::endl;
        std::cout << "===========================================================" << std::endl;
        
        std::cout << "[Polynomial Constraints]" << std::endl;
        size_t total_constraints = 0;
        
        for (const auto& tag_pair : stat_constraints) {
            std::cout << "  \033[36m[" << tag_pair.first << "]\033[0m" << std::endl; // 打印 Tag
            
            size_t sub_total = 0;
            // 遍历该 Tag 下的所有 Degree
            for (const auto& deg_pair : tag_pair.second) {
                std::cout << "    |-- Degree " << std::left << std::setw(5) << deg_pair.first 
                          << ": " << std::setw(10) << deg_pair.second << " constraints" << std::endl;
                sub_total += deg_pair.second;
                total_constraints += deg_pair.second;
            }
            std::cout << "    `-- Sub-Total : " << sub_total << "\n" << std::endl;
        }
        std::cout << "  > TOTAL Constraints Generated : " << total_constraints << "\n\n";

        std::cout << "[Range Checks (LUT Lookups)]" << std::endl;
        size_t total_rc = 0;
        for (const auto& pair : stat_range_checks) {
            std::cout << "  - " << std::left << std::setw(25) << pair.first 
                      << ": " << pair.second << " elements" << std::endl;
            total_rc += pair.second;
        }
        std::cout << "  > TOTAL Range Checks          : " << total_rc << std::endl;
        std::cout << "===========================================================\n" << std::endl;
    }
};

#endif