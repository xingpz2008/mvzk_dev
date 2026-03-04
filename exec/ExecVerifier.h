#ifndef MVZK_EXECUTION_VERIFIER_H__
#define MVZK_EXECUTION_VERIFIER_H__

#include "MVZKExec.h"
#include "emp-zk/emp-zk-arith/ostriple.h"
#include "emp-zk/emp-vole/emp-vole.h"
#include "emp-zk/emp-vole/vole_triple.h"
#include "../config.h"
#include "../data_type/PolyDelta.h"
#include "../data_type/PolyTensor.h"
#include "../utility.h"
#include "emp-tool/utils/hash.h"
#include <omp.h>

#define LOW64(x) _mm_extract_epi64((block)x, 0)
#define HIGH64(x) _mm_extract_epi64((block)x, 1)

template <typename IO> 
class MVZKExecVerifier : public MVZKExec {
public:
    IO *io = nullptr;
    VoleTriple<IO> *vole = nullptr;
    int party;
    int threads;
    __uint128_t delta; //TODO: Check if this can be converted to uint64_t
    PRG prg;
    std::vector<uint64_t> delta_powers_cache;
    
    // 【新增】Buffer：用于存储待验证的约束多项式
    std::vector<PolyDelta> check_buffer;
    std::vector<PolyTensor> check_tensor_buffer;
    std::vector<PolyTensor> check_matmul_buffer;
    // 【新增】Buffer 阈值：防止内存溢出，可根据需要调整

    MVZKExecVerifier(IO **ios) : MVZKExec() {
        this->party = BOB; 
        this->io = ios[0];
        this->threads = MVZK_CONFIG_THREADS_NUM; // 确保使用定义的线程数
        
        // 【关键】将自己注册为当前的全局执行环境
        MVZKExec::mvzk_exec = this;

        this->vole = new VoleTriple<IO>(ALICE, threads, ios); // 对方是 ALICE
        
        delta_gen();
        //vole->setup(10);
        vole->setup(delta);
        
        // VoleTriple 预生成 VOLE triples
        __uint128_t tmp;
        vole->extend(&tmp, 1);

        delta_powers_cache.resize(2);
        delta_powers_cache[0] = 1;
        delta_powers_cache[1] = (uint64_t)delta;
    }

    virtual ~MVZKExecVerifier() {
        // 析构前强制检查剩余的 buffer
        flush_all_luts();
        check_all();
        if(vole) delete vole;
    }

    // =========================================================
    // 1. Buffer 管理接口实现
    // =========================================================

    void submit_to_buffer(PolyDelta&& pd) override {
        if (!pd.is_constraint){
            LOG_WARN("A non-constraint PolyDelta received by buffer!");
        }
        // 1. 标记为已消耗
        pd.is_consumed = true;

        // 2. 移动语义存入 Buffer
        // Verifier 存的是包含 Key 和 Degree 的 PolyDelta
        check_buffer.emplace_back(std::move(pd));

        // 3. 检查阈值
        if (check_buffer.size() >= MVZK_MULT_CHECK_CNT) {
            //WHITE("[INFO] Check buffer threshold reached, instant check now.");
            check_all();
        }
    }

    void submit_tensor_to_buffer(PolyTensor&& pt) override {
        if (!pt.is_constraint){
            LOG_WARN("A non-constraint PolyTensor received by buffer!");
        }
        //BLUE("[INFO] PolyTensor Submitted to buffer with size " << pt.total_elements << ", deg = " << pt.degree);
        pt.is_consumed = true;
        check_tensor_buffer.emplace_back(std::move(pt));
        if (check_tensor_buffer.size() >= MVZK_CONFIG_TENSOR_CHECK_CNT) {
            //WHITE("[INFO] Check tensor buffer threshold reached, instant check now.");
            check_all();
        }
    }

    //  Verifier 侧的 submit_non_zero_tensor_to_buffer 实现
    void submit_non_zero_tensor_to_buffer(const PolyTensor& target) override {
        //if (!target.is_constraint) {
            //std::cout << "[WARNING] A non-constraint non-zero PolyTensor received by buffer!" << std::endl;
        //}
        
        // 完美利用 mutable 特性，打上防悬空标签，且不破坏数据生命周期！
        target.is_consumed = true; 

        // 1. 构造更高 1 阶的新张量
        int new_degree = target.degree + 1;
        PolyTensor padded_tensor(target.shape, new_degree);
        padded_tensor.is_constraint = true;
        
        // 2. Verifier 侧的核心魔法：Keys 的等价性
        // 因为最高阶系数为 0，所以 \Delta^{d+1} 项对 Key 没有任何贡献。
        // 我们只需要把旧的 Keys 完美拷贝给新的张量即可！
        const uint64_t* in_keys = target.get_keys_ptr();
        uint64_t* out_keys = padded_tensor.get_keys_ptr();
        
        // 使用高速内存拷贝
        std::copy(in_keys, in_keys + target.total_elements, out_keys);

        // 3. 将伪造好的张量，利用 std::move 压入底层的零校验池
        this->check_tensor_buffer.push_back(std::move(padded_tensor));
        if (check_tensor_buffer.size() >= MVZK_CONFIG_TENSOR_CHECK_CNT) {
            //WHITE("[INFO] Check tensor buffer threshold reached, instant check now.");
            check_all();
        }
    }

    /*
    void submit_matmul_tensor_to_buffer(PolyTensor&& pt){
        if (!pt.is_constraint) {
             std::cout << "[WARNING] A non-constraint PolyTensor (from MatMul) received by buffer!" << std::endl;
        }
        pt.is_consumed = true;

        check_matmul_buffer.emplace_back(std::move(pt));
        if (check_tensor_buffer.size() >= MVZK_CONFIG_MATMUL_TENSOR_CHECK_CNT) { 
            check_all();
        }
    }
    */

    void check_all() override {

        flush_all_luts();
        flush_all_range_checks();

        if (check_buffer.empty() && check_tensor_buffer.empty()) return;

        // 1. 生成并发送种子
        block seed;
        prg.random_block(&seed, 1);
        io->send_data(&seed, sizeof(block));
        this->prg.reseed(&seed);

        // 2. 全局累加器
        PolyDelta final_checked_item;

        // 3. 分别累加
        accumulate_delta_buffer(final_checked_item);
        accumulate_tensor_buffer(final_checked_item);

        if (final_checked_item.degree > MVZK_FFT_WARNING_LIMIT){
            LOG_WARN("FFT Optimization is recommended, deg = " << final_checked_item.degree << ", alert limit = " << MVZK_FFT_WARNING_LIMIT << ".");
        }

        // 4. 生成 Masked VOPE
        if (final_checked_item.degree > 1){
            PolyDelta vope = std::move(extend_vope_from_vole(final_checked_item.degree - 1, 1)[0]);

            //debug_print(vope, "Verifier VOPE");

            final_checked_item.key = add_mod(final_checked_item.key, vope.key);
            vope.is_consumed = true;
            //final_checked_item = final_checked_item + vope;
            
            // 5. 接收并验证
            PolyDelta recvd_poly = std::move(recv_poly());

            if (verify_poly(final_checked_item, recvd_poly) == false){
                LOG_ERROR("Check failed. (Continue to other test case if in dev_test mode.)");
                if (MVZK_CONFIG_INSTANT_ABORT){
                    exit(-1);
                }
            }else{
                LOG_INFO("Check success.");
            }
        }else{
            emp::Hash hash_instance;
            char dig[emp::Hash::DIGEST_SIZE];
            hash_instance.put(&final_checked_item.key, sizeof(uint64_t));
            hash_instance.digest(dig);
            char dig_recv[emp::Hash::DIGEST_SIZE];
            io->recv_data(dig_recv, emp::Hash::DIGEST_SIZE);
            if (!cmpBlock((block *)dig, (block *)dig_recv,
                    emp::Hash::DIGEST_SIZE / 16)) {
                LOG_ERROR("Check failed. (Hash failure)");
                if (MVZK_CONFIG_INSTANT_ABORT){ exit(-1); }
            }else{
                LOG_INFO("Check success.");
            }
        }
        final_checked_item.is_consumed = true;
    }

    // =========================================================
    // 2. 多项式运算接口实现 (PolyDelta vs PolyDelta)
    // =========================================================
    // 注意：Prover 需要操作 PolyDelta 中的 coeffs 向量进行多项式运算

    PolyDelta add(const PolyDelta& lhs, const PolyDelta& rhs) override {
        PolyDelta res;
        res.degree = std::max(lhs.degree, rhs.degree);
        
        size_t offset_lhs = res.degree - lhs.degree;
        size_t offset_rhs = res.degree - rhs.degree;

        // 优化核心：同阶相加时，直接相加，跳过所有乘法
        uint64_t k1 = lhs.key;
        if (offset_lhs > 0) {
            // 只有需要移位时才调用乘法
            // 此时 pow_mod(..., 1) 会很快返回，但 mult_mod 必须执行
            k1 = mult_mod(k1, delta_pow(this->delta, offset_lhs));
        }

        uint64_t k2 = rhs.key;
        if (offset_rhs > 0) {
            k2 = mult_mod(k2, delta_pow(this->delta, offset_rhs));
        }

        res.key = add_mod(k1, k2);
        
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        return res;
    }

    PolyTensor add(const PolyTensor& lhs, const PolyTensor& rhs) override {
        // 1. 维度检查
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("Tensor shape mismatch in verifier add!");
            exit(-1);
        }

        // 2. 构造结果对象 (只分配 flat_keys 内存，coeffs 是空的/0)
        int res_degree = std::max(lhs.degree, rhs.degree);
        PolyTensor res(lhs.shape, res_degree);
        
        size_t size = lhs.total_elements;

        // 3. 获取指针 (Direct Access)
        const uint64_t* lhs_keys = lhs.get_keys_ptr();
        const uint64_t* rhs_keys = rhs.get_keys_ptr();
        uint64_t* res_keys = res.get_keys_ptr();

        // 4. 准备计算参数
        // 将类成员 delta 转为局部变量，帮助编译器优化寄存器使用
        uint64_t local_delta = (uint64_t)this->delta;
        
        int offset_lhs = res_degree - lhs.degree;
        int offset_rhs = res_degree - rhs.degree;

        // 5. 分情况并行计算 (避免循环内 if 分支)
        
        if (offset_lhs == 0 && offset_rhs == 0) {
            // 【情况 A: 同阶相加】 (最常见)
            // K_res = K_lhs + K_rhs
            // 没有任何乘法，纯加法，速度最快
            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                res_keys[i] = add_mod(lhs_keys[i], rhs_keys[i]);
            }
        } 
        else if (offset_lhs > 0 && offset_rhs == 0) {
            // 【情况 B: LHS 低阶，需移位】
            // K_res = K_lhs * delta^diff + K_rhs
            uint64_t scale = delta_pow(local_delta, offset_lhs);
            
            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                uint64_t term_lhs = mult_mod(lhs_keys[i], scale);
                res_keys[i] = add_mod(term_lhs, rhs_keys[i]);
            }
        } 
        else if (offset_lhs == 0 && offset_rhs > 0) {
            // 【情况 C: RHS 低阶，需移位】
            // K_res = K_lhs + K_rhs * delta^diff
            uint64_t scale = delta_pow(local_delta, offset_rhs);

            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                uint64_t term_rhs = mult_mod(rhs_keys[i], scale);
                res_keys[i] = add_mod(lhs_keys[i], term_rhs);
            }
        } 
        else {
            // 【情况 D: 两者都低阶】 (罕见，但也可能发生)
            uint64_t scale_lhs = delta_pow(local_delta, offset_lhs);
            uint64_t scale_rhs = delta_pow(local_delta, offset_rhs);

            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                uint64_t term_lhs = mult_mod(lhs_keys[i], scale_lhs);
                uint64_t term_rhs = mult_mod(rhs_keys[i], scale_rhs);
                res_keys[i] = add_mod(term_lhs, term_rhs);
            }
        }

        // 6. 状态管理
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        res.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;

        return res; // RVO
    }

    PolyDelta sub(const PolyDelta& lhs, const PolyDelta& rhs) override {
        // 【逻辑留空】：实现多项式减法 lhs-rhs
        PolyDelta res;
        res.degree = std::max(lhs.degree, rhs.degree);
        
        size_t offset_lhs = res.degree - lhs.degree;
        size_t offset_rhs = res.degree - rhs.degree;

        // 优化核心：同阶相加时，直接相加，跳过所有乘法
        uint64_t k1 = lhs.key;
        if (offset_lhs > 0) {
            // 只有需要移位时才调用乘法
            // 此时 pow_mod(..., 1) 会很快返回，但 mult_mod 必须执行
            k1 = mult_mod(k1, delta_pow(this->delta, offset_lhs));
        }

        uint64_t k2 = rhs.key;
        if (offset_rhs > 0) {
            k2 = mult_mod(k2, delta_pow(this->delta, offset_rhs));
        }

        res.key = add_mod(k1, (PR - k2));
        
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        return res;
    }

    PolyTensor sub(const PolyTensor& lhs, const PolyTensor& rhs) override {
        // 1. 维度检查
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("Tensor shape mismatch in verifier sub!");
            exit(-1);
        }

        // 2. 构造结果
        int res_degree = std::max(lhs.degree, rhs.degree);
        PolyTensor res(lhs.shape, res_degree);
        
        size_t size = lhs.total_elements;

        // 3. 获取指针
        const uint64_t* lhs_keys = lhs.get_keys_ptr();
        const uint64_t* rhs_keys = rhs.get_keys_ptr();
        uint64_t* res_keys = res.get_keys_ptr();

        // 4. 准备参数
        uint64_t local_delta = (uint64_t)this->delta;
        int offset_lhs = res_degree - lhs.degree;
        int offset_rhs = res_degree - rhs.degree;

        // 5. 分情况并行计算 (K_res = K_lhs - K_rhs)
        
        if (offset_lhs == 0 && offset_rhs == 0) {
            // 【情况 A: 同阶相减】 (最快)
            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                res_keys[i] = add_mod(lhs_keys[i], PR - rhs_keys[i]);
            }
        } 
        else if (offset_lhs > 0 && offset_rhs == 0) {
            // 【情况 B: LHS 低阶】 K_new = (K_lhs * scale) - K_rhs
            uint64_t scale = delta_pow(local_delta, offset_lhs);
            
            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                uint64_t term_lhs = mult_mod(lhs_keys[i], scale);
                res_keys[i] = add_mod(term_lhs, PR - rhs_keys[i]);
            }
        } 
        else if (offset_lhs == 0 && offset_rhs > 0) {
            // 【情况 C: RHS 低阶】 K_new = K_lhs - (K_rhs * scale)
            uint64_t scale = delta_pow(local_delta, offset_rhs);

            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                uint64_t term_rhs = mult_mod(rhs_keys[i], scale);
                res_keys[i] = add_mod(lhs_keys[i], PR - term_rhs);
            }
        } 
        else {
            // 【情况 D: 两者都低阶】
            uint64_t scale_lhs = delta_pow(local_delta, offset_lhs);
            uint64_t scale_rhs = delta_pow(local_delta, offset_rhs);

            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                uint64_t term_lhs = mult_mod(lhs_keys[i], scale_lhs);
                uint64_t term_rhs = mult_mod(rhs_keys[i], scale_rhs);
                res_keys[i] = add_mod(term_lhs, PR - term_rhs);
            }
        }

        // 6. 状态管理
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        res.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;

        return res; 
    }

    PolyDelta mul(const PolyDelta& lhs, const PolyDelta& rhs) override {
        // 【逻辑留空】：实现多项式乘法 (卷积 Convolution)
        // Verifer side multiplciation
        PolyDelta res;

        res.degree = lhs.degree + rhs.degree;

        res.key = mult_mod(lhs.key, rhs.key);

        if (res.key > PR) LOG_ERROR("keys! > PR (mul)");

        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;

        return res;
    }

    PolyTensor mul(const PolyTensor& lhs, const PolyTensor& rhs) override {
        // 1. 维度检查
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("Tensor shape mismatch in verifier mul!");
            exit(-1);
        }

        // 2. 构造结果 (阶数相加)
        int res_degree = lhs.degree + rhs.degree;
        PolyTensor res(lhs.shape, res_degree);
        
        size_t size = lhs.total_elements;

        // 3. 获取指针
        const uint64_t* lhs_keys = lhs.get_keys_ptr();
        const uint64_t* rhs_keys = rhs.get_keys_ptr();
        uint64_t* res_keys = res.get_keys_ptr();

        // 4. 并行乘法
        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            res_keys[i] = mult_mod(lhs_keys[i], rhs_keys[i]);
        }

        // 5. 状态管理
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        res.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;

        return res;
    }

    void add_assign(PolyDelta& lhs, const PolyDelta& rhs) override {
        if (rhs.degree > lhs.degree){
            size_t offset_diff = rhs.degree - lhs.degree;
            lhs.key = add_mod(rhs.key, mult_mod(lhs.key, delta_pow(this->delta, offset_diff)));
            lhs.degree = rhs.degree;
        }else{
            size_t offset_diff = lhs.degree - rhs.degree;
            lhs.key = add_mod(lhs.key, mult_mod(rhs.key, delta_pow(this->delta, offset_diff)));
        }

        lhs.is_consumed = false;
        rhs.is_consumed = true;
    }

    void add_assign(PolyTensor& lhs, const PolyTensor& rhs) override {
        // 1. 维度检查
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("Tensor shape mismatch in verifier add_assign!");
            exit(-1);
        }

        size_t size = lhs.total_elements;
        uint64_t* lhs_keys = lhs.get_keys_ptr();
        const uint64_t* rhs_keys = rhs.get_keys_ptr();
        
        uint64_t local_delta = (uint64_t)this->delta;

        if (rhs.degree > lhs.degree) {
            // 【情况 A: LHS 升阶】 (LHS degree < RHS degree)
            // 公式: K_new = K_lhs * delta^(diff) + K_rhs
            // 意味着原有的 LHS 数据相当于变成了高阶项，需要乘 Delta
            
            int offset = rhs.degree - lhs.degree;
            uint64_t scale = delta_pow(local_delta, offset);

            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                // 原地更新： lhs[i] = lhs[i] * scale + rhs[i]
                uint64_t shifted_lhs = mult_mod(lhs_keys[i], scale);
                lhs_keys[i] = add_mod(shifted_lhs, rhs_keys[i]);
            }
            
            // 【关键】更新阶数
            lhs.degree = rhs.degree;

        } else {
            // 【情况 B: LHS 吞噬 RHS】 (LHS degree >= RHS degree)
            // 公式: K_new = K_lhs + K_rhs * delta^(diff)
            // LHS 保持原位，RHS 相当于加到了低阶部分，需要乘 Delta 对齐
            
            int offset = lhs.degree - rhs.degree;

            if (offset == 0) {
                // 优化：同阶直接加
                #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
                for (size_t i = 0; i < size; ++i) {
                    lhs_keys[i] = add_mod(lhs_keys[i], rhs_keys[i]);
                }
            } else {
                uint64_t scale = delta_pow(local_delta, offset);
                #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
                for (size_t i = 0; i < size; ++i) {
                    uint64_t shifted_rhs = mult_mod(rhs_keys[i], scale);
                    lhs_keys[i] = add_mod(lhs_keys[i], shifted_rhs);
                }
            }
        }

        // 状态更新
        lhs.is_consumed = false;
        rhs.is_consumed = true;
        lhs.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;
    }

    void sub_assign(PolyDelta& lhs, const PolyDelta& rhs) override {
        if (rhs.degree > lhs.degree){
            size_t offset_diff = rhs.degree - lhs.degree;
            lhs.key = add_mod((PR - rhs.key), mult_mod(lhs.key, delta_pow(this->delta, offset_diff)));
            lhs.degree = rhs.degree;
        }else{
            size_t offset_diff = lhs.degree - rhs.degree;
            lhs.key = add_mod(lhs.key, (PR - mult_mod(rhs.key, delta_pow(this->delta, offset_diff))));
        }

        lhs.is_consumed = false;
        rhs.is_consumed = true;
    }

    void sub_assign(PolyTensor& lhs, const PolyTensor& rhs) override {
        // 1. 维度检查
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("Tensor shape mismatch in verifier add_assign!");
            exit(-1);
        }

        size_t size = lhs.total_elements;
        uint64_t* lhs_keys = lhs.get_keys_ptr();
        const uint64_t* rhs_keys = rhs.get_keys_ptr();
        
        uint64_t local_delta = (uint64_t)this->delta;

        if (rhs.degree > lhs.degree) {
            // 【情况 A: LHS 升阶】 (LHS degree < RHS degree)
            // 公式: K_new = K_lhs * delta^(diff) + K_rhs
            // 意味着原有的 LHS 数据相当于变成了高阶项，需要乘 Delta
            
            int offset = rhs.degree - lhs.degree;
            uint64_t scale = delta_pow(local_delta, offset);

            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                // 原地更新： lhs[i] = lhs[i] * scale + rhs[i]
                uint64_t shifted_lhs = mult_mod(lhs_keys[i], scale);
                lhs_keys[i] = add_mod(shifted_lhs, PR - rhs_keys[i]);
            }
            
            // 【关键】更新阶数
            lhs.degree = rhs.degree;

        } else {
            // 【情况 B: LHS 吞噬 RHS】 (LHS degree >= RHS degree)
            // 公式: K_new = K_lhs + K_rhs * delta^(diff)
            // LHS 保持原位，RHS 相当于加到了低阶部分，需要乘 Delta 对齐
            
            int offset = lhs.degree - rhs.degree;

            if (offset == 0) {
                // 优化：同阶直接加
                #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
                for (size_t i = 0; i < size; ++i) {
                    lhs_keys[i] = add_mod(lhs_keys[i], PR - rhs_keys[i]);
                }
            } else {
                uint64_t scale = delta_pow(local_delta, offset);
                #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
                for (size_t i = 0; i < size; ++i) {
                    uint64_t shifted_rhs = mult_mod(rhs_keys[i], scale);
                    lhs_keys[i] = add_mod(lhs_keys[i], PR - shifted_rhs);
                }
            }
        }

        // 状态更新
        lhs.is_consumed = false;
        rhs.is_consumed = true;
        lhs.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;
    }

    void mul_assign(PolyDelta& lhs, uint64_t val) override {
        lhs.key = mult_mod(lhs.key, val);

        if (lhs.key > PR) LOG_WARN("keys! > PR (mul assign)");

        lhs.is_consumed = false;
    }

    void mul_assign(PolyTensor& lhs, uint64_t val) override {
        size_t size = lhs.total_elements;
        uint64_t* keys_ptr = lhs.get_keys_ptr();

        // 直接更新所有 Key
        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            keys_ptr[i] = mult_mod(keys_ptr[i], val);
        }

        // 状态：结果是活跃的
        lhs.is_consumed = false;
    }

    void mul_assign(PolyTensor& lhs, const PolyTensor& rhs) override {
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("Tensor shape mismatch in verifier mul_assign!");
            exit(-1);
        }

        size_t size = lhs.total_elements;
        uint64_t* lhs_keys = lhs.get_keys_ptr();
        const uint64_t* rhs_keys = rhs.get_keys_ptr();

        // 1. 更新 Keys
        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            lhs_keys[i] = mult_mod(lhs_keys[i], rhs_keys[i]);
        }

        // 2. 更新阶数
        lhs.degree += rhs.degree;

        // 3. 状态管理
        lhs.is_consumed = false;
        rhs.is_consumed = true;
        lhs.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;
    }

    void add_assign_const(PolyDelta& lhs, uint64_t val) override {
        // K' = K + val * delta^deg
        uint64_t shift_val = mult_mod(val, delta_pow(this->delta, lhs.degree));
        lhs.key = add_mod(lhs.key, shift_val);

        lhs.is_consumed = false;
    }

    void add_assign_const(PolyTensor& lhs, uint64_t val) override {
        size_t size = lhs.total_elements;
        uint64_t* keys_ptr = lhs.get_keys_ptr();

        // 1. 预计算 Shift 值 (标量)
        // Shift = val * Delta^degree
        // 这个计算只做一次，极其廉价
        uint64_t shift = mult_mod(val, delta_pow(this->delta, lhs.degree));
        
        // 2. 并行更新所有 Key
        // K_new = K_old + Shift
        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            keys_ptr[i] = add_mod(keys_ptr[i], shift);
        }

        lhs.is_consumed = false;
    }

    void sub_assign_const(PolyDelta& lhs, uint64_t val) override {
        // K' = K - val * delta^deg
        uint64_t shift_val = mult_mod(val, delta_pow(this->delta, lhs.degree));
        
        // 模减法: a - b -> a + (PR - b)
        uint64_t neg_shift_val = (shift_val == 0) ? 0 : (PR - shift_val);
        lhs.key = add_mod(lhs.key, neg_shift_val);

        lhs.is_consumed = false;
    }

    // =========================================================
    // 3. 常数运算接口实现 (PolyDelta vs Constant)
    // =========================================================

    PolyDelta add_const(const PolyDelta& lhs, uint64_t val) override {
        // 【逻辑留空】：多项式加常数 
        // Verifier side: K' = K + c * delta ^n
        if (lhs.degree == 0){
            LOG_WARN("add_const called on degree 0 PolyDelta! Potential circuit design issue.");
        }

        PolyDelta res;

        res = lhs.clone();
        auto delta_powers_n = delta_pow(this->delta, lhs.degree);
        auto val_mult_delta_powers_n = mult_mod(val, delta_powers_n);
        res.key = add_mod(res.key, val_mult_delta_powers_n);

        res.is_consumed = false;
        lhs.is_consumed = true;

        return res;
    }

    PolyDelta sub_const(const PolyDelta& lhs, uint64_t val) override {
        // 【逻辑留空】：多项式减常数 x-val
        // Verifier site: nearly identical to add const
        if (lhs.degree == 0){
            LOG_WARN("sub_const called on degree 0 PolyDelta! Potential circuit design issue.");
        }

        PolyDelta res;

        res = lhs.clone();
        auto delta_powers_n = delta_pow(this->delta, lhs.degree);
        auto val_mult_delta_powers_n = mult_mod(val, delta_powers_n);
        res.key = add_mod(res.key, (PR - val_mult_delta_powers_n));

        res.is_consumed = false;
        lhs.is_consumed = true;

        return res;
    }

    PolyDelta sub_const_rev(uint64_t val, const PolyDelta& rhs) override {
        // 【逻辑留空】：常数减多项式 (val - Poly), val - x
        // Verifier site, K'=-K+c*delta^n
        if (rhs.degree == 0){
            LOG_WARN("sub_const_rev called on degree 0 PolyDelta! Potential circuit design issue.");
        }

        PolyDelta res;

        res = rhs.clone();
        auto delta_powers_n = delta_pow(this->delta, rhs.degree);
        auto val_mult_delta_powers_n = mult_mod(val, delta_powers_n);
        res.key = add_mod((PR - res.key), val_mult_delta_powers_n);

        res.is_consumed = false;
        rhs.is_consumed = true;

        return res;
    }

    PolyDelta mul_const(const PolyDelta& lhs, uint64_t val) override {
        // 【逻辑留空】：多项式乘常数 (所有系数乘以 val), x*c
        PolyDelta res;

        res = lhs.clone();
        res.key = mult_mod(res.key, val);

        if (res.key > PR) {
            LOG_WARN("key > PR! (mul_const)");
            exit(-1);
        }
        

        res.is_consumed = false;
        lhs.is_consumed = true;

        return res;
    }

    PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs) override {
        // Matrix multiplication verifier 2d
        // 1. 维度检查
        if (lhs.shape.size() != 2 || rhs.shape.size() != 2 || lhs.shape[1] != rhs.shape[0]) {
            LOG_ERROR("Invalid shape for matmul!");
            exit(-1);
        }
        int M = lhs.shape[0];
        int K = lhs.shape[1];
        int N = rhs.shape[1];

        // 2. 构造结果 Z
        // 阶数必须与 Prover 保持一致 (DegA + DegB)，否则后续层无法对齐
        int res_degree = lhs.degree + rhs.degree;
        PolyTensor res({M, N}, res_degree); 
        // 构造函数已将 res.flat_keys 初始化为 0

        // 3. 计算 Keys (Z_Key = X_Key * Y_Key)
        // 这是一个标准的矩阵乘法。
        // 注意：Verifier 的 flat_keys 是未分层的 (如果按我们最后的讨论，Verifier 只有一个 Key 矩阵)
        // 所以直接拿首地址即可。
        
        matrix_mul_acc_kernel(
            res.get_keys_ptr(),  // Dst Key Matrix
            lhs.get_keys_ptr(),  // LHS Key Matrix
            rhs.get_keys_ptr(),  // RHS Key Matrix
            M, K, N
        );

        // 4. 状态管理
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        res.is_from_fresh_matmul = true;

        return res;
    }

    void permute_and_add_bias(
        const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
        int N, int H_out, int W_out, int C_out
    ) override {
        bool has_bias = (bias.total_elements > 0);

        if(src.degree < bias.degree){
            LOG_WARN("Non-implemented with bias.deg > WX.deg at Conv2D!");
        }

        // Keys: 高阶对齐 (Shift with Delta)
        if (!src.flat_keys.empty()) {
            const uint64_t* b_ptr = has_bias ? bias.get_keys_ptr() : nullptr;
            
            uint64_t scale = 1;
            
            // 计算 offset 和 scale
            if (has_bias && src.degree > bias.degree) {
                int offset = src.degree - bias.degree;
                scale = delta_pow(this->delta, offset);
            }
            // 如果 bias.degree > src.degree，根据你的 add_assign 逻辑，
            // 应该是 src 去迁就 bias (src 升阶)。
            // 但 Conv2D 语义限定了 Output 阶数由 Input*Weight 决定。
            // 这种情况下我们暂且假设 bias 阶数不会超过 src，或者不做处理。

            permute_and_add_bias_kernel(
                src.get_keys_ptr(), dst.get_keys_ptr(), b_ptr, 
                scale, // 传入 Delta^offset
                N, H_out, W_out, C_out
            );
        }

        // Update bias flag
        src.is_consumed = true;
        bias.is_consumed = true;
        dst.is_consumed = false;
    }

    void helper_linear_add_bias(
        PolyTensor& data, const PolyTensor& bias, int Rows, int Cols
    ) override {
        bool has_bias = (bias.total_elements > 0);

        // Keys: 高阶对齐 (Delta Scale)
        if (!data.flat_keys.empty()) {
            const uint64_t* b_ptr = has_bias ? bias.get_keys_ptr() : nullptr;
            
            uint64_t scale = 1;
            if (has_bias && data.degree > bias.degree) {
                int offset = data.degree - bias.degree;
                scale = delta_pow(this->delta, offset);
            }

            if (b_ptr){
                add_bias_row_broadcast_kernel(data.get_keys_ptr(), b_ptr, scale, Rows, Cols);
            }
        }
    }

    void helper_permute_and_add_bias_1d(
        const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
        int N, int L_out, int C_out
    ) override {
        bool has_bias = (bias.total_elements > 0);

        // Keys: 高阶对齐 (Delta Scale)
        if (!src.flat_keys.empty()) {
            const uint64_t* b_ptr = has_bias ? bias.get_keys_ptr() : nullptr;
            
            uint64_t scale = 1;
            if (has_bias && src.degree > bias.degree) {
                int offset = src.degree - bias.degree;
                scale = delta_pow(this->delta, offset);
            }

            permute_and_add_bias_1d_kernel(
                src.get_keys_ptr(), dst.get_keys_ptr(), b_ptr, 
                scale, 
                N, L_out, C_out
            );
        }
        src.is_consumed = true;
    }

    PolyTensor relu(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) override {
        size_t n = x.total_elements;
        
        // 核心解耦：保持与 Prover 绝对一致的切分逻辑
        int s_0 = scale;
        int s_rest = (bitlen > (uint64_t)s_0 && digdec_k > 1) ? 
                     (bitlen - s_0 + digdec_k - 2) / (digdec_k - 1) : 0;
                     
        std::vector<uint64_t> dummy_zeros(n, 0);

        PolyTensor authenticated_bq = input(x.shape, dummy_zeros);
        PolyTensor b_Q_check = authenticated_bq * (authenticated_bq - 1);
        PolyTensor::store_zero_relation(b_Q_check);

        // 创建两个 Range Table
        uint64_t table_size_0 = 1ULL << s_0;
        std::vector<uint64_t> rangeTableData_0(table_size_0);
        for (size_t i = 0; i < table_size_0; i++) rangeTableData_0[i] = i;
        RangeCheckTable rangeTable_0(rangeTableData_0);

        uint64_t table_size_rest = 1ULL << s_rest;
        std::vector<uint64_t> rangeTableData_rest(table_size_rest);
        for (size_t i = 0; i < table_size_rest; i++) rangeTableData_rest[i] = i;
        RangeCheckTable rangeTable_rest(rangeTableData_rest);

        PolyTensor X_recon; 
        PolyTensor X_out;   

        for (int j = 0; j < digdec_k; ++j) {
            PolyTensor authenticated_seg_j = input(x.shape, dummy_zeros);
            
            if (j == 0) {
                rangeTable_0.range_check(authenticated_seg_j);
                X_recon = authenticated_seg_j.clone(); 
            } else {
                rangeTable_rest.range_check(authenticated_seg_j);
                
                // 1. 完整重构 
                uint64_t shift_recon = 1ULL << (s_0 + (j - 1) * s_rest);
                X_recon = X_recon + authenticated_seg_j * shift_recon;

                // 2. 截断重构 (降去 Scale 操作)
                if (do_truncation) {
                    uint64_t trunc_shift = 1ULL << ((j - 1) * s_rest);
                    if (j == 1) X_out = authenticated_seg_j * trunc_shift;
                    else        X_out = X_out + authenticated_seg_j * trunc_shift;
                }
            }
        }
        
        if (!do_truncation) {
            X_out = X_recon.clone();
        }

        // ======================================================
        // Step 6, 7, 8: Consistency Check (严格使用 X_recon)
        // ======================================================
        PolyTensor term1 = authenticated_bq * (X_recon - x);
        PolyTensor term2 = (1 - authenticated_bq) * (X_recon + x);

        PolyTensor consistency = term1 + term2;
        PolyTensor::store_zero_relation(consistency);

        // ======================================================
        // Step 9: Output (严格使用 X_out)
        // ======================================================
        PolyTensor result = authenticated_bq * X_out;

        return result;
    }

    [[deprecated("This relu should not be used as it fails to consider the truncation bit and the scale.")]]
    PolyTensor relu_legacy(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation) {
        size_t n = x.total_elements;
        int s = (bitlen + digdec_k - 1) / digdec_k;
        std::vector<uint64_t> dummy_zeros(n, 0);

        PolyTensor authenticated_bq = input(x.shape, dummy_zeros);
        PolyTensor b_Q_check = authenticated_bq * (authenticated_bq - 1);
        PolyTensor::store_zero_relation(b_Q_check);

        uint64_t table_size = 1ULL << s;
        std::vector<uint64_t> rangeTableData(table_size);
        for (size_t i = 0; i < table_size; i++) rangeTableData[i] = i;
        RangeCheckTable rangeTable(rangeTableData);

        PolyTensor X_recon; // 用于校验
        PolyTensor X_out;   // 用于输出

        for (int j = 0; j < digdec_k; ++j) {
            PolyTensor authenticated_seg_j = input(x.shape, dummy_zeros);
            rangeTable.range_check(authenticated_seg_j);
            
            // 1. 完整重构 (用于一致性校验)
            uint64_t shift_scale = 1ULL << (j * s);
            if (j == 0) X_recon = authenticated_seg_j * shift_scale;
            else        X_recon = X_recon + authenticated_seg_j * shift_scale;

            // 2. 截断重构 (降 Scale 操作)
            if (do_truncation && j > 0) {
                // 原来的权重是 j*s，现在整体降阶变成 (j-1)*s
                uint64_t trunc_shift = 1ULL << ((j - 1) * s);
                if (j == 1) X_out = authenticated_seg_j * trunc_shift;
                else        X_out = X_out + authenticated_seg_j * trunc_shift;
            }
        }
        
        if (!do_truncation) {
            X_out = X_recon.clone();
        }

        // ======================================================
        // Step 6, 7, 8: Consistency Check (严格使用 X_recon)
        // ======================================================
        PolyTensor term1 = authenticated_bq * (X_recon - x);
        PolyTensor term2 = (1 - authenticated_bq) * (X_recon + x);

        PolyTensor consistency = term1 + term2;
        PolyTensor::store_zero_relation(consistency);

        // ======================================================
        // Step 9: Output (严格使用 X_out)
        // ======================================================
        PolyTensor result = authenticated_bq * X_out;

        return result;
    }

    PolyTensor maxpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, uint64_t scale) override {
        int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
        int kH = kernel_size, kW = kernel_size;
        int H_out = (H_in + 2 * padding - kH) / stride + 1;
        int W_out = (W_in + 2 * padding - kW) / stride + 1;
        
        int M = N * C * H_out * W_out; 
        int h = kH * kW;               

        std::vector<uint64_t> dummy_max(M, 0);
        PolyTensor pt_max = this->input({N, C, H_out, W_out}, dummy_max);

        //  解耦参数计算
        int s_0 = scale; 
        int s_rest = (bitlen > (uint64_t)s_0 && digdec_k > 1) ? 
                     (bitlen - s_0 + digdec_k - 2) / (digdec_k - 1) : 0;

        uint64_t table_size_0 = 1ULL << s_0;
        std::vector<uint64_t> rangeTableData_0(table_size_0);
        for (size_t i = 0; i < table_size_0; i++) rangeTableData_0[i] = i;
        RangeCheckTable rangeTable_0(rangeTableData_0);

        uint64_t table_size_rest = 1ULL << s_rest;
        std::vector<uint64_t> rangeTableData_rest(table_size_rest);
        for (size_t i = 0; i < table_size_rest; i++) rangeTableData_rest[i] = i;
        RangeCheckTable rangeTable_rest(rangeTableData_rest);

        PolyTensor pt_prod; 

        for (int j = 0; j < h; ++j) {
            int kh = j / kW;
            int kw = j % kW;

            PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree);

            #pragma omp parallel for collapse(4) schedule(static) if(M>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(int n=0; n<N; ++n) {
                for(int c=0; c<C; ++c) {
                    for(int ho=0; ho<H_out; ++ho) {
                        for(int wo=0; wo<W_out; ++wo) {
                            int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;

                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                pt_in_aligned.flat_keys[m] = pt_in.flat_keys[in_idx];
                            } else {
                                pt_in_aligned.flat_keys[m] = add_mod(pt_max.flat_keys[m], PR - delta_pow(this->delta, pt_in.degree));
                            }
                        }
                    }
                }
            }

            PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;

            // --- Upper Bound Check ---
            PolyTensor X_recon;

            for (int d = 0; d < digdec_k; ++d) {
                PolyTensor auth_seg = this->input({M}, dummy_max);
                
                //  双轨检查与重构
                if (d == 0) {
                    rangeTable_0.range_check(auth_seg);
                    X_recon = auth_seg.clone(); 
                } else {
                    rangeTable_rest.range_check(auth_seg);
                    uint64_t shift_recon = 1ULL << (s_0 + (d - 1) * s_rest);
                    X_recon = X_recon + auth_seg * shift_recon;
                }
            }
            
            PolyTensor::store_relation(X_recon, pt_y_hat_j);

            // --- Existence Check ---
            if (j == 0) {
                pt_prod = pt_y_hat_j.clone();
                pt_y_hat_j.is_consumed = true;
            } else {
                pt_prod = pt_prod * pt_y_hat_j;
            }
        }

        PolyTensor::store_zero_relation(pt_prod);

        pt_in.mark_consumed();

        return pt_max;
    }

    [[deprecated("This maxpool should not be used as it fails to consider the truncation bit and the scale.")]]
    PolyTensor maxpool2d_legacy(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k) {
        int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
        int kH = kernel_size, kW = kernel_size;
        int H_out = (H_in + 2 * padding - kH) / stride + 1;
        int W_out = (W_in + 2 * padding - kW) / stride + 1;
        
        int M = N * C * H_out * W_out; 
        int h = kH * kW;               

        std::vector<uint64_t> dummy_max(M, 0);
        PolyTensor pt_max = this->input({N, C, H_out, W_out}, dummy_max);

        int s = (bitlen + digdec_k - 1) / digdec_k;
        uint64_t table_size = 1ULL << s;
        std::vector<uint64_t> rangeTableData(table_size);
        for (size_t i = 0; i < table_size; i++) rangeTableData[i] = i;
        RangeCheckTable rangeTable(rangeTableData);

        PolyTensor pt_prod; 

        for (int j = 0; j < h; ++j) {
            int kh = j / kW;
            int kw = j % kW;

            PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree);

            #pragma omp parallel for collapse(4) schedule(static) if(M>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(int n=0; n<N; ++n) {
                for(int c=0; c<C; ++c) {
                    for(int ho=0; ho<H_out; ++ho) {
                        for(int wo=0; wo<W_out; ++wo) {
                            int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;

                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                // 【Valid】: 纯粹搬运 Key
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                pt_in_aligned.flat_keys[m] = pt_in.flat_keys[in_idx];
                            } else {
                                // 【Padding】: K_in = K_max - Delta^(pt_in.degree)
                                // 使得 K_max - K_in 完美等于 Delta^(pt_in.degree)
                                pt_in_aligned.flat_keys[m] = add_mod(pt_max.flat_keys[m], PR - delta_pow(this->delta, pt_in.degree));
                            }
                        }
                    }
                }
            }

            PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;

            // --- Upper Bound Check ---
            PolyTensor X_recon;

            for (int d = 0; d < digdec_k; ++d) {
                PolyTensor auth_seg = this->input({M}, dummy_max);
                rangeTable.range_check(auth_seg);

                uint64_t shift_scale = 1ULL << (d * s);
                if (d == 0) X_recon = auth_seg * shift_scale;
                else        X_recon = X_recon + auth_seg * shift_scale;
            }
            
            //PolyTensor pt_y_for_bound = pt_y_hat_j.clone();
            PolyTensor::store_relation(X_recon, pt_y_hat_j);

            // --- Existence Check ---
            if (j == 0) {
                pt_prod = pt_y_hat_j.clone();
                pt_y_hat_j.is_consumed = true;
            } else {
                pt_prod = pt_prod * pt_y_hat_j;
            }
        }

        PolyTensor::store_zero_relation(pt_prod);

        pt_in.mark_consumed();

        return pt_max;
    }

    PolyTensor integrated_nl(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) override {
        int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
        int kH = kernel_size, kW = kernel_size;
        int H_out = (H_in + 2 * padding - kH) / stride + 1;
        int W_out = (W_in + 2 * padding - kW) / stride + 1;
        
        int M = N * C * H_out * W_out; 
        int h = kH * kW;               

        std::vector<uint64_t> dummy_max(M, 0);
        PolyTensor pt_max = this->input({N, C, H_out, W_out}, dummy_max);

        // 解耦参数计算
        int s_0 = scale; 
        int s_rest = (bitlen > (uint64_t)s_0 && digdec_k > 1) ? 
                     (bitlen - s_0 + digdec_k - 2) / (digdec_k - 1) : 0;

        uint64_t table_size_0 = 1ULL << s_0;
        std::vector<uint64_t> rangeTableData_0(table_size_0);
        for (size_t i = 0; i < table_size_0; i++) rangeTableData_0[i] = i;
        RangeCheckTable rangeTable_0(rangeTableData_0);

        uint64_t table_size_rest = 1ULL << s_rest;
        std::vector<uint64_t> rangeTableData_rest(table_size_rest);
        for (size_t i = 0; i < table_size_rest; i++) rangeTableData_rest[i] = i;
        RangeCheckTable rangeTable_rest(rangeTableData_rest);

        // ======================================================
        // Step 3 & 9: Range Check x_max & Reconstruct Output
        // ======================================================
        PolyTensor X_max_recon;
        PolyTensor X_max_out;

        for (int d = 0; d < digdec_k; ++d) {
            PolyTensor auth_seg = this->input({M}, dummy_max);
            
            if (d == 0) {
                rangeTable_0.range_check(auth_seg);
                X_max_recon = auth_seg.clone();
            } else {
                rangeTable_rest.range_check(auth_seg);
                uint64_t shift_recon = 1ULL << (s_0 + (d - 1) * s_rest);
                X_max_recon = X_max_recon + auth_seg * shift_recon;

                if (do_truncation) {
                    uint64_t trunc_shift = 1ULL << ((d - 1) * s_rest);
                    if (d == 1) X_max_out = auth_seg * trunc_shift;
                    else        X_max_out = X_max_out + auth_seg * trunc_shift;
                }
            }
        }
        if (!do_truncation) X_max_out = X_max_recon.clone();
        
        PolyTensor::store_relation(X_max_recon, pt_max);

        // ======================================================
        // Step 4 - 8: Loop over sub-matrix elements
        // ======================================================
        PolyTensor pt_prod = pt_max.clone(); 

        for (int j = 0; j < h; ++j) {
            int kh = j / kW;
            int kw = j % kW;

            PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree);

            #pragma omp parallel for collapse(4) schedule(static) if(M>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(int n=0; n<N; ++n) {
                for(int c=0; c<C; ++c) {
                    for(int ho=0; ho<H_out; ++ho) {
                        for(int wo=0; wo<W_out; ++wo) {
                            int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;

                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                pt_in_aligned.flat_keys[m] = pt_in.flat_keys[in_idx];
                            } else {
                                pt_in_aligned.flat_keys[m] = add_mod(pt_max.flat_keys[m], PR - delta_pow(this->delta, pt_in.degree));
                            }
                        }
                    }
                }
            }

            PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;
            PolyTensor X_recon;

            for (int d = 0; d < digdec_k; ++d) {
                PolyTensor auth_seg = this->input({M}, dummy_max);
                
                if (d == 0) {
                    rangeTable_0.range_check(auth_seg);
                    X_recon = auth_seg.clone();
                } else {
                    rangeTable_rest.range_check(auth_seg);
                    uint64_t shift_recon = 1ULL << (s_0 + (d - 1) * s_rest);
                    X_recon = X_recon + auth_seg * shift_recon;
                }
            }
            
            PolyTensor::store_relation(X_recon, pt_y_hat_j);
            pt_prod = pt_prod * pt_y_hat_j;
        }

        PolyTensor::store_zero_relation(pt_prod);
        pt_in.mark_consumed();

        X_max_out.shape = {N, C, H_out, W_out};
        return X_max_out;
    }
    
    [[deprecated("This integrated_nl should not be used as it fails to consider the truncation bit and the scale.")]]
    PolyTensor integrated_nl_legacy(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, bool do_truncation) {
        int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
        int kH = kernel_size, kW = kernel_size;
        int H_out = (H_in + 2 * padding - kH) / stride + 1;
        int W_out = (W_in + 2 * padding - kW) / stride + 1;
        
        int M = N * C * H_out * W_out; 
        int h = kH * kW;               

        std::vector<uint64_t> dummy_max(M, 0);
        PolyTensor pt_max = this->input({N, C, H_out, W_out}, dummy_max);

        int s = (bitlen + digdec_k - 1) / digdec_k;
        uint64_t table_size = 1ULL << s;
        std::vector<uint64_t> rangeTableData(table_size);
        for (size_t i = 0; i < table_size; i++) rangeTableData[i] = i;
        RangeCheckTable rangeTable(rangeTableData);

        // ======================================================
        // Step 3 & 9: Range Check x_max & Reconstruct Output
        // ======================================================
        PolyTensor X_max_recon;
        PolyTensor X_max_out;

        for (int d = 0; d < digdec_k; ++d) {
            PolyTensor auth_seg = this->input({M}, dummy_max);
            rangeTable.range_check(auth_seg);

            uint64_t shift_scale = 1ULL << (d * s);
            if (d == 0) X_max_recon = auth_seg * shift_scale;
            else        X_max_recon = X_max_recon + auth_seg * shift_scale;

            if (do_truncation && d > 0) {
                uint64_t trunc_shift = 1ULL << ((d - 1) * s);
                if (d == 1) X_max_out = auth_seg * trunc_shift;
                else        X_max_out = X_max_out + auth_seg * trunc_shift;
            }
        }
        if (!do_truncation) X_max_out = X_max_recon.clone();
        
        PolyTensor::store_relation(X_max_recon, pt_max);

        // ======================================================
        // Step 4 - 8: Loop over sub-matrix elements
        // ======================================================
        PolyTensor pt_prod = pt_max.clone(); // d_0 = x_max

        for (int j = 0; j < h; ++j) {
            int kh = j / kW;
            int kw = j % kW;

            PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree);

            #pragma omp parallel for collapse(4) schedule(static)
            for(int n=0; n<N; ++n) {
                for(int c=0; c<C; ++c) {
                    for(int ho=0; ho<H_out; ++ho) {
                        for(int wo=0; wo<W_out; ++wo) {
                            int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;

                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                pt_in_aligned.flat_keys[m] = pt_in.flat_keys[in_idx];
                            } else {
                                pt_in_aligned.flat_keys[m] = add_mod(pt_max.flat_keys[m], PR - delta_pow(this->delta, pt_in.degree));
                            }
                        }
                    }
                }
            }

            PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;
            PolyTensor X_recon;

            for (int d = 0; d < digdec_k; ++d) {
                PolyTensor auth_seg = this->input({M}, dummy_max);
                rangeTable.range_check(auth_seg);

                uint64_t shift_scale = 1ULL << (d * s);
                if (d == 0) X_recon = auth_seg * shift_scale;
                else        X_recon = X_recon + auth_seg * shift_scale;
            }
            
            PolyTensor::store_relation(X_recon, pt_y_hat_j);
            pt_prod = pt_prod * pt_y_hat_j;
        }

        PolyTensor::store_zero_relation(pt_prod);
        pt_in.mark_consumed();

        X_max_out.shape = {N, C, H_out, W_out};
        return X_max_out;
    }

    PolyTensor avgpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding) override {
        int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
        int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
        int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;
        
        PolyTensor pt_out({N, C, H_out, W_out}, pt_in.degree);
        
        uint64_t pool_area = (uint64_t)(kernel_size * kernel_size);
        uint64_t inv_area = pow_mod(pool_area, PR - 2); 

        const uint64_t* in_keys = pt_in.get_keys_ptr();
        uint64_t* out_keys = pt_out.get_keys_ptr();

        #pragma omp parallel for collapse(4)
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                for(int ho=0; ho<H_out; ++ho) {
                    for(int wo=0; wo<W_out; ++wo) {
                        int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        
                        uint64_t sum_keys = 0;

                        for(int kh=0; kh<kernel_size; ++kh) {
                            for(int kw=0; kw<kernel_size; ++kw) {
                                int hi = ho * stride - padding + kh;
                                int wi = wo * stride - padding + kw;
                                
                                if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                    int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                    sum_keys = add_mod(sum_keys, in_keys[in_idx]);
                                }
                            }
                        }
                        
                        out_keys[out_idx] = mult_mod(sum_keys, inv_area);
                    }
                }
            }
        }
        
        pt_in.mark_consumed();
        return pt_out;
    }

    [[deprecated("WARNING: DO NOT USE Standalone BatchNorm2D. Please use offline Conv-BN folding instead.")]]
    PolyTensor batchnorm2d(PolyTensor& pt_in, const std::vector<uint64_t>& A, const std::vector<uint64_t>& B) override {
        int N = pt_in.shape[0], C = pt_in.shape[1], H = pt_in.shape[2], W = pt_in.shape[3];
        PolyTensor pt_out({N, C, H, W}, pt_in.degree);

        const uint64_t* in_keys = pt_in.get_keys_ptr();
        uint64_t* out_keys = pt_out.get_keys_ptr();

        #pragma omp parallel for collapse(2)
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                uint64_t a_val = A[c]; // 直接从 vector 读取
                uint64_t b_val = B[c];
                
                // Verifier 将常数提升到对应阶数 Delta^D
                uint64_t b_scaled = mult_mod(b_val, delta_pow(this->delta, pt_in.degree));

                for(int h=0; h<H; ++h) {
                    for(int w=0; w<W; ++w) {
                        size_t idx = ((size_t)n * C + c) * H * W + h * W + w;
                        out_keys[idx] = add_mod(mult_mod(in_keys[idx], a_val), b_scaled);
                    }
                }
            }
        }
        //debug_instant_check(pt_out);
        pt_in.mark_consumed();
        return pt_out;
    }

    // =========================================================
    // 4. 输入接口实现
    // =========================================================

    void input(std::vector<PolyDelta>& pdList, const std::vector<uint64_t>& raw_data) override {
        // Here raw_data will not be used.

        size_t len = pdList.size();
        if (len == 0) {
            LOG_ERROR("Verifier input list must be pre-sized!");
            exit(-1);
        }
        
        std::vector<__uint128_t> vole_returned(len);
        std::vector<uint64_t> lam(len);

        vole->extend(vole_returned.data(), len);

        io->recv_data(lam.data(), len * sizeof(uint64_t));

        for (size_t i = 0; i < len; i++){
            uint64_t delta_mult_u_add_x = mult_mod(this->delta, lam[i]);
            pdList[i].key = add_mod(vole_returned[i], delta_mult_u_add_x);
            pdList[i].degree = 1;
            pdList[i].real_val = 0;
            pdList[i].is_consumed = false;
        }
    }

    PolyDelta input(uint64_t raw_data) override {
        PolyDelta res;
        __uint128_t vole_returned;
        uint64_t lam;

        vole->extend(&vole_returned, 1);

        io->recv_data(&lam, sizeof(uint64_t));
        uint64_t delta_mult_u_add_x = mult_mod(this->delta, lam);
        res.key = add_mod(vole_returned, delta_mult_u_add_x);
        res.degree = 1;
        res.real_val = 0;
        res.is_consumed = false;

        return res;
    }

    std::vector<PolyDelta> extend_vope_from_vole(uint64_t deg, int size = 1) override {
        // If we are asked to pre-generate a deg-n VOPE, where deg=1 -> standard VOLE,
        // then we will need the VOLE triple size to be 2 * (n-1) + 1 = 2 * n - 1
        assert(size > 0 && deg > 0);
        
        // 【逻辑正确】总共需要的 VOLE 数量
        int each_vole_size = (2 * deg - 1);
        int vole_size = size * each_vole_size;

        std::vector<PolyDelta> res;
        res.resize(size);
        std::vector<__uint128_t> vole_returned; 
        vole_returned.resize(vole_size);

        this->vole->extend(vole_returned.data(), vole_size);

        //std::cout << "[Verifier] Verifier naive VOLE res:";
        //std::cout << "HIGH64 = " << HIGH64(vole_returned[0]) << ", LOW64 = " << LOW64(vole_returned[0]) << std::endl;

        // 1. 去掉 move，利用 RVO 优化
        std::vector<PolyDelta> from_vole_returned = vole2pd(vole_returned, vole_size);
        
        for (int i = 0; i < size; i++) {
            // Base offset for this batch
            int base_offset = i * each_vole_size;

            // 2. Base term (Index 0)
            res[i] = std::move(from_vole_returned[base_offset]);
            
            // 【逻辑正确】循环 deg-1 次
            for (int j = 0; j < deg - 1; j++) {
                // 3. 乘法项与加法项
                // Mul Index: base + 1 + j
                // Add Index: base + deg + j  <--- 【修正点】这里原来是 1 + deg + j，多加了1
                
                res[i] = res[i] * from_vole_returned[base_offset + 1 + j] 
                         + from_vole_returned[base_offset + deg + j];
            }
            res[i].is_consumed = false;
            res[i].is_pre_generated = true;
        }
        return res;
    }

    PolyTensor input(const std::vector<int>& shape, const std::vector<uint64_t>& raw_data) override { // Verifier 版本没有 raw_data
        // The second varibale raw data will not be used in the verifier side, just set it to zero.
        
        // 1. 初始化 Tensor
        PolyTensor res(shape, 1);
        size_t size = res.total_elements;

        // 2. 准备缓冲区
        std::vector<__uint128_t> vole_returned(size);
        std::vector<uint64_t> lams(size); // 接收缓冲区

        // 3. 批量执行 VOLE
        vole->extend(vole_returned.data(), size);

        // 4. 批量接收 Mask
        io->recv_data(lams.data(), size * sizeof(uint64_t));

        // 5. 获取指针
        uint64_t* keys = res.get_keys_ptr();
        
        // 缓存 delta 到局部变量，可能有助于编译器优化
        uint64_t local_delta = (uint64_t)this->delta;

        // 6. 并行计算 Key (Parallel Compute)
        #pragma omp parallel for if (size>=MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            // 提取 Key (LOW64)
            uint64_t K = (uint64_t)LOW64(vole_returned[i]);
            uint64_t lam = lams[i];

            // 计算 delta * lam
            uint64_t delta_term = mult_mod(local_delta, lam);

            // 更新 Key: K' = K + delta * (u + x)
            keys[i] = add_mod(K, delta_term);
            
            if (keys[i] > PR) LOG_WARN("keys! > PR (input)");
        }

        // Verifier 不需要填充 coeffs 和 real_vals (它们是 0)
        // 但为了安全，coeffs 已经被构造函数 resize 并置零了，这是对的。

        // 7. 标记状态
        res.is_consumed = false;
        res.is_from_fresh_matmul = false;

        return res;
    }

    // =========================================================
    // 5. Helper Function
    // =========================================================

    // 辅助函数：带缓存的 Delta 幂次计算
    // 参数说明：虽然 pow_mod 是 (a, deg)，但既然是 delta_pow，底数固定为 this->delta，
    // 所以这里只传 deg 即可 (或者为了接口一致性，你可以保留 a 但忽略它/断言它)
    uint64_t delta_pow(uint64_t a, uint64_t deg) {
        // 1. 缓存命中 (Cache Hit)
        // vector 的 size 代表它包含 [0, size-1] 的幂次
        assert(a == (uint64_t)this->delta);
        if (deg < delta_powers_cache.size()) {
            return delta_powers_cache[deg];
        }

        // 2. 缓存未命中 (Cache Miss) - 增量扩展
        // 记录旧的大小
        size_t old_size = delta_powers_cache.size();
        
        // 扩展 vector 空间 (新分配的空间会被初始化为 0，但我们马上会覆盖)
        delta_powers_cache.resize(deg + 1);

        // 3. 线性填充 (Sequential Fill)
        // 核心优化：利用上一阶的结果 * delta，比重新调 pow_mod 快得多
        // cache[i] = cache[i-1] * delta
        for (size_t i = old_size; i <= deg; ++i) {
            delta_powers_cache[i] = mult_mod(delta_powers_cache[i-1], this->delta);
        }

        // 4. 返回结果
        //std::cout << "[INFO] Currnt delta cache = ";
        //for (int i = 0; i < delta_powers_cache.size(); i++){
        //   std::cout << delta_powers_cache[i] << ". ";
        //}
        //std::cout << std::endl;
        return delta_powers_cache[deg];
    }

    std::vector<PolyDelta> vole2pd(std::vector<__uint128_t>& vole_data, int size) override {
        std::vector<PolyDelta> res;
        res.resize(size);
        for (int i = 0; i < size; i++){
            res[i].is_consumed = false;
            res[i].degree = 1;
            // Note: this step is to make K = M + delta * x, instead of M = K + delta * x
            res[i].key = add_mod((PR - LOW64(vole_data[i])), 0);
        }
        return res;
    }

    PolyDelta recv_poly() {
        PolyDelta res;

        // 1. 接收 Degree
        io->recv_data(&res.degree, sizeof(int));

        // 2. 【新增】接收 Real Value
        // 在 Check 阶段，Verifier 需要这个值来验证逻辑
        io->recv_data(&res.real_val, sizeof(uint64_t));

        // 3. 准备 Coefficients 内存
        size_t num_coeffs = res.degree + 1;
        res.coeffs.resize(num_coeffs);

        // 4. 接收 Coefficients
        io->recv_data(res.coeffs.data(), num_coeffs * sizeof(uint64_t));

        // 5. 初始化状态
        // Key 在这里通常是未知的(或者是本地持有的 secret)，如果是 Opening，Verifier 会自己计算
        res.key = 0;      
        
        // 设置为 true，防止它被错误地再次放入待验证队列
        res.is_consumed = true; 

        return res; 
    }

    // 批量接收函数不需要修改逻辑，因为它复用了 recv_poly
    // 它会自动调用上面更新过的 recv_poly，从而正确读取 real_val
    std::vector<PolyDelta> recv_poly_vector() {
        size_t count;
        io->recv_data(&count, sizeof(size_t));

        std::vector<PolyDelta> res;
        res.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            res.push_back(recv_poly()); 
        }

        return res;
    }

    bool verify_poly(PolyDelta& mine, PolyDelta& hers) {
        // This function check the validity of a received PolyDelta.
        // Hers from prover, have coefs and real value

        if (mine.degree != hers.degree){
            LOG_ERROR("Inconsistent degree for verification! Mine = " << mine.degree << ", hers = " << hers.degree);
            return false; 
        }
        if (hers.real_val != hers.coeffs[hers.degree]){
            LOG_ERROR("Invalid commitment from prover (real_val)! Real_value = " << hers.real_val << ", Highest coef = " << hers.coeffs[hers.degree]);
            return false; 
        }

        // If the two condition passed, we calculate.
        uint64_t res = 0;
        for (int i = 0; i < hers.coeffs.size(); i++){
            res = add_mod(res, mult_mod(hers.coeffs[i], delta_pow(this->delta, i)));
        }
        if (res != mine.key){
            LOG_ERROR("Commitment check failed! RES = " << res << ", Key = " << mine.key);
            if ((res % PR) == (mine.key % PR)){
                LOG_ERROR("MULT MOD ERROR....");
            }else{
                LOG_ERROR("REAL ERROR!!!!!");
            }
            return false; 
        }

        return true;
    }

    // 打印 PolyDelta (Verifier 视角: Key)
    void debug_print(const PolyDelta& pd, std::string name = "") override {
        std::cout << std::left;
        std::cout << "\033[36m[Verif ] " << std::setw(15) << name << "\033[0m"; // 青色
        std::cout << " (Deg=" << pd.degree << ") ";
        std::cout << "Delta=" << (uint64_t)this->delta << ", ";
        std::cout << "Key=" << pd.key << std::endl;
    }

    // 打印 PolyTensor (Verifier 视角: Key)
    void debug_print(const PolyTensor& pt, std::string name, int limit = 4) override {
        std::cout << "--------------------------------------------------------" << std::endl;
        size_t count = (size_t)limit < pt.total_elements ? (size_t)limit : pt.total_elements;

        for(size_t i=0; i<count; ++i) {
            std::string item_name = name + "[" + std::to_string(i) + "]";
            std::cout << "\033[36m[Verif ] " << std::setw(15) << item_name << "\033[0m"; // 青色
            std::cout << " (Deg=" << pt.degree << ") ";
            
            if (i < pt.flat_keys.size())
                std::cout << "Key=" << pt.flat_keys[i];
            else
                std::cout << "Key=ERR";
            std::cout << std::endl;
        }
        if (pt.total_elements > count) std::cout << " ... (" << (pt.total_elements - count) << " omitted)" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
    }

    void debug_print_recvd(const PolyDelta& pd, std::string name) {
        std::cout << std::left;
        std::cout << "\033[32m[Received] " << std::setw(15) << name << "\033[0m"; // 绿色
        std::cout << " (Deg=" << pd.degree << ") ";
        std::cout << "Val=" << std::setw(6) << pd.real_val << " | Poly = ";
        
        if (pd.coeffs.empty()) {
            std::cout << "(empty)";
        } else {
            for (size_t i = 0; i < pd.coeffs.size(); ++i) {
                if (i > 0) std::cout << " + ";
                std::cout << pd.coeffs[i];
                if (i == 1) std::cout << "x";
                else if (i > 1) std::cout << "x^" << i;
            }
        }
        std::cout << std::endl;
    }

    bool debug_instant_check(const PolyDelta& pd) override {
        YELLOW("[Verifier] Instant Check");
        debug_print(pd, "");
        PolyDelta recvd = recv_poly();
        WHITE("Received:");
        debug_print_recvd(recvd, "");
        // Compute
        uint64_t rhs = 0;
        for (size_t i = 0; i < recvd.coeffs.size(); i++){
            uint64_t this_term = mult_mod(recvd.coeffs[i], delta_pow(this->delta, i));
            rhs = add_mod(rhs, this_term);
        }
        std::cout << "RHS = " << rhs << ", LHS (Key) = " << pd.key << std::endl;
        bool res = (rhs == pd.key);
        if (res){
            GREEN("[Verifer] INSTANT CHECK PASS");
            return res;
        }
        else{
            RED("[VERIFIER] INSTANT CHECK FAILED");
            return res;
        }
    }

    bool debug_instant_check(const PolyTensor& pt) override {
        YELLOW("[Verifier] Instant Tensor Check");
        
        // 1. 接收元数据
        size_t recv_len;
        int recv_deg;
        io->recv_data(&recv_len, sizeof(size_t));
        io->recv_data(&recv_deg, sizeof(int));

        // 2. 基础检查
        if (recv_len != pt.total_elements) {
            RED("[Verifier] Size Mismatch!");
            std::cout << "Recv Len: " << recv_len << ", Local Len: " << pt.total_elements << std::endl;
            // 接收剩余数据防止 socket 错位，然后返回 false
            std::vector<uint64_t> trash(recv_len * (recv_deg + 1));
            io->recv_data(trash.data(), trash.size() * sizeof(uint64_t));
            return false;
        }

        // 3. 接收系数
        std::vector<uint64_t> recv_coeffs(recv_len * (recv_deg + 1));
        io->recv_data(recv_coeffs.data(), recv_coeffs.size() * sizeof(uint64_t));

        WHITE("Received Tensor Data, Verifying...");

        // 4. 预计算 Delta 的幂次 (Delta^0, Delta^1, ..., Delta^deg)
        std::vector<uint64_t> d_pows(recv_deg + 1);
        for(int d = 0; d <= recv_deg; ++d) {
            d_pows[d] = delta_pow(this->delta, d);
        }

        // 5. 逐个元素验证
        const uint64_t* local_keys = pt.get_keys_ptr(); // 获取本地的 K
        bool all_pass = true;
        int fail_count = 0;

        for (size_t i = 0; i < recv_len; ++i) {
            uint64_t rhs = 0;
            
            // 计算 RHS = Sum( Coeff[d][i] * Delta^d )
            for (int d = 0; d <= recv_deg; ++d) {
                // 定位系数：SoA 布局 -> [d * len + i]
                uint64_t coeff = recv_coeffs[d * recv_len + i];
                uint64_t term = mult_mod(coeff, d_pows[d]);
                rhs = add_mod(rhs, term);
            }

            // 比较: RHS (Calculated) == LHS (Local Key)
            if (rhs != local_keys[i]) {
                if (fail_count < 10) { // 防止错误太多刷屏
                    std::cout << "\033[31m[FAIL] Index " << i 
                              << " | Calc RHS=" << rhs 
                              << " != Local Key=" << local_keys[i] << "\033[0m" << std::endl;
                }
                all_pass = false;
                fail_count++;
            }
        }

        if (all_pass) {
            GREEN("[Verifier] INSTANT TENSOR CHECK PASS");
            return true;
        } else {
            RED("[Verifier] INSTANT TENSOR CHECK FAILED");
            std::cout << "Total Failures: " << fail_count << " / " << recv_len << std::endl;
            return false;
        }
    }

protected:
    // --- Helper 1: 处理 PolyDelta ---
    void accumulate_delta_buffer(PolyDelta& final_item) override {
        if (check_buffer.empty()) return;

        size_t size = check_buffer.size();
        std::vector<uint64_t> chi(size);
        this->prg.random_data(chi.data(), size * sizeof(uint64_t));

        for (size_t i = 0; i < size; i++) {
            chi[i] = chi[i] % PR;
            final_item += chi[i] * check_buffer[i];
            check_buffer[i].is_consumed = true;
        }
        check_buffer.clear();
    }

    // --- Helper 2: 处理 PolyTensor (只处理 Keys) ---
    void accumulate_tensor_buffer(PolyDelta& final_item) override {
        if (check_tensor_buffer.empty()) return;

        for (auto& tensor : check_tensor_buffer) {
            size_t len = tensor.total_elements;
            
            // 1. 生成随机数
            std::vector<uint64_t> chi_vec(len);
            this->prg.random_data(chi_vec.data(), len * sizeof(uint64_t));

            #pragma omp parallel for if (len >= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel())
            for(size_t i=0; i<len; ++i) {
                chi_vec[i] = chi_vec[i] % PR; 
            }

            // 更新 final_item 的阶数
            final_item.degree = std::max(final_item.degree, tensor.degree);

            // =========================================================
            // 2. 【Verifier核心修改】累加 Keys (手动并行归约)
            // =========================================================
            // 注意：Verifier 处理的是 Key
            const uint64_t* key_ptr = tensor.get_keys_ptr();
            uint64_t sum_key = 0;

            if (len >= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel()) {
                #pragma omp parallel
                {
                    uint64_t local_sum = 0; // 线程私有变量
                    
                    #pragma omp for nowait
                    for (size_t i = 0; i < len; ++i) {
                        uint64_t term = mult_mod(key_ptr[i], chi_vec[i]);
                        local_sum = add_mod(local_sum, term);
                    }

                    #pragma omp critical
                    {
                        sum_key = add_mod(sum_key, local_sum);
                    }
                }
            } else {
                // 串行
                for (size_t i = 0; i < len; ++i) {
                    uint64_t term = mult_mod(key_ptr[i], chi_vec[i]);
                    sum_key = add_mod(sum_key, term);
                }
            }

            final_item.key = add_mod(final_item.key, sum_key);
            tensor.is_consumed = true;
        }
        check_tensor_buffer.clear();
    }
    
    /*
    void accumulate_tensor_buffer(PolyDelta& final_item) override {
        if (check_tensor_buffer.empty()) return;

        for (auto& tensor : check_tensor_buffer) {
            size_t len = tensor.total_elements;
            std::vector<uint64_t> chi_vec(len);
            this->prg.random_data(chi_vec.data(), len * sizeof(uint64_t));

            #pragma omp parallel for if (len >= MVZK_OMP_SIZE_THRESHOLD)
            for(size_t i=0; i<len; ++i) {
                chi_vec[i] = chi_vec[i] % PR; 
            }

            // 更新 final_item 的阶数 (取最大值)
            final_item.degree = std::max(final_item.degree, tensor.degree);

            // 累加 Keys (SoA)
            const uint64_t* key_ptr = tensor.get_keys_ptr();
            uint64_t sum_key = 0;

            for (size_t i = 0; i < len; ++i) {
                uint64_t term = mult_mod(key_ptr[i], chi_vec[i]);
                sum_key = add_mod(sum_key, term);
            }

            final_item.key = add_mod(final_item.key, sum_key);
            
            tensor.is_consumed = true;
        }
        check_tensor_buffer.clear();
    }*/

    // =========================================================
    // 5. LUT Section
    // =========================================================
    int register_lut_table(const std::vector<std::pair<uint64_t, uint64_t>>& data) override {
        auto entry = std::make_unique<LUTData>(data);
        size_t new_hash = compute_vector_hash(data);

        // Step 2: 遍历现有的表，寻找是否存在一样的
        for (size_t i = 0; i < lut_tables.size(); ++i) {
            LUTData* existing_table = lut_tables[i].get();

            // 快速检查 1: Hash 是否相等？ (如果不等，绝对不是同一张表)
            if (existing_table->data_hash != new_hash) {
                continue; 
            }

            // 快速检查 2: 大小是否相等？
            if (existing_table->data.size() != data.size()) {
                continue;
            }

            // 慢速检查 3: 只有 Hash 和 Size 都一样，才逐个字节比对内容
            // std::vector 重载了 == 运算符，会自动比较内容
            if (existing_table->data == data) {
                // 【命中！】直接返回旧的 ID
                // std::cout << "[INFO] LUT Table Deduplication: Reusing ID " << i << std::endl;
                return (int)i;
            }
        }
        lut_tables.push_back(std::move(entry));
        return (int)lut_tables.size() - 1;
    }

    // 在 MVZKExecVerifier 类中

    void submit_lut_check(const PolyTensor& keys, const PolyTensor& vals, int table_id) override {
        // 1. 安全检查
        if (table_id < 0 || table_id >= lut_tables.size()) {
            LOG_ERROR("Invalid Table ID: " << table_id);
            return;
        }

        LUTData* table = lut_tables[table_id].get();

        // 2. 准备请求对象
        LUTData::Request req;
        req.keys = keys.clone();
        req.vals = vals.clone();

        // =========================================================
        // 3. 【Verifier 逻辑】无 Hint
        // =========================================================
        // Verifier 不知道 tracker，也没有 tracker 数据。
        // req.v_j_hint 保持为空即可 (std::vector 默认构造为空)。
        // 这一步直接跳过。

        // 4. 入队
        table->buffer.push_back(std::move(req));
        table->current_buffer_size += keys.total_elements;
        keys.is_consumed = true;
        vals.is_consumed = true;

        // 5. 阈值检查
        // Verifier 必须和 Prover 保持同步的 Flush 节奏
        // 因为 Prover 触发 process_lut_batch 会发网络包，Verifier 必须同时也触发接收
        if (table->current_buffer_size >= MVZK_CONFIG_LUT_REQUEST_BUFFER_THRESHOLD) {
            process_lut_batch(table); 
        }
    }

    void process_lut_batch(LUTData* table) override {
        // 0. 空检查
        if (table->buffer.empty()) return;

        size_t N_Query = table->current_buffer_size;
        size_t N_Table = table->data.size();

        // =========================================================
        // Phase 1: Receive Commit (Verifier 独有逻辑)
        // =========================================================
        
        // Verifier 不需要提供数据，只需要指定形状来接收 VOLE 扩展
        // 这两个 Tensor 内部存储的是 Key/Delta
        std::vector<uint64_t> null_raw_data;
        PolyTensor V_J = this->input({(int)N_Query, 1}, null_raw_data);
        PolyTensor V_FINAL = this->input({(int)N_Table, 1}, null_raw_data);

        //debug_print(V_J, "V_j");
        //debug_print(V_FINAL, "V_Final");

        // =========================================================
        // Phase 2: Send Challenge (Verifier 生成随机数)
        // =========================================================
        block seed;
        this->prg.random_block(&seed, 1);
        io->send_data(&seed, sizeof(block)); // 发送给 Prover
        this->prg.reseed(&seed);             // 同步 PRG 状态

        uint64_t sx, sy, sv, r;
        this->prg.random_data(&sx, sizeof(uint64_t));
        this->prg.random_data(&sy, sizeof(uint64_t));
        this->prg.random_data(&sv, sizeof(uint64_t));
        this->prg.random_data(&r, sizeof(uint64_t));
        sx = sx % PR;
        sy = sy % PR;
        sv = sv % PR;
        r = r % PR;

        // =========================================================
        // Phase 3: RLC & Term Construction (代码同 Prover)
        // =========================================================

        // 3.1 拼接 Buffer
        // 这里的 Big_Keys/Vals 内部是密文形式 (或者 Verifier 持有的 Key/Delta)

        auto [Big_Keys, Big_Vals] = concat_buffer_tensors(table);

        //debug_print(Big_Keys, "Big_Keys");
        //debug_print(Big_Vals, "Big_Vals");

        // 3.2 计算 Query Read 项
        // Verifier 的 PolyTensor 运算符重载会自动处理 Key/Delta 逻辑
        PolyTensor Term_Q_Read = (Big_Keys * sx) + (Big_Vals * sy) + (V_J * sv) - r;

        //debug_instant_check(Term_Q_Read);

        // 3.3 计算 Query Write 项
        PolyTensor Term_Q_Write = Term_Q_Read + sv;

        //debug_instant_check(Term_Q_Write);

        // 3.4 计算 Final 项 和 Init 标量
        // 这里的逻辑必须和 Prover 一模一样，因为 table->data 是公开的
        std::vector<uint64_t> table_consts;
        table_consts.reserve(N_Table);
        uint64_t P_Init_Scalar = 1;

        for (const auto& kv : table->data) {
            uint64_t key = kv.first;
            uint64_t val = kv.second;
            
            uint64_t term_val = add_mod(mult_mod(key, sx), mult_mod(val, sy));
            table_consts.push_back(term_val);

            // Verifier 也在本地算出 Init 的总积
            P_Init_Scalar = mult_mod(P_Init_Scalar, add_mod(term_val, PR - r));
        }

        PolyTensor P_Table_Const = PolyTensor::from_public({(int)N_Table, 1}, table_consts);
        
        PolyTensor Term_Final = P_Table_Const + (V_FINAL * sv) - r;
        //debug_instant_check(Term_Final);

        // =========================================================
        // Phase 4: Grand Product (代码同 Prover)
        // =========================================================

        // 4.1 LHS
        std::vector<PolyTensor> lhs_list = split_to_scalars(Term_Q_Read);
        std::vector<PolyTensor> final_list = split_to_scalars(Term_Final);
        
        lhs_list.reserve(lhs_list.size() + final_list.size());
        lhs_list.insert(lhs_list.end(), 
                        std::make_move_iterator(final_list.begin()), 
                        std::make_move_iterator(final_list.end()));
        
        PolyTensor P_LHS = fast_tree_product(lhs_list);

        // 4.2 RHS
        std::vector<PolyTensor> rhs_list = split_to_scalars(Term_Q_Write);
        PolyTensor P_RHS_Write = fast_tree_product(rhs_list);

        // =========================================================
        // Phase 5: Check Zero
        // =========================================================
        
        PolyTensor Z = P_LHS - (P_RHS_Write * P_Init_Scalar);
        Z.is_constraint = true;
        
        //debug_print(Z, "");

        // =========================================================
        // Phase 6: Cleanup
        // =========================================================
        table->buffer.clear();
        table->current_buffer_size = 0;

        for (auto& pair : table->version_tracker) {
            pair.second = 0; 
        }

        // Verifier 将 Z 加入到全局验证队列中
        this->submit_tensor_to_buffer(std::move(Z));
    }

    // =========================================================
    // 1. 注册表 (Verifier 版：优化初始化)
    // =========================================================
    int register_range_check_table(const std::vector<uint64_t>& data) override {
        // RangeCheckData 构造函数里已经 resize 了 version_tracker (vector)
        if (data[0] != real2fp(0)){
            LOG_ERROR("Not implemented: Range Table must start from 0 in F_p, range check with offset is not implemented currently.");
            exit(-1);
        }
        auto entry = std::make_unique<RangeCheckData>(data);
        size_t new_hash = compute_vector_hash(data);

        // Step 2: 遍历现有的表，寻找是否存在一样的
        for (size_t i = 0; i < range_check_tables.size(); ++i) {
            RangeCheckData* existing_table = range_check_tables[i].get();

            // 快速检查 1: Hash 是否相等？ (如果不等，绝对不是同一张表)
            if (existing_table->data_hash != new_hash) {
                continue; 
            }

            // 快速检查 2: 大小是否相等？
            if (existing_table->data.size() != data.size()) {
                continue;
            }

            // 慢速检查 3: 只有 Hash 和 Size 都一样，才逐个字节比对内容
            // std::vector 重载了 == 运算符，会自动比较内容
            if (existing_table->data == data) {
                // 【命中！】直接返回旧的 ID
                // std::cout << "[INFO] LUT Table Deduplication: Reusing ID " << i << std::endl;
                return (int)i;
            }
        }
        range_check_tables.push_back(std::move(entry));
        return (int)range_check_tables.size() - 1;
    }

    // =========================================================
    // 2. 提交检查 (Verifier 版：优化切片 & 指针操作)
    // =========================================================
    void submit_range_check(const PolyTensor& x, int table_id) override {
        if (table_id < 0 || table_id >= range_check_tables.size()) {
            LOG_ERROR("Invalid Table ID: " << table_id);
            return;
        }

        RangeCheckData* table = range_check_tables[table_id].get();

        size_t len = x.total_elements;
        size_t offset = 0; // 切片游标
        size_t threshold = MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD;
        
        // 【优化】获取 Keys 的裸指针，加速拷贝
        const uint64_t* x_keys_ptr = x.get_keys_ptr();

        // 【核心修复】：循环切片
        while (offset < len) {
            size_t capacity = threshold - table->current_buffer_size;
            size_t current_chunk = std::min(capacity, len - offset);

            // 1. 构造切片子张量
            PolyTensor sub_x({(int)current_chunk}, x.degree);
            
            // Verifier 只需要拷贝 Keys (密文)
            // 使用 memcpy 替代循环，速度更快
            if (!x.flat_keys.empty()) {
                std::memcpy(sub_x.get_keys_ptr(), x_keys_ptr + offset, current_chunk * sizeof(uint64_t));
            }

            // 2. 构造 Request
            RangeCheckData::Request req;
            req.vals = std::move(sub_x); 
            
            // Verifier 不需要 v_j_hint，保持为空

            // 3. 入队
            table->buffer.push_back(std::move(req));
            table->current_buffer_size += current_chunk;

            // 4. 触达阈值，立刻 Flush！
            // 必须与 Prover 保持同步
            if (table->current_buffer_size >= threshold) {
                //WHITE("[INFO] Range check buffer full. Instant submit now.");
                process_range_check_batch(table); 
            }

            // 游标前进
            offset += current_chunk;
        }

        x.is_consumed = true;
    }

    // =========================================================
    // 3. 批处理 (Verifier 版：优化并行 & 清理)
    // =========================================================
    void process_range_check_batch(RangeCheckData* table) override {
        // 0. 空检查
        if (table->buffer.empty()) return;

        size_t N_Query = table->current_buffer_size;
        size_t N_Table = table->data.size();

        // =========================================================
        // Phase 1: Receive Commit (Verifier 独有逻辑)
        // =========================================================
        
        // Verifier 不需要提供数据，只需要指定形状来接收 VOLE 扩展
        // 这两个 Tensor 内部存储的是 Key/Delta
        std::vector<uint64_t> null_raw_data;
        // 注意：Input 函数内部可能会涉及网络接收，这里必须串行
        PolyTensor V_J = this->input({(int)N_Query, 1}, null_raw_data);
        PolyTensor V_FINAL = this->input({(int)N_Table, 1}, null_raw_data);

        //debug_print(V_J, "V_j");
        //debug_print(V_FINAL, "V_Final");

        // =========================================================
        // Phase 2: Send Challenge (Verifier 生成随机数)
        // =========================================================
        block seed;
        this->prg.random_block(&seed, 1);
        io->send_data(&seed, sizeof(block)); // 发送给 Prover
        this->prg.reseed(&seed);             // 同步 PRG 状态

        uint64_t sy, sv, r;
        //this->prg.random_data(&sx, sizeof(uint64_t));
        this->prg.random_data(&sy, sizeof(uint64_t));
        this->prg.random_data(&sv, sizeof(uint64_t));
        this->prg.random_data(&r, sizeof(uint64_t));
        //sx = sx % PR;
        sy = sy % PR;
        sv = sv % PR;
        r = r % PR;

        // =========================================================
        // Phase 3: RLC & Term Construction (代码同 Prover)
        // =========================================================

        // 3.1 拼接 Buffer
        // 这里的 Big_Keys/Vals 内部是密文形式 (或者 Verifier 持有的 Key/Delta)
        // concat_buffer_tensors 内部如果是 vector 拷贝，速度很快
        PolyTensor Big_Vals = concat_buffer_tensors(table);

        //debug_print(Big_Vals, "Big_Vals");

        // 3.2 计算 Query Read 项
        // Verifier 的 PolyTensor 运算符重载会自动处理 Key/Delta 逻辑
        PolyTensor Term_Q_Read = (Big_Vals * sy) + (V_J * sv) - r;

        //debug_instant_check(Term_Q_Read);

        // 3.3 计算 Query Write 项
        PolyTensor Term_Q_Write = Term_Q_Read + sv;

        //debug_instant_check(Term_Q_Write);

        // 3.4 计算 Final 项 和 Init 标量
        // 这里的逻辑必须和 Prover 一模一样，因为 table->data 是公开的
        // data.size() 通常较小 (256/65536)，串行计算即可，不必并行
        std::vector<uint64_t> table_consts;
        table_consts.reserve(N_Table);
        uint64_t P_Init_Scalar = 1;

        for (const auto& kv : table->data) {
            
            uint64_t term_val = mult_mod(kv, sy);
            table_consts.push_back(term_val);

            // Verifier 也在本地算出 Init 的总积
            P_Init_Scalar = mult_mod(P_Init_Scalar, add_mod(term_val, PR - r));
        }

        PolyTensor P_Table_Const = PolyTensor::from_public({(int)N_Table, 1}, table_consts);
        
        PolyTensor Term_Final = P_Table_Const + (V_FINAL * sv) - r;
        //debug_instant_check(Term_Final);

        // =========================================================
        // Phase 4: Grand Product (代码同 Prover)
        // =========================================================

        // 4.1 LHS
        // 【关键】调用并行化的 split_to_scalars
        std::vector<PolyTensor> lhs_list = this->split_to_scalars(Term_Q_Read);
        std::vector<PolyTensor> final_list = this->split_to_scalars(Term_Final);
        
        lhs_list.reserve(lhs_list.size() + final_list.size());
        lhs_list.insert(lhs_list.end(), 
                        std::make_move_iterator(final_list.begin()), 
                        std::make_move_iterator(final_list.end()));
        
        // 并行归约
        PolyTensor P_LHS = fast_tree_product(lhs_list);

        // 4.2 RHS
        std::vector<PolyTensor> rhs_list = this->split_to_scalars(Term_Q_Write);
        PolyTensor P_RHS_Write = fast_tree_product(rhs_list);

        // =========================================================
        // Phase 5: Check Zero
        // =========================================================
        
        PolyTensor Z = P_LHS - (P_RHS_Write * P_Init_Scalar);
        Z.is_constraint = true;
        
        //debug_print(Z, "");

        // =========================================================
        // Phase 6: Cleanup (优化)
        // =========================================================
        table->buffer.clear();
        table->current_buffer_size = 0;

        // 【优化】快速清零 vector (虽然 Verifier 不怎么用 tracker，但保持状态一致是个好习惯)
        // 原代码: for (auto& pair : table->version_tracker) pair.second = 0; 
        std::fill(table->version_tracker.begin(), table->version_tracker.end(), 0);

        // Verifier 将 Z 加入到全局验证队列中
        this->submit_tensor_to_buffer(std::move(Z));
    }

private:
    void delta_gen() {
        PRG prg;
        prg.random_data(&delta, sizeof(__uint128_t));
        extract_fp(delta);
    }
};

#endif