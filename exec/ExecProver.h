#ifndef MVZK_EXECUTION_PROVER_H__
#define MVZK_EXECUTION_PROVER_H__

#include "exec/MVZKExec.h"
#include "emp-zk/emp-vole/emp-vole.h"
#include "emp-zk/emp-vole/vole_triple.h"
#include "../config.h"
#include "../data_type/PolyDelta.h"
#include "../data_type/PolyTensor.h"
#include "../utility.h"
#include "emp-tool/utils/hash.h"
#include "../operations/nonlinear.h"
#include "../operations/lut.h"
#include <omp.h>
#include <cstring> 

#define LOW64(x) _mm_extract_epi64((block)x, 0)
#define HIGH64(x) _mm_extract_epi64((block)x, 1)

template <typename IO> 
class MVZKExecProver : public MVZKExec {
public:
    IO *io = nullptr;
    VoleTriple<IO> *vole = nullptr;
    int party;
    int threads;
    PRG prg;
    
    // 【新增】Buffer：用于存储待验证的约束多项式
    std::vector<PolyDelta> check_buffer;
    std::vector<PolyTensor> check_tensor_buffer;
    //std::vector<PolyTensor> check_matmul_buffer;
    // 【新增】Buffer 阈值：防止内存溢出，可根据需要调整

    MVZKExecProver(IO **ios) : MVZKExec() {
        this->party = ALICE; // Prover 通常作为 ALICE
        this->io = ios[0];
        this->threads = MVZK_CONFIG_THREADS_NUM; // 确保使用定义的线程数
        
        // 【关键】将自己注册为当前的全局执行环境
        MVZKExec::mvzk_exec = this;

        this->vole = new VoleTriple<IO>(BOB, threads, ios); // 对方是 BOB
        vole->setup();
        
        // VoleTriple 预生成 VOLE triples
        __uint128_t tmp;
        vole->extend(&tmp, 1);
    }

    virtual ~MVZKExecProver() {
        // 析构前强制检查剩余的 buffer
        // flush_all_luts();
        // check_all();
        if(vole) delete vole;
    }

    // =========================================================
    // 1. Buffer 管理接口实现
    // =========================================================

    void submit_to_buffer(PolyDelta&& pd) override {
        if (!pd.is_constraint){
            LOG_WARN("A non-constraint PolyDelta received by buffer!");
        }
        pd.is_consumed = true;
        check_buffer.emplace_back(std::move(pd));

        if (check_buffer.size() >= MVZK_MULT_CHECK_CNT) {
            //WHITE("[INFO] Check buffer threshold reached, instant check now.");
            check_all();
        }
    }

    void submit_tensor_to_buffer(PolyTensor&& pt) override {
        if (!pt.is_constraint) {
             LOG_WARN("A non-constraint PolyTensor received by buffer!");
        }
        //BLUE("[INFO] PolyTensor Submitted to buffer with size " << pt.total_elements << ", deg = " << pt.degree);
        pt.is_consumed = true;
        check_tensor_buffer.emplace_back(std::move(pt));
        
        // Tensor 比较大，阈值设小一点，防止内存爆炸
        if (check_tensor_buffer.size() >= MVZK_CONFIG_TENSOR_CHECK_CNT) { 
            //WHITE("[INFO] Check tensor buffer threshold reached, instant check now.");
            check_all();
        }
    }

    void submit_non_zero_tensor_to_buffer(const PolyTensor& target) override {
        /*
        if (!target.is_constraint) {
            LOG_WARN("A non-constraint non-zero PolyTensor received by buffer!");
        }
            // We disable this warning here.
            */
        
        target.is_consumed = true; 

        // 1. 构造更高 1 阶的新张量
        int new_degree = target.degree + 1;
        PolyTensor padded_tensor(target.shape, new_degree);
        padded_tensor.is_constraint = true;
        
        // 2. 真实值 (Real Vals) 全部设为 0
        // 使用 assign 确保无论是空 vector 还是已被构造函数初始化的 vector，都能被正确覆盖为 0

        // 3. 完美拷贝所有低阶系数 (0 到 target.degree)
        for (int d = 0; d <= target.degree; ++d) {
            const uint64_t* in_coeffs = target.get_coeffs_ptr(d);
            uint64_t* out_coeffs = padded_tensor.get_coeffs_ptr(d);
            
            // 使用标准库的高效内存拷贝，比写 for 循环逐个赋值更快
            std::copy(in_coeffs, in_coeffs + target.total_elements, out_coeffs);
        }

        // 4. 将最高阶 (new_degree) 的系数全部强制设为 0
        uint64_t* highest_coeffs = padded_tensor.get_coeffs_ptr(new_degree);
        std::fill(highest_coeffs, highest_coeffs + target.total_elements, 0);

        // 5. 将伪造好的张量，利用 std::move 压入底层的零校验池
        // 注意：请将 check_tensor_buffer 替换为你们实际存放零校验多项式的 Buffer 变量名
        this->check_tensor_buffer.push_back(std::move(padded_tensor));
        if (check_tensor_buffer.size() >= MVZK_CONFIG_TENSOR_CHECK_CNT) { 
            //WHITE("[INFO] Check tensor buffer threshold reached, instant check now.");
            check_all();
        }
    }

    PolyTensor refresh_tensor_degree(const PolyTensor& high_degree_tensor, const std::string& check_name) override {
        // 1. 获取 Prover 声称的明文数据 (警告：此时数据不可信)
        std::vector<uint64_t> plaintext_val = high_degree_tensor.get_real_vals_vector();
        std::vector<int> shape_vec = high_degree_tensor.shape;

        // 2. 注入系统，获取崭新的 Degree 1 张量 res
        PolyTensor res = input(shape_vec, plaintext_val);

        PolyTensor::store_relation(high_degree_tensor, res, check_name);

        return res;
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

        // 1. 空检查
        if (check_buffer.empty() && check_tensor_buffer.empty()){
            LOG_WARN("check_all() called with no items in all buffers.");
            return;
        } 

        // 2. 接收种子，同步随机数生成器
        block seed;
        io->recv_data(&seed, sizeof(block));
        this->prg.reseed(&seed);

        // 3. 全局累加器
        PolyDelta final_checked_item;

        // 4. 分别累加两类 Buffer
        accumulate_delta_buffer(final_checked_item);
        accumulate_tensor_buffer(final_checked_item);

        if (final_checked_item.degree > MVZK_FFT_WARNING_LIMIT){
            LOG_WARN("FFT Optimization is recommended, deg = " << final_checked_item.degree << ", alert limit = " << MVZK_FFT_WARNING_LIMIT << ".");
        }

        // 5. 生成 Masked VOPE 并发送
        // 此时 final_checked_item 包含了所有约束的线性组合
        if (final_checked_item.degree > 1){
            PolyDelta vope = std::move(extend_vope_from_vole(final_checked_item.degree - 1, 1)[0]);

            //debug_print(vope, "Prover VOPE");

            // Here we have to mask it, instead of directly add it.
            for (int i = 0; i < final_checked_item.coeffs.size() - 1; i++){
                final_checked_item.coeffs[i] = add_mod(final_checked_item.coeffs[i], vope.coeffs[i]);
            }
            vope.is_consumed = true;
            //final_checked_item = final_checked_item + vope;
        
            send_poly(final_checked_item);
        }else{
            // We consider another different case. If final checked item degree = 0, then it is check zero by a hash oracle.
            emp::Hash hash_instance;
            hash_instance.put(&final_checked_item.coeffs[0], sizeof(uint64_t));
            char dig[emp::Hash::DIGEST_SIZE];
            hash_instance.digest(dig);
            io->send_data(dig, emp::Hash::DIGEST_SIZE);
        }
        final_checked_item.is_consumed = true;
    }

    void finalize_protocol() {
        flush_all_luts();
        flush_all_range_checks();
        check_all(); 
    }

    // =========================================================
    // 2. 多项式运算接口实现 (PolyDelta vs PolyDelta)
    // =========================================================
    // 注意：Prover 需要操作 PolyDelta 中的 coeffs 向量进行多项式运算

    PolyDelta add(const PolyDelta& lhs, const PolyDelta& rhs) override {
        // Prover side, need align the polydelta

        PolyDelta res;
        
        // 1. 确定元数据
        res.degree = std::max(lhs.degree, rhs.degree);
        
        // 2. 预分配结果内存 
        size_t size = res.degree + 1;
        res.coeffs.resize(size);

        // 3. 计算偏移量 (Offset)
        size_t offset_lhs = res.degree - lhs.degree;
        size_t offset_rhs = res.degree - rhs.degree;

        // 4. 单次循环直接计算 (Hot Loop)
        // 编译器极易对此进行自动向量化 (SIMD) 优化
        for (size_t i = 0; i < size; ++i) {
            // 【核心逻辑】逻辑对齐，取代 insert
            // 如果 i < offset，说明这一位是补的 0
            // 否则，取 coeffs[i - offset]
            uint64_t u = (i < offset_lhs) ? 0 : lhs.coeffs[i - offset_lhs];
            uint64_t v = (i < offset_rhs) ? 0 : rhs.coeffs[i - offset_rhs];

            res.coeffs[i] = add_mod(u, v);
        }

        // 5. 状态管理
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;

        return res; // RVO (返回值优化) 会生效，没有拷贝开销
    }

    PolyTensor add(const PolyTensor& lhs, const PolyTensor& rhs) override {
        // 1. 维度检查
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("shape mismatch in add!");
            exit(-1);
        }
        
        // 2. 构造结果对象 (使用 max degree)
        int res_degree = std::max(lhs.degree, rhs.degree);
        PolyTensor res(lhs.shape, res_degree); // 这里分配了全新的内存，全 0
        
        size_t size = lhs.total_elements;

        // 3. 处理 Real Values (res = lhs + rhs)
        // 获取指针 (Direct Access)

        // 4. 处理系数 (逻辑对齐)
        // 这里的逻辑和 PolyDelta 完全一致，只是变成了并行版
        
        // 预先获取所有需要的指针
        const std::vector<uint64_t>& lhs_coeffs = lhs.flat_coeffs;
        const std::vector<uint64_t>& rhs_coeffs = rhs.flat_coeffs;
        std::vector<uint64_t>& res_coeffs = res.flat_coeffs;

        // 计算偏移量
        int offset_lhs = res_degree - lhs.degree;
        int offset_rhs = res_degree - rhs.degree;

        // 计算总的系数块大小 (SIMD Friendly Loop)
        // 我们遍历结果的所有系数： res_coeffs[k * size + i]
        // 但为了性能，我们直接把整个 flat_coeffs 当作一维大数组遍历
        
        size_t total_coeffs_len = size * (res_degree + 1);

        // 为了并行方便，我们需要知道当前 index 属于哪一阶
        // 但那样需要除法。更高效的方法是：分阶处理。
        
        // 4.1 遍历每一阶 (d from 0 to res_degree)
        for (int d = 0; d <= res_degree; ++d) {
            // 计算当前阶对应的数据源 (LHS/RHS) 的阶数
            // 如果 d < offset，说明对于该操作数来说，这一阶还没到（是补零区）
            // 对应关系：res 的 d 阶对应 lhs 的 (d - offset) 阶
            
            int d_lhs = d - offset_lhs;
            int d_rhs = d - offset_rhs;

            // 获取当前阶的写入指针
            uint64_t* res_ptr = res.get_coeffs_ptr(d);
            
            // 获取源指针 (如果存在)
            const uint64_t* lhs_ptr = (d_lhs >= 0) ? lhs.get_coeffs_ptr(d_lhs) : nullptr;
            const uint64_t* rhs_ptr = (d_rhs >= 0) ? rhs.get_coeffs_ptr(d_rhs) : nullptr;

            // 并行加法
            #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                uint64_t u = (lhs_ptr) ? lhs_ptr[i] : 0;
                uint64_t v = (rhs_ptr) ? rhs_ptr[i] : 0;
                res_ptr[i] = add_mod(u, v);
            }
        }

        // 5. 状态管理
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        res.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;

        return res; // RVO 完美生效
    }

    PolyDelta sub(const PolyDelta& lhs, const PolyDelta& rhs) override {
        // 【逻辑留空】：实现多项式减法, lhs - rhs
        // Prover side, need align the polydelta

        PolyDelta res;
        
        // 1. 确定元数据
        res.degree = std::max(lhs.degree, rhs.degree);
        
        // 2. 预分配结果内存 
        size_t size = res.degree + 1;
        res.coeffs.resize(size);

        // 3. 计算偏移量 (Offset)
        size_t offset_lhs = res.degree - lhs.degree;
        size_t offset_rhs = res.degree - rhs.degree;

        // 4. 单次循环直接计算 (Hot Loop)
        // 编译器极易对此进行自动向量化 (SIMD) 优化
        for (size_t i = 0; i < size; ++i) {
            // 【核心逻辑】逻辑对齐，取代 insert
            // 如果 i < offset，说明这一位是补的 0
            // 否则，取 coeffs[i - offset]
            uint64_t u = (i < offset_lhs) ? 0 : lhs.coeffs[i - offset_lhs];
            uint64_t v = (i < offset_rhs) ? 0 : rhs.coeffs[i - offset_rhs];

            res.coeffs[i] = add_mod(u, (PR - v));
        }

        // 5. 状态管理
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;

        return res; // RVO (返回值优化) 会生效，没有拷贝开销
    }

    PolyTensor sub(const PolyTensor& lhs, const PolyTensor& rhs) override {
        // 1. 维度检查
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("shape mismatch in sub!");
            exit(-1);
        }
        
        // 2. 构造结果对象 (使用 max degree)
        int res_degree = std::max(lhs.degree, rhs.degree);
        PolyTensor res(lhs.shape, res_degree); 
        
        size_t size = lhs.total_elements;

        // 3. 处理 Real Values (res = lhs - rhs)

        // 4. 处理系数 (逻辑对齐)
        // 计算偏移量
        int offset_lhs = res_degree - lhs.degree;
        int offset_rhs = res_degree - rhs.degree;
        
        // 遍历每一阶 (d from 0 to res_degree)
        for (int d = 0; d <= res_degree; ++d) {
            // 计算数据源对应的阶数
            int d_lhs = d - offset_lhs;
            int d_rhs = d - offset_rhs;

            // 获取指针
            uint64_t* res_ptr = res.get_coeffs_ptr(d);
            
            // 如果 d_lhs < 0，说明对于 lhs 来说这是低位补零区，指针设为 nullptr
            const uint64_t* lhs_ptr = (d_lhs >= 0) ? lhs.get_coeffs_ptr(d_lhs) : nullptr;
            const uint64_t* rhs_ptr = (d_rhs >= 0) ? rhs.get_coeffs_ptr(d_rhs) : nullptr;

            // 并行减法
            #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                uint64_t u = (lhs_ptr) ? lhs_ptr[i] : 0;
                uint64_t v = (rhs_ptr) ? rhs_ptr[i] : 0;
                
                // 【差异点】u - v
                // 注意：如果 v > u，sub_mod 会自动加 PR
                res_ptr[i] = add_mod(u, PR - v);
            }
        }

        // 5. 状态管理
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        res.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;

        return res; 
    }

    PolyDelta mul(const PolyDelta& lhs, const PolyDelta& rhs) override {
        // Prover side multiplication
        PolyDelta res;

        // 1. 确定元数据
        res.degree = lhs.degree + rhs.degree;

        // 2. 初始化系数向量
        // 必须初始化为 0，因为后面要进行累加 (Convolution)
        size_t size = res.degree + 1;
        res.coeffs.assign(size, 0); 

        // 3. 执行卷积 (Convolution)
        // 遍历 lhs 的每一项和 rhs 的每一项，乘积累加到对应的指数位置 (i+j)
        for (size_t i = 0; i < lhs.coeffs.size(); ++i) {
            // 优化：如果 lhs 系数是 0，跳过内层循环 (稀疏优化)
            if (lhs.coeffs[i] == 0) continue;

            for (size_t j = 0; j < rhs.coeffs.size(); ++j) {
                // 计算项：coeffs[i] * coeffs[j]
                uint64_t term = mult_mod(lhs.coeffs[i], rhs.coeffs[j]);
                
                // 累加到结果的 (i+j) 位置
                // res[i+j] += term
                res.coeffs[i + j] = add_mod(res.coeffs[i + j], term);
            }
        }

        // 4. 状态管理
        // 结果是新的活跃对象
        res.is_consumed = false; 
        // 输入操作数被消耗
        lhs.is_consumed = true;
        rhs.is_consumed = true;

        return res;
    }

    /*
    PolyTensor mul(const PolyTensor& lhs, const PolyTensor& rhs) override {
        // 1. 维度检查
        if (lhs.total_elements != rhs.total_elements) {
            std::cerr << "[ERROR] Tensor shape mismatch in mul!" << std::endl;
            exit(-1);
        }

        // 2. 确定结果阶数
        int res_degree = lhs.degree + rhs.degree;

        // 3. 构造结果对象 (全0初始化)
        // 注意：Convolution 是累加过程，必须初始化为 0
        PolyTensor res(lhs.shape, res_degree);
        size_t size = lhs.total_elements;

        // 4. 处理 Real Values (直接相乘)
        const uint64_t* lhs_real = lhs.get_real_vals_ptr();
        const uint64_t* rhs_real = rhs.get_real_vals_ptr();
        uint64_t* res_real = res.get_real_vals_ptr();

        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            res_real[i] = mult_mod(lhs_real[i], rhs_real[i]);
        }

        // 5. 处理系数 (多项式卷积 Convolution)
        // 目标：res[k] += lhs[i] * rhs[j]  where i+j = k
        
        // 遍历 LHS 的每一阶
        for (int i = 0; i <= lhs.degree; ++i) {
            const uint64_t* lhs_ptr = lhs.get_coeffs_ptr(i);

            // 遍历 RHS 的每一阶
            for (int j = 0; j <= rhs.degree; ++j) {
                const uint64_t* rhs_ptr = rhs.get_coeffs_ptr(j);
                
                // 目标阶数
                int k = i + j;
                uint64_t* res_ptr = res.get_coeffs_ptr(k);

                // 并行累加 (Hot Loop)
                #pragma omp parallel for
                for (size_t idx = 0; idx < size; ++idx) {
                    uint64_t term = mult_mod(lhs_ptr[idx], rhs_ptr[idx]);
                    res_ptr[idx] = add_mod(res_ptr[idx], term);
                }
            }
        }

        // 6. 状态管理
        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        res.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;

        return res;
    }
        

    PolyTensor mul(const PolyTensor& lhs, const PolyTensor& rhs) override {
        if (lhs.total_elements != rhs.total_elements) {
            std::cerr << "[ERROR] Tensor shape mismatch in mul!" << std::endl;
            exit(-1);
        }

        int res_degree = lhs.degree + rhs.degree;
        PolyTensor res(lhs.shape, res_degree); 
        size_t size = lhs.total_elements;

        const uint64_t* lhs_real = lhs.get_real_vals_ptr();
        const uint64_t* rhs_real = rhs.get_real_vals_ptr();
        uint64_t* res_real = res.get_real_vals_ptr();

        // Real Values: 受阈值保护的并行
        #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD)
        for (size_t i = 0; i < size; ++i) {
            res_real[i] = mult_mod(lhs_real[i], rhs_real[i]);
        }

        // Coefficients: 动态双规避并行策略
        if (size >= MVZK_OMP_SIZE_THRESHOLD) {
            // 【策略 A】: 大张量，内层空间并行
            for (int k = 0; k <= res_degree; ++k) {
                uint64_t* res_ptr = res.get_coeffs_ptr(k);
                int start_i = std::max(0, k - rhs.degree);
                int end_i = std::min(lhs.degree, k);

                #pragma omp parallel for
                for (size_t idx = 0; idx < size; ++idx) {
                    uint64_t sum = 0;
                    for (int i = start_i; i <= end_i; ++i) {
                        int j = k - i;
                        uint64_t term = mult_mod(lhs.get_coeffs_ptr(i)[idx], rhs.get_coeffs_ptr(j)[idx]);
                        sum = add_mod(sum, term);
                    }
                    res_ptr[idx] = sum;
                }
            }
        } else {
            // 【策略 B】: 极小张量，外层阶数并行
            #pragma omp parallel for if(res_degree >= MVZK_OMP_DEGREE_THRESHOLD)
            for (int k = 0; k <= res_degree; ++k) {
                uint64_t* res_ptr = res.get_coeffs_ptr(k);
                int start_i = std::max(0, k - rhs.degree);
                int end_i = std::min(lhs.degree, k);

                for (size_t idx = 0; idx < size; ++idx) {
                    uint64_t sum = 0;
                    for (int i = start_i; i <= end_i; ++i) {
                        int j = k - i;
                        uint64_t term = mult_mod(lhs.get_coeffs_ptr(i)[idx], rhs.get_coeffs_ptr(j)[idx]);
                        sum = add_mod(sum, term);
                    }
                    res_ptr[idx] = sum;
                }
            }
        }

        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        res.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;
        return res;
    } */

    PolyTensor mul(const PolyTensor& lhs, const PolyTensor& rhs) override {
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("Tensor shape mismatch in mul!");
            exit(-1);
        }

        int res_degree = lhs.degree + rhs.degree;
        PolyTensor res(lhs.shape, res_degree); 
        size_t size = lhs.total_elements;

        // 1. Real Values (明文) 并行乘法

        // 2. 调用 utility.h 中的全局 Karatsuba 引擎计算系数
        karatsuba_core(
            lhs.get_coeffs_ptr(0), lhs.degree,
            rhs.get_coeffs_ptr(0), rhs.degree,
            res.get_coeffs_ptr(0), size
        );

        res.is_consumed = false;
        lhs.is_consumed = true;
        rhs.is_consumed = true;
        res.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;
        return res;
    }
    
    // =========================================================
    // 5. In-Place 运算实现 (针对 Prover 的特殊对齐逻辑)
    // =========================================================

    void add_assign(PolyDelta& lhs, const PolyDelta& rhs) override {
        // 1. 处理 Real Value (直接累加)

        // 2. 比较阶数，决定策略
        if (rhs.degree > lhs.degree) {
            // 【情况 A: LHS 需要扩容并移位】
            // 这是一个相对昂贵的操作，但在 check_all 累加过程中，
            // 只要 lhs 增长到最大阶数，后续就不再触发此分支。
            
            size_t old_size = lhs.coeffs.size();
            size_t new_size = rhs.coeffs.size();
            size_t offset_diff = rhs.degree - lhs.degree; // LHS 需要向右移动的步长

            // 扩容 (此时新增加的空间在末尾，且未初始化/为0)
            lhs.coeffs.resize(new_size);

            // 关键步骤：数据搬移 (Right Shift)
            // 将原有的 [0, old_size) 数据移动到 [offset, new_size)
            // 必须从后往前搬，防止覆盖
            for (int i = (int)old_size - 1; i >= 0; --i) {
                lhs.coeffs[i + offset_diff] = lhs.coeffs[i];
            }

            // 填充高位零 (Padding)
            // 新腾出来的 [0, offset_diff) 是高阶项的补零位
            for (size_t i = 0; i < offset_diff; ++i) {
                lhs.coeffs[i] = 0;
            }

            // 更新 LHS 的阶数
            lhs.degree = rhs.degree;
        }

        // 【情况 B: LHS 足够大 (或刚扩容完)】
        // 此时 lhs.degree >= rhs.degree
        // 我们只需要把 rhs 加到 lhs 的 "低位" 部分
        
        size_t offset_rhs = lhs.degree - rhs.degree;
        size_t rhs_len = rhs.coeffs.size();

        // 核心计算循环
        for (size_t i = 0; i < rhs_len; ++i) {
            // lhs 的对齐位置 = i + offset
            size_t target_idx = i + offset_rhs;
            lhs.coeffs[target_idx] = add_mod(lhs.coeffs[target_idx], rhs.coeffs[i]);
        }

        // Update consumption flag
        rhs.is_consumed = true;
        lhs.is_consumed = false;
    }

    void add_assign(PolyTensor& lhs, const PolyTensor& rhs) override {
        // 1. 维度检查
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("Tensor shape mismatch in prover add_assign!");
            exit(-1);
        }

        size_t size = lhs.total_elements;

        // =================================================
        // Part 1: 处理 Real Values (永远是 In-place 加法)
        // =================================================

        // =================================================
        // Part 2: 处理 Coefficients (需要处理内存布局)
        // =================================================
        
        if (rhs.degree > lhs.degree) {
            // 【情况 A: LHS 需要升阶扩容】
            // 假设 LHS 原来是 [Deg0] (size=N)
            // RHS 是 [Deg0][Deg1] (size=2N)
            // 结果应该是 LHS(作为高阶) + RHS(作为低阶)
            // 内存需要变成: [RHS_0] [RHS_1 + LHS_0] 
            // 也就是 LHS 的数据要移动到高位去。

            int old_deg = lhs.degree;
            int new_deg = rhs.degree;
            int offset = new_deg - old_deg;

            // 2.1 扩容 vector
            // 注意：resize 可能会导致重新分配内存，旧指针会失效！
            lhs.flat_coeffs.resize(size * (new_deg + 1));
            
            // 2.2 重新获取指针
            uint64_t* lhs_base = lhs.flat_coeffs.data();
            const uint64_t* rhs_base = rhs.flat_coeffs.data();

            // 2.3 数据搬移 (Shift to High Degree)
            // 我们要把旧的 LHS 数据搬到后面去。
            // 旧数据占据 [0, size * (old_deg + 1))
            // 新位置应该在 [offset * size, ...)
            
            // 计算搬移字节数
            size_t bytes_to_move = size * (old_deg + 1) * sizeof(uint64_t);
            
            // 使用 memmove (处理内存重叠更安全)
            // Dest: lhs_base + offset * size
            // Src:  lhs_base
            std::memmove(lhs_base + (offset * size), lhs_base, bytes_to_move);

            // 2.4 低位清零 (Padding)
            // 腾出来的低位空间 [0, offset * size) 先清零
            std::memset(lhs_base, 0, offset * size * sizeof(uint64_t));

            // 2.5 更新 LHS 元数据
            lhs.degree = new_deg;

            // 2.6 执行加法 (把 RHS 加进来)
            // 此时 LHS 已经是 [0, 0, ..., Old_LHS] 的形状了
            // 直接把 RHS 加到对应位置即可
            
            size_t total_len = size * (new_deg + 1);
            #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < total_len; ++i) {
                lhs_base[i] = add_mod(lhs_base[i], rhs_base[i]);
            }

        } else {
            // 【情况 B: LHS 够大，直接加】 (常见情况)
            // LHS: [Deg0] [Deg1] ...
            // RHS: [Deg0] ...
            // 我们只需要把 RHS 加到 LHS 的“低位”部分去。
            // 这里的“低位”是指逻辑上的低阶项，在 SoA 布局里，
            // P_new = P_lhs + P_rhs * X^diff
            // 意味着 RHS 对应的是 LHS 中 index 较大的部分 (高阶内存块)。
            
            int offset = lhs.degree - rhs.degree;
            
            // RHS 的数据应该加到 LHS 的 [offset * size] 开始的地方
            size_t start_idx = offset * size;
            size_t rhs_len = size * (rhs.degree + 1);

            uint64_t* lhs_base = lhs.flat_coeffs.data();
            const uint64_t* rhs_base = rhs.flat_coeffs.data();

            #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < rhs_len; ++i) {
                lhs_base[start_idx + i] = add_mod(lhs_base[start_idx + i], rhs_base[i]);
            }
        }

        // 状态更新
        lhs.is_consumed = false;
        rhs.is_consumed = true;
        lhs.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;
    }

    void sub_assign(PolyDelta& lhs, const PolyDelta& rhs) override {
        // 1. 处理 Real Value (lhs = lhs - rhs)
        // 使用 add_mod(a, PR - b) 来实现减法

        // 2. 比较阶数，决定扩容策略 (逻辑同 add_assign)
        if (rhs.degree > lhs.degree) {
            // 【情况 A: LHS 需要扩容并移位】
            size_t old_size = lhs.coeffs.size();
            size_t new_size = rhs.coeffs.size();
            size_t offset_diff = rhs.degree - lhs.degree;

            // 扩容
            lhs.coeffs.resize(new_size);

            // 数据搬移 (Right Shift)，从后往前搬
            for (int i = (int)old_size - 1; i >= 0; --i) {
                lhs.coeffs[i + offset_diff] = lhs.coeffs[i];
            }

            // 填充高位零
            for (size_t i = 0; i < offset_diff; ++i) {
                lhs.coeffs[i] = 0;
            }

            // 更新 LHS 的阶数
            lhs.degree = rhs.degree;
        }

        // 3. 执行减法
        // 此时 lhs.degree >= rhs.degree，只需要处理对齐部分
        
        size_t offset_rhs = lhs.degree - rhs.degree;
        size_t rhs_len = rhs.coeffs.size();

        for (size_t i = 0; i < rhs_len; ++i) {
            size_t target_idx = i + offset_rhs;
            
            // 核心减法逻辑: lhs[target] = lhs[target] - rhs[i]
            // 注意处理 rhs[i] == 0 的情况，防止 PR - 0 导致可能的边界问题
            // (虽然 add_mod 通常能处理 PR，但严谨一点更好)
            uint64_t subtrahend = (rhs.coeffs[i] == 0) ? 0 : (PR - rhs.coeffs[i]);
            
            lhs.coeffs[target_idx] = add_mod(lhs.coeffs[target_idx], subtrahend);
        }
        rhs.is_consumed = true;
        lhs.is_consumed = false;
    }

    void sub_assign(PolyTensor& lhs, const PolyTensor& rhs) override {
        // 1. 维度检查
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("Tensor shape mismatch in prover add_assign!");
            exit(-1);
        }

        size_t size = lhs.total_elements;

        // =================================================
        // Part 1: 处理 Real Values (永远是 In-place 加法)
        // =================================================

        // =================================================
        // Part 2: 处理 Coefficients (需要处理内存布局)
        // =================================================
        
        if (rhs.degree > lhs.degree) {
            // 【情况 A: LHS 需要升阶扩容】
            // 假设 LHS 原来是 [Deg0] (size=N)
            // RHS 是 [Deg0][Deg1] (size=2N)
            // 结果应该是 LHS(作为高阶) + RHS(作为低阶)
            // 内存需要变成: [RHS_0] [RHS_1 + LHS_0] 
            // 也就是 LHS 的数据要移动到高位去。

            int old_deg = lhs.degree;
            int new_deg = rhs.degree;
            int offset = new_deg - old_deg;

            // 2.1 扩容 vector
            // 注意：resize 可能会导致重新分配内存，旧指针会失效！
            lhs.flat_coeffs.resize(size * (new_deg + 1));
            
            // 2.2 重新获取指针
            uint64_t* lhs_base = lhs.flat_coeffs.data();
            const uint64_t* rhs_base = rhs.flat_coeffs.data();

            // 2.3 数据搬移 (Shift to High Degree)
            // 我们要把旧的 LHS 数据搬到后面去。
            // 旧数据占据 [0, size * (old_deg + 1))
            // 新位置应该在 [offset * size, ...)
            
            // 计算搬移字节数
            size_t bytes_to_move = size * (old_deg + 1) * sizeof(uint64_t);
            
            // 使用 memmove (处理内存重叠更安全)
            // Dest: lhs_base + offset * size
            // Src:  lhs_base
            std::memmove(lhs_base + (offset * size), lhs_base, bytes_to_move);

            // 2.4 低位清零 (Padding)
            // 腾出来的低位空间 [0, offset * size) 先清零
            std::memset(lhs_base, 0, offset * size * sizeof(uint64_t));

            // 2.5 更新 LHS 元数据
            lhs.degree = new_deg;

            // 2.6 执行加法 (把 RHS 加进来)
            // 此时 LHS 已经是 [0, 0, ..., Old_LHS] 的形状了
            // 直接把 RHS 加到对应位置即可
            
            size_t total_len = size * (new_deg + 1);
            #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < total_len; ++i) {
                lhs_base[i] = add_mod(lhs_base[i], PR - rhs_base[i]);
            }

        } else {
            // 【情况 B: LHS 够大，直接加】 (常见情况)
            // LHS: [Deg0] [Deg1] ...
            // RHS: [Deg0] ...
            // 我们只需要把 RHS 加到 LHS 的“低位”部分去。
            // 这里的“低位”是指逻辑上的低阶项，在 SoA 布局里，
            // P_new = P_lhs + P_rhs * X^diff
            // 意味着 RHS 对应的是 LHS 中 index 较大的部分 (高阶内存块)。
            
            int offset = lhs.degree - rhs.degree;
            
            // RHS 的数据应该加到 LHS 的 [offset * size] 开始的地方
            size_t start_idx = offset * size;
            size_t rhs_len = size * (rhs.degree + 1);

            uint64_t* lhs_base = lhs.flat_coeffs.data();
            const uint64_t* rhs_base = rhs.flat_coeffs.data();

            #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < rhs_len; ++i) {
                lhs_base[start_idx + i] = add_mod(lhs_base[start_idx + i], PR - rhs_base[i]);
            }
        }

        // 状态更新
        lhs.is_consumed = false;
        rhs.is_consumed = true;
        lhs.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;
    }

    void mul_assign(PolyDelta& lhs, uint64_t val) override {
        // Prover 侧乘法：所有系数和 real_val 都要乘
        // 这个没有对齐问题，直接遍历即可

        // 使用引用遍历，避免下标访问开销，极快
        for (auto& c : lhs.coeffs) {
            c = mult_mod(c, val);
        }
        lhs.is_consumed = false;
    }

    void mul_assign(PolyTensor& lhs, uint64_t val) override {
        size_t size = lhs.total_elements;

        // 1. 更新 Real Values (全部乘 val)

        // 2. 更新所有系数 (Coefficients)
        // 无论是 0阶(MAC) 还是 1阶(x) 还是 高阶，全都要乘
        // 直接遍历整个 flat_coeffs 数组效率最高
        size_t total_coeffs_len = lhs.flat_coeffs.size();
        uint64_t* coeffs_ptr = lhs.flat_coeffs.data();

        #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < total_coeffs_len; ++i) {
            coeffs_ptr[i] = mult_mod(coeffs_ptr[i], val);
        }

        // 状态：结果是活跃的
        lhs.is_consumed = false;
    }

    void mul_assign(PolyTensor& lhs, const PolyTensor& rhs) override {
        LOG_WARN("Invocation of in-place multiplication for PolyTensor may cause performance bottleneck!");
        if (lhs.total_elements != rhs.total_elements) {
            LOG_ERROR("Tensor shape mismatch in mul_assign!");
            exit(-1);
        }

        PolyTensor old_lhs = lhs.clone(); 
        int new_degree = old_lhs.degree + rhs.degree;
        size_t size = lhs.total_elements;
        
        lhs.flat_coeffs.assign(size * (new_degree + 1), 0);
        lhs.degree = new_degree;

        // 卷积：动态双规避并行策略
        if (size >= MVZK_OMP_SIZE_THRESHOLD) {
            // 【策略 A】: 大张量，内层空间并行
            for (int k = 0; k <= new_degree; ++k) {
                uint64_t* res_ptr = lhs.get_coeffs_ptr(k);
                int start_i = std::max(0, k - rhs.degree);
                int end_i = std::min(old_lhs.degree, k);

                #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
                for (size_t idx = 0; idx < size; ++idx) {
                    uint64_t sum = 0;
                    for (int i = start_i; i <= end_i; ++i) {
                        int j = k - i;
                        uint64_t term = mult_mod(old_lhs.get_coeffs_ptr(i)[idx], rhs.get_coeffs_ptr(j)[idx]);
                        sum = add_mod(sum, term);
                    }
                    res_ptr[idx] = sum;
                }
            }
        } else {
            // 【策略 B】: 极小张量，外层阶数并行
            #pragma omp parallel for if(new_degree >= MVZK_OMP_DEGREE_THRESHOLD && !omp_in_parallel())
            for (int k = 0; k <= new_degree; ++k) {
                uint64_t* res_ptr = lhs.get_coeffs_ptr(k);
                int start_i = std::max(0, k - rhs.degree);
                int end_i = std::min(old_lhs.degree, k);

                for (size_t idx = 0; idx < size; ++idx) {
                    uint64_t sum = 0;
                    for (int i = start_i; i <= end_i; ++i) {
                        int j = k - i;
                        uint64_t term = mult_mod(old_lhs.get_coeffs_ptr(i)[idx], rhs.get_coeffs_ptr(j)[idx]);
                        sum = add_mod(sum, term);
                    }
                    res_ptr[idx] = sum;
                }
            }
        }

        lhs.is_consumed = false;
        rhs.is_consumed = true;
        lhs.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;
    }

    /*
    void mul_assign(PolyTensor& lhs, const PolyTensor& rhs) override {
        // In-Place 乘法意味着 LHS 的阶数会膨胀
        // LHS = LHS * RHS
        std::cout << "[WARNING] Invocation of in-place multiplication for PolyTensor may cause performance bottleneck!" <<std::endl;
        if (lhs.total_elements != rhs.total_elements) {
            std::cerr << "[ERROR] Tensor shape mismatch in mul_assign!" << std::endl;
            exit(-1);
        }

        // 1. 备份旧数据 (必须备份，因为卷积计算依赖旧值，原地修改会覆盖)
        // 这是一个比较重的操作，所以通常建议用 mul 生成新对象。
        // 但为了完整性，我们这里实现它。
        PolyTensor old_lhs = lhs.clone(); // Deep Copy

        // 2. 调整 LHS 大小并清零
        int new_degree = old_lhs.degree + rhs.degree;
        size_t size = lhs.total_elements;
        
        // resize 并重置内容，因为我们要重新累加
        lhs.flat_coeffs.assign(size * (new_degree + 1), 0);
        lhs.degree = new_degree;

        // 3. 计算 Real Values
        const uint64_t* lhs_real_old = old_lhs.get_real_vals_ptr();
        const uint64_t* rhs_real = rhs.get_real_vals_ptr();
        uint64_t* lhs_real_new = lhs.get_real_vals_ptr();

        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            lhs_real_new[i] = mult_mod(lhs_real_old[i], rhs_real[i]);
        }

        // 4. 执行卷积 (利用 old_lhs 和 rhs 计算到 lhs)
        for (int i = 0; i <= old_lhs.degree; ++i) {
            const uint64_t* ptr_a = old_lhs.get_coeffs_ptr(i);
            
            for (int j = 0; j <= rhs.degree; ++j) {
                const uint64_t* ptr_b = rhs.get_coeffs_ptr(j);
                uint64_t* ptr_res = lhs.get_coeffs_ptr(i + j);

                #pragma omp parallel for
                for (size_t idx = 0; idx < size; ++idx) {
                    uint64_t term = mult_mod(ptr_a[idx], ptr_b[idx]);
                    ptr_res[idx] = add_mod(ptr_res[idx], term);
                }
            }
        }

        // 5. 状态管理
        lhs.is_consumed = false;
        rhs.is_consumed = true;
        lhs.is_from_fresh_matmul = lhs.is_from_fresh_matmul and rhs.is_from_fresh_matmul;
    }
        */

    void add_assign_const(PolyDelta& lhs, uint64_t val) override {
        lhs.coeffs[lhs.degree] = add_mod(lhs.coeffs[lhs.degree], val);
        lhs.is_consumed = false; // 结果是新的，活跃
    }

    void sub_assign_const(PolyDelta& lhs, uint64_t val) override {
        lhs.coeffs[lhs.degree] = add_mod(lhs.coeffs[lhs.degree], (PR - val));
        lhs.is_consumed = false;
    }

    // =========================================================
    // 3. 常数运算接口实现 (PolyDelta vs Constant)
    // =========================================================

    PolyDelta add_const(const PolyDelta& lhs, uint64_t val) override {
        // 【逻辑留空】：多项式加常数 (通常修改常数项或一次项，取决于定义)
        // 提示：VOLE关系 K = M + x*Delta，加常数通常意味着 x 变了

        // At the prover site, nothing but the highest coef (same as real_val) has to be changed.
        PolyDelta res;

        res = lhs.clone();
        res.coeffs[lhs.degree] = add_mod(res.coeffs[lhs.degree], val);

        res.is_consumed = false;
        lhs.is_consumed = true;

        return res;
    }

    PolyDelta sub_const(const PolyDelta& lhs, uint64_t val) override {
        // 【逻辑留空】：多项式减常数, x-val
        // Prover site, nearly identical to add const

        PolyDelta res;

        res = lhs.clone();
        res.coeffs[lhs.degree] = add_mod(res.coeffs[lhs.degree], (PR - val));

        res.is_consumed = false;
        lhs.is_consumed = true;

        return res;
    }

    PolyDelta sub_const_rev(uint64_t val, const PolyDelta& rhs) override {
        // 【逻辑留空】：常数减多项式 (val - Poly), val - x
        // Prover side: set all coefs to -coefs

        PolyDelta res;

        res = rhs.clone();
        for (size_t i = 0; i < res.coeffs.size(); i++){
            res.coeffs[i] = add_mod(0, (PR - res.coeffs[i]));
        }
        res.coeffs[rhs.degree] = add_mod(res.coeffs[rhs.degree], val);

        res.is_consumed = false;
        rhs.is_consumed = true;

        return res;
    }

    PolyDelta mul_const(const PolyDelta& lhs, uint64_t val) override {
        // 【逻辑留空】：多项式乘常数 (所有系数乘以 val), x*c
        // Prover site, all times c
        PolyDelta res;

        res = lhs.clone();
        for (size_t i = 0; i < res.coeffs.size(); i++){
            res.coeffs[i] = mult_mod(res.coeffs[i], val);
        }

        res.is_consumed = false;
        lhs.is_consumed = true;

        return res;
    }

    void add_assign_const(PolyTensor& lhs, uint64_t val) override {
        size_t size = lhs.total_elements;

        // 1. 更新 Real Values (所有元素都加 val)


        // 2. 更新系数 (Coefficients)
        // 根据 VOLE 承诺逻辑，常数加法通常作用于最高阶
        // P_new(X) = P_old(X) + val * X^degree
        
        // 获取最高阶系数块的指针
        uint64_t* highest_coeff_ptr = lhs.get_coeffs_ptr(lhs.degree);

        #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            highest_coeff_ptr[i] = add_mod(highest_coeff_ptr[i], val);
        }

        // 状态：结果是活跃的
        lhs.is_consumed = false;
    }

    PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs) override {
        // Verifier Site MatMul2D
        // 1. 维度检查
        if (lhs.shape.size() != 2 || rhs.shape.size() != 2 || lhs.shape[1] != rhs.shape[0]) {
            LOG_ERROR("Invalid shape for matmul!");
            exit(-1);
        }
        int M = lhs.shape[0];
        int K = lhs.shape[1];
        int N = rhs.shape[1];

        // 2. 构造结果 Z
        // 阶数相加: ResDeg = DegA + DegB
        int res_degree = lhs.degree + rhs.degree;
        
        // 构造函数会将内部内存 (flat_coeffs, flat_real_vals) 全部初始化为 0
        PolyTensor res({M, N}, res_degree); 

        // ==========================================
        // Part 1: 计算 Real Values (Z = X * Y)
        // ==========================================
        // 这是一个单纯的矩阵乘法，只调用一次内核

        // ==========================================
        // Part 2: 计算 Coefficients (处理交叉项)
        // ==========================================
        // 我们需要计算 Res 的每一阶 k (从 0 到 res_degree)
        
        for (int k = 0; k <= res_degree; ++k) {
            uint64_t* dst_ptr = res.get_coeffs_ptr(k); // 获取结果第 k 阶的矩阵指针

            // 寻找所有满足 i + j = k 的组合
            // i 的范围受到 lhs.degree 和 k 的双重限制
            int start_i = std::max(0, k - rhs.degree);
            int end_i   = std::min(lhs.degree, k);

            for (int i = start_i; i <= end_i; ++i) {
                int j = k - i;
                
                // 【核心逻辑】
                // 这里我们把 LHS 的第 i 阶系数矩阵 和 RHS 的第 j 阶系数矩阵相乘
                // 并 **累加** 到结果的第 k 阶矩阵中。
                // 这完美体现了 C_k += A_i * B_j
                matrix_mul_acc_kernel(
                    dst_ptr,                // Dst (+=)
                    lhs.get_coeffs_ptr(i),  // LHS Matrix A_i
                    rhs.get_coeffs_ptr(j),  // RHS Matrix B_j
                    M, K, N
                );
            }
        }

        // 3. 状态管理 (无需网络通信)
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
        
        // 1. Real Values: 永远直接相加 (无视阶数，因为 RealValue 是明文值/Payload)
        // 对应 add_assign 中的: lhs_real[i] = add_mod(lhs_real[i], rhs_real[i]);

        // 2. Coeffs: 高阶对齐逻辑 (MSB Alignment)
        // 对应 add_assign 中的 shift 逻辑
        if (!src.flat_coeffs.empty()) {
            int offset = 0;
            // 如果 src 阶数更高，计算 offset
            if (src.degree >= bias.degree) {
                offset = src.degree - bias.degree;
            } else{
                LOG_WARN("Non-implemented with bias.deg > WX.deg at Conv2D!");
            }
            
            for (int d = 0; d <= src.degree; ++d) {
                const uint64_t* b_ptr = nullptr;
                
                // 逻辑：d_src = d_bias + offset
                // 所以：d_bias = d - offset
                int d_bias = d - offset;

                if (has_bias && d_bias >= 0 && d_bias <= bias.degree) {
                    b_ptr = bias.get_coeffs_ptr(d_bias);
                }
                
                // Coeffs 本身不需要乘 scale，只是位置偏移了
                permute_and_add_bias_kernel(
                    src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d), b_ptr, 
                    1, // scale = 1
                    N, H_out, W_out, C_out
                );
            }
        }

        src.is_consumed = true;
        bias.is_consumed = true;
        dst.is_consumed = false;
    }

    void helper_linear_add_bias(
        PolyTensor& data, const PolyTensor& bias, int Rows, int Cols
    ) override {
        bool has_bias = (bias.total_elements > 0);

        // 1. Real Values: 直接加

        // 2. Coeffs: 高阶对齐 (Shift/Offset)
        if (!data.flat_coeffs.empty()) {
            int offset = 0;
            if (data.degree >= bias.degree) {
                offset = data.degree - bias.degree;
            } else {
                LOG_WARN("Linear: bias.deg > data.deg");
            }

            for (int d = 0; d <= data.degree; ++d) {
                const uint64_t* b_ptr = nullptr;
                
                // d_data = d_bias + offset
                int d_bias = d - offset;

                if (has_bias && d_bias >= 0 && d_bias <= bias.degree) {
                    b_ptr = bias.get_coeffs_ptr(d_bias);
                }

                if (b_ptr){
                    // If b_ptr is null, it means that this deg has been shifted.
                    add_bias_row_broadcast_kernel(data.get_coeffs_ptr(d), b_ptr, 1, Rows, Cols);
                }
                
            }
        }
    }

    void helper_permute_and_add_bias_1d(
        const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
        int N, int L_out, int C_out
    ) override {
        bool has_bias = (bias.total_elements > 0);

        // 1. Real Values: 永远直接相加

        // 2. Coeffs: 高阶对齐 (Offset Logic)
        if (!src.flat_coeffs.empty()) {
            int offset = 0;
            if (src.degree >= bias.degree) {
                offset = src.degree - bias.degree;
            } else {
                // YELLOW("[WARNING] Conv1D: bias.deg > src.deg");
            }

            for (int d = 0; d <= src.degree; ++d) {
                const uint64_t* b_ptr = nullptr;
                
                // d_src = d_bias + offset  =>  d_bias = d - offset
                int d_bias = d - offset;

                if (has_bias && d_bias >= 0 && d_bias <= bias.degree) {
                    b_ptr = bias.get_coeffs_ptr(d_bias);
                }

                // 调用 1D 专用内核
                /*
                if (b_ptr){
                    permute_and_add_bias_1d_kernel(
                    src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d), b_ptr, 
                    1, 
                    N, L_out, C_out);
                } */
                permute_and_add_bias_1d_kernel(
                    src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d), b_ptr, 
                    1, 
                    N, L_out, C_out);
            }
        }
        src.is_consumed = true;
    }

    // =========================================================
    // Non-linear operation
    // =========================================================

    PolyTensor relu(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) override {
        size_t n = x.total_elements;
        
        // 核心解耦：最低段绑定 Scale，剩余段均分剩余 bit
        int s_0 = scale; 
        int s_rest = (bitlen > (uint64_t)s_0 && digdec_k > 1) ? 
                     (bitlen - s_0 + digdec_k - 2) / (digdec_k - 1) : 0;

        // ======================================================
        // Step 1: Plaintext Pre-computation (包含非均匀拆解)
        // ======================================================
        std::vector<uint64_t> plain_abs_x(n);
        std::vector<uint64_t> plain_sign_bq(n); 
        const uint64_t* x_ptr = x.get_real_vals_ptr();
        uint64_t zero_point = PR >> 1;

        std::vector<uint64_t> flat_dig_dec(n * digdec_k, 0);
        uint64_t mask_0 = (1ULL << s_0) - 1;
        uint64_t mask_rest = (1ULL << s_rest) - 1;

        #pragma omp parallel for schedule(static) if(n >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(size_t i = 0; i < n; ++i) {
            uint64_t val = x_ptr[i];
            if (val > zero_point) {
                plain_abs_x[i] = PR - val; // |x|
                plain_sign_bq[i] = 0;      // b_Q = 0 (Negative)
            } else {
                plain_abs_x[i] = val;      // |x|
                plain_sign_bq[i] = 1;      // b_Q = 1 (Positive)
            }

            // 本地直接执行非均匀位拆解，抛弃旧的强制等长 helper
            uint64_t abs_val = plain_abs_x[i];
            flat_dig_dec[i * digdec_k + 0] = abs_val & mask_0;
            abs_val >>= s_0;
            for(int j = 1; j < digdec_k; ++j) {
                flat_dig_dec[i * digdec_k + j] = abs_val & mask_rest;
                abs_val >>= s_rest;
            }
        }

        // ======================================================
        // Step 2 & 3: Authenticate b_Q
        // ======================================================
        PolyTensor authenticated_bq = input(x.shape, plain_sign_bq);
        
        PolyTensor b_Q_check = authenticated_bq * (authenticated_bq - 1); 
        PolyTensor::store_zero_relation(b_Q_check, "[ReLU] b_Q");

        // ======================================================
        // Step 4 & 5: Reconstruct |x| (双轨重构 + 双查表)
        // ======================================================
        // 分别为截断段(s_0)和其余段(s_rest)创建两个范围检验表
        uint64_t table_size_0 = 1ULL << s_0;
        std::vector<uint64_t> rangeTableData_0(table_size_0);
        for (size_t i = 0; i < table_size_0; i++) rangeTableData_0[i] = i;
        RangeCheckTable rangeTable_0(rangeTableData_0);

        uint64_t table_size_rest = 1ULL << s_rest;
        std::vector<uint64_t> rangeTableData_rest(table_size_rest);
        for (size_t i = 0; i < table_size_rest; i++) rangeTableData_rest[i] = i;
        RangeCheckTable rangeTable_rest(rangeTableData_rest);

        PolyTensor X_recon; // 用于校验 (完整值)
        PolyTensor X_out;   // 用于输出 (截断值)

        for (int j = 0; j < digdec_k; ++j) {
            std::vector<uint64_t> seg_j_data(n);
            #pragma omp parallel for schedule(static) if(n >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(size_t i = 0; i < n; ++i) seg_j_data[i] = flat_dig_dec[i * digdec_k + j];

            PolyTensor authenticated_seg_j = input(x.shape, seg_j_data);

            if (j == 0) {
                // 最低段：专门承载 Scale 位数的区间证明
                rangeTable_0.range_check(authenticated_seg_j, "[ReLU] x_i range (low)");
                X_recon = authenticated_seg_j.clone(); // shift = 0 (即 1ULL << 0)
            } else {
                // 其余段：专门用于填满剩余位数的证明
                rangeTable_rest.range_check(authenticated_seg_j, "[ReLU] x_i range (high)");
                
                // 完整重构位移：Scale + 之前所有的 rest 段
                uint64_t shift_recon = 1ULL << (s_0 + (j - 1) * s_rest); 
                X_recon = X_recon + authenticated_seg_j * shift_recon;

                // 截断重构位移：丢弃了 Scale 的影响，直接从 0 开始铺 rest 段
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
        // Step 6, 7, 8: Consistency Check 
        // ======================================================
        PolyTensor term1 = authenticated_bq * (X_recon - x);
        PolyTensor term2 = (1 - authenticated_bq) * (X_recon + x);

        PolyTensor consistency = term1 + term2;
        PolyTensor::store_zero_relation(consistency, "[ReLU (Conv)] from prev");

        // ======================================================
        // Step 9: Compute Output
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

        // ======================================================
        // Step 1: Plaintext Pre-computation (Prover finds Max)
        // ======================================================
        std::vector<uint64_t> plain_max(M, 0);
        const uint64_t* in_ptr = pt_in.get_real_vals_ptr();

        #pragma omp parallel for collapse(2) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                for(int ho=0; ho<H_out; ++ho) {
                    for(int wo=0; wo<W_out; ++wo) {
                        int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        uint64_t x_max = 0;
                        bool first = true;

                        for(int kh=0; kh<kH; ++kh) {
                            for(int kw=0; kw<kW; ++kw) {
                                int hi = ho * stride - padding + kh;
                                int wi = wo * stride - padding + kw;
                                if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                    int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                    uint64_t val = in_ptr[in_idx];
                                    if(first) {
                                        x_max = val;
                                        first = false;
                                    } else {
                                        if(helper_plaintext_fp_greater(val, x_max)) x_max = val;
                                    }
                                }
                            }
                        }
                        plain_max[out_idx] = x_max;
                    }
                }
            }
        }

        PolyTensor pt_max = this->input({N, C, H_out, W_out}, plain_max);

        // ======================================================
        // Step 2 & 3: Protocol Execution (双轨 Range Table)
        // ======================================================
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

            #pragma omp parallel for collapse(4) schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(int n=0; n<N; ++n) {
                for(int c=0; c<C; ++c) {
                    for(int ho=0; ho<H_out; ++ho) {
                        for(int wo=0; wo<W_out; ++wo) {
                            int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;

                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                // pt_in_aligned.flat_real_vals[m] = pt_in.flat_real_vals[in_idx];
                                for(int d=0; d<=pt_in.degree; ++d) {
                                    pt_in_aligned.get_coeffs_ptr(d)[m] = pt_in.get_coeffs_ptr(d)[in_idx];
                                }
                            } else {
                                // pt_in_aligned.flat_real_vals[m] = add_mod(pt_max.flat_real_vals[m], PR - 1);
                                for(int d=0; d<pt_in.degree; ++d) {
                                    pt_in_aligned.get_coeffs_ptr(d)[m] = (d <= pt_max.degree) ? pt_max.get_coeffs_ptr(d)[m] : 0;
                                }
                                uint64_t max_coeff_high = (pt_in.degree <= pt_max.degree) ? pt_max.get_coeffs_ptr(pt_in.degree)[m] : 0;
                                pt_in_aligned.get_coeffs_ptr(pt_in.degree)[m] = add_mod(max_coeff_high, PR - 1);
                            }
                        }
                    }
                }
            }

            PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;

            // --- Upper Bound Check ---
            // 内部实现非均匀位拆解
            std::vector<uint64_t> flat_dig_dec(M * digdec_k, 0);
            uint64_t mask_0 = (1ULL << s_0) - 1;
            uint64_t mask_rest = (1ULL << s_rest) - 1;
            
            const uint64_t* y_hat_ptr = pt_y_hat_j.get_real_vals_ptr();

            #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(size_t i = 0; i < M; ++i) {
                uint64_t abs_val = y_hat_ptr[i];
                flat_dig_dec[i * digdec_k + 0] = abs_val & mask_0;
                abs_val >>= s_0;
                for(int d = 1; d < digdec_k; ++d) {
                    flat_dig_dec[i * digdec_k + d] = abs_val & mask_rest;
                    abs_val >>= s_rest;
                }
            }

            PolyTensor X_recon;

            for (int d = 0; d < digdec_k; ++d) {
                std::vector<uint64_t> seg_data(M);
                #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
                for(size_t i=0; i<M; ++i) seg_data[i] = flat_dig_dec[i * digdec_k + d];

                PolyTensor auth_seg = this->input({M}, seg_data);
                
                //  双轨检查与重构
                if (d == 0) {
                    rangeTable_0.range_check(auth_seg, "[MaxPool] x_i range (low)");
                    X_recon = auth_seg.clone(); 
                } else {
                    rangeTable_rest.range_check(auth_seg, "[MaxPool] x_i range (high)");
                    uint64_t shift_recon = 1ULL << (s_0 + (d - 1) * s_rest);
                    X_recon = X_recon + auth_seg * shift_recon;
                }
            }
            
            PolyTensor::store_relation(X_recon, pt_y_hat_j, "[MaxPool] greater than");

            // --- Existence Check ---
            if (j == 0) {
                pt_prod = pt_y_hat_j.clone();
                pt_y_hat_j.is_consumed = true;
            } else {
                pt_prod = pt_prod * pt_y_hat_j;
            }
        }

        PolyTensor::store_zero_relation(pt_prod, "[MaxPool] Existence check");

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

        // ======================================================
        // Step 1: Plaintext Pre-computation
        // ======================================================
        std::vector<uint64_t> plain_max(M, 0); 
        const uint64_t* in_ptr = pt_in.get_real_vals_ptr();

        #pragma omp parallel for collapse(2) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                for(int ho=0; ho<H_out; ++ho) {
                    for(int wo=0; wo<W_out; ++wo) {
                        int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        uint64_t x_max = 0; 

                        for(int kh=0; kh<kH; ++kh) {
                            for(int kw=0; kw<kW; ++kw) {
                                int hi = ho * stride - padding + kh;
                                int wi = wo * stride - padding + kw;
                                if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                    int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                    uint64_t val = in_ptr[in_idx];
                                    if(helper_plaintext_fp_greater(val, x_max)) x_max = val;
                                }
                            }
                        }
                        plain_max[out_idx] = x_max;
                    }
                }
            }
        }

        // Step 2: Authenticate x_max
        PolyTensor pt_max = this->input({N, C, H_out, W_out}, plain_max);

        //  解耦参数计算与双轨表初始化
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

        uint64_t mask_0 = (1ULL << s_0) - 1;
        uint64_t mask_rest = (1ULL << s_rest) - 1;

        // ======================================================
        // Step 3 & 9: Range Check x_max & Reconstruct Output (Truncated)
        // ======================================================
        std::vector<uint64_t> flat_max_dec(M * digdec_k, 0);
        #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(size_t i = 0; i < M; ++i) {
            uint64_t abs_val = plain_max[i];
            flat_max_dec[i * digdec_k + 0] = abs_val & mask_0;
            abs_val >>= s_0;
            for(int d = 1; d < digdec_k; ++d) {
                flat_max_dec[i * digdec_k + d] = abs_val & mask_rest;
                abs_val >>= s_rest;
            }
        }

        PolyTensor X_max_recon;
        PolyTensor X_max_out;

        for (int d = 0; d < digdec_k; ++d) {
            std::vector<uint64_t> seg_data(M);
            #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(size_t i=0; i<M; ++i) seg_data[i] = flat_max_dec[i * digdec_k + d];

            PolyTensor auth_seg = this->input({M}, seg_data);

            if (d == 0) {
                rangeTable_0.range_check(auth_seg, "[Integrated NL] x_max range (low)");
                X_max_recon = auth_seg.clone();
            } else {
                rangeTable_rest.range_check(auth_seg, "[Integrated NL] x_max range (high)");
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
        
        PolyTensor::store_relation(X_max_recon, pt_max, "[Integrated NL] x_max digdec check");

        // ======================================================
        // Step 4 - 8: Loop over sub-matrix elements
        // ======================================================
        PolyTensor pt_prod = pt_max.clone(); 

        for (int j = 0; j < h; ++j) {
            int kh = j / kW;
            int kw = j % kW;

            PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree);

            #pragma omp parallel for collapse(4) schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(int n=0; n<N; ++n) {
                for(int c=0; c<C; ++c) {
                    for(int ho=0; ho<H_out; ++ho) {
                        for(int wo=0; wo<W_out; ++wo) {
                            int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;

                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                //pt_in_aligned.flat_real_vals[m] = pt_in.flat_real_vals[in_idx];
                                for(int d=0; d<=pt_in.degree; ++d) {
                                    pt_in_aligned.get_coeffs_ptr(d)[m] = pt_in.get_coeffs_ptr(d)[in_idx];
                                }
                            } else {
                                //pt_in_aligned.flat_real_vals[m] = add_mod(pt_max.flat_real_vals[m], PR - 1);
                                for(int d=0; d<pt_in.degree; ++d) {
                                    pt_in_aligned.get_coeffs_ptr(d)[m] = (d <= pt_max.degree) ? pt_max.get_coeffs_ptr(d)[m] : 0;
                                }
                                uint64_t max_coeff_high = (pt_in.degree <= pt_max.degree) ? pt_max.get_coeffs_ptr(pt_in.degree)[m] : 0;
                                pt_in_aligned.get_coeffs_ptr(pt_in.degree)[m] = add_mod(max_coeff_high, PR - 1);
                            }
                        }
                    }
                }
            }

            PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;

            // 🌟 y_hat 也要用双轨拆解
            std::vector<uint64_t> flat_y_hat_dec(M * digdec_k, 0);
            const uint64_t* y_hat_ptr = pt_y_hat_j.get_real_vals_ptr();
            #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(size_t i = 0; i < M; ++i) {
                uint64_t abs_val = y_hat_ptr[i];
                flat_y_hat_dec[i * digdec_k + 0] = abs_val & mask_0;
                abs_val >>= s_0;
                for(int d = 1; d < digdec_k; ++d) {
                    flat_y_hat_dec[i * digdec_k + d] = abs_val & mask_rest;
                    abs_val >>= s_rest;
                }
            }

            PolyTensor X_recon;
            for (int d = 0; d < digdec_k; ++d) {
                std::vector<uint64_t> seg_data(M);
                #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
                for(size_t i=0; i<M; ++i) seg_data[i] = flat_y_hat_dec[i * digdec_k + d];

                PolyTensor auth_seg = this->input({M}, seg_data);

                if (d == 0) {
                    rangeTable_0.range_check(auth_seg, "[Integrated NL] x_max - x (low)");
                    X_recon = auth_seg.clone();
                } else {
                    rangeTable_rest.range_check(auth_seg, "[Integrated NL] x_max - x (high)");
                    uint64_t shift_recon = 1ULL << (s_0 + (d - 1) * s_rest);
                    X_recon = X_recon + auth_seg * shift_recon;
                }
            }
            
            PolyTensor::store_relation(X_recon, pt_y_hat_j, "[Integrated NL] x_max greater than (size = from Conv)");
            pt_prod = pt_prod * pt_y_hat_j;
        }

        PolyTensor::store_zero_relation(pt_prod, "[Integrated NL] existence check");
        pt_in.mark_consumed();

        X_max_out.shape = {N, C, H_out, W_out};
        return X_max_out;
    }


    PolyTensor avgpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, bool back_to_sum_pool) override {
        int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
        int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
        int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;
        
        // AvgPool 是线性操作，输出的阶数和输入的阶数一模一样！
        PolyTensor pt_out({N, C, H_out, W_out}, pt_in.degree);
        
        // 核心数学魔术：除以窗口面积 = 乘以面积的模逆元
        // PyTorch 默认 count_include_pad=True，所以无论有没有 padding，分母都是 kernel_size^2
        uint64_t pool_area = (uint64_t)(kernel_size * kernel_size);
        uint64_t inv_area = pow_mod(pool_area, PR - 2); // 费马小定理求逆元

        size_t total_elements = (size_t)N * C * H_out * W_out;

        #pragma omp parallel for collapse(4) if(total_elements >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                for(int ho=0; ho<H_out; ++ho) {
                    for(int wo=0; wo<W_out; ++wo) {
                        int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        
                        std::vector<uint64_t> sum_coeffs(pt_in.degree + 1, 0);

                        for(int kh=0; kh<kernel_size; ++kh) {
                            for(int kw=0; kw<kernel_size; ++kw) {
                                int hi = ho * stride - padding + kh;
                                int wi = wo * stride - padding + kw;
                                
                                // 如果在合法范围内，累加明文和多项式系数
                                if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                    int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                    
                                    for(int d=0; d<=pt_in.degree; ++d) {
                                        sum_coeffs[d] = add_mod(sum_coeffs[d], pt_in.get_coeffs_ptr(d)[in_idx]);
                                    }
                                }
                                // 如果是 Padding 区域，PyTorch 视作 0，我们加 0 等于没加，直接跳过
                            }
                        }

                        // 最后统一乘以模逆元（除法操作）
                        if (back_to_sum_pool == false){
                            for(int d=0; d<=pt_in.degree; ++d) {
                                pt_out.get_coeffs_ptr(d)[out_idx] = mult_mod(sum_coeffs[d], inv_area);
                            }
                        } else {
                            for(int d=0; d<=pt_in.degree; ++d) {
                                pt_out.get_coeffs_ptr(d)[out_idx] = sum_coeffs[d];
                            }
                        }
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
        
        // 安全检查：防止外部传入的 A 和 B 大小不对导致越界
        if (A.size() != (size_t)C || B.size() != (size_t)C) {
            LOG_ERROR("BatchNorm2D Scale/Shift vector size mismatch!");
            exit(-1);
        }

        PolyTensor pt_out({N, C, H, W}, pt_in.degree);

        const uint64_t* in_real = pt_in.get_real_vals_ptr();
        uint64_t* out_real = pt_out.get_real_vals_ptr();

        int max_deg = pt_in.degree;

        // 性能优化：提前把各阶的连续内存指针取出来，避免在最内层循环重复计算偏移量
        std::vector<const uint64_t*> in_coeffs(max_deg + 1);
        std::vector<uint64_t*> out_coeffs(max_deg + 1);
        for(int d = 0; d <= max_deg; ++d) {
            in_coeffs[d] = pt_in.get_coeffs_ptr(d);
            out_coeffs[d] = pt_out.get_coeffs_ptr(d);
        }

        #pragma omp parallel for collapse(2)
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                uint64_t a_val = A[c]; 
                uint64_t b_val = B[c];

                for(int h=0; h<H; ++h) {
                    for(int w=0; w<W; ++w) {
                        size_t idx = ((size_t)n * C + c) * H * W + h * W + w;

                        // 1. Real Value 正常乘加
                        out_real[idx] = add_mod(mult_mod(in_real[idx], a_val), b_val);

                        // 2. Coefficients 乘加 (直接使用外提的指针，速度极快)
                        for(int d=0; d<=max_deg; ++d) {
                            uint64_t coeff = mult_mod(in_coeffs[d][idx], a_val);
                            
                            // 常量平移只能发生在承载明文的最高阶上
                            if (d == max_deg) {
                                coeff = add_mod(coeff, b_val);
                            }
                            out_coeffs[d][idx] = coeff;
                        }
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
        size_t len = raw_data.size();
        
        // 1. 调整 pdList 大小
        pdList.resize(len);

        // 2. 准备缓冲区 (使用 vector 代替 new/delete)
        std::vector<__uint128_t> vole_returned(len);
        std::vector<uint64_t> lam(len);

        // 3. 执行 VOLE 扩展
        // vector.data() 返回内部数组指针，兼容 C 接口
        vole->extend(vole_returned.data(), len);

        for (size_t i = 0; i < len; i++) {
            // 解析 VOLE 输出
            // 假设: LOW64 是 MAC(M), HIGH64 是 Random Value(u)
            // 务必确保这与你的底层 emp-vole 实现一致
            uint64_t M_u = (uint64_t)LOW64(vole_returned[i]);
            uint64_t u   = (uint64_t)HIGH64(vole_returned[i]);
            uint64_t x   = raw_data[i];

            // 设置 PolyDelta
            pdList[i].degree = 1;
            pdList[i].coeffs.resize(2); // 0阶和1阶
            pdList[i].coeffs[0] = M_u;  // 常数项 = MAC
            pdList[i].coeffs[1] = x;    // 一次项 = 真实值
            pdList[i].is_consumed = false;

            // 计算修正值 (Masking)
            // 发送 diff = u + x
            // Verifier 侧需计算: K_new = K_old + diff * Delta
            lam[i] = add_mod(u, x);
        }

        // 4. 发送修正值
        io->send_data(lam.data(), len * sizeof(uint64_t));
        
        // 函数结束，vector 自动释放内存，无需 delete
    }

    PolyDelta input(uint64_t raw_data) override {
        __uint128_t vole_returned;
        uint64_t lam;
        PolyDelta res;

        vole->extend(&vole_returned, 1);
        uint64_t M_u = (uint64_t)LOW64(vole_returned);
        uint64_t u   = (uint64_t)HIGH64(vole_returned);
        uint64_t x   = raw_data;

        res.degree = 1;
        res.coeffs.resize(2);
        res.coeffs[0] = M_u;  // 常数项 = MAC
        res.coeffs[1] = x;    // 一次项 = 真实值
        res.is_consumed = false;

        lam = add_mod(u, x);
        io->send_data(&lam, sizeof(uint64_t));
        return res;
    }

    PolyTensor input(const std::vector<int>& shape, const std::vector<uint64_t>& raw_data) override {
        // 1. 初始化 Tensor (Degree = 1)
        PolyTensor res(shape, 1);
        size_t size = res.total_elements;

        // 检查输入数据大小
        if (raw_data.size() != size) {
            LOG_ERROR("Input data size mismatch! Expected " << size << ", got " << raw_data.size());
            exit(-1);
        }

        // 2. 准备缓冲区
        // VOLE 结果缓冲区
        std::vector<__uint128_t> vole_returned(size);
        // 待发送的 Mask 缓冲区 (lam)
        std::vector<uint64_t> lams(size);

        // 3. 批量执行 VOLE (Batch VOLE)
        // 这是极其关键的一步，把 N 次网络交互压缩为 1 次
        vole->extend(vole_returned.data(), size);

        // 4. 获取 PolyTensor 的原始指针 (Direct Memory Access)
        // 0阶系数 (存 MAC M_u)
        uint64_t* coeffs_0 = res.get_coeffs_ptr(0);
        // 1阶系数 (存 真实值 x)
        uint64_t* coeffs_1 = res.get_coeffs_ptr(1);
        // Real Values (存 真实值 x)

        // 5. 并行填充数据 (Parallel Fill)
        // 利用 SoA 布局优势，数据是连续的，OpenMP + SIMD 效率极高
        #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            // 提取 VOLE 数据
            // 假设 LOW64 是 MAC, HIGH64 是 Random u
            uint64_t M_u = (uint64_t)LOW64(vole_returned[i]);
            uint64_t u   = (uint64_t)HIGH64(vole_returned[i]);
            uint64_t x   = raw_data[i];

            // 填充 PolyTensor
            coeffs_0[i] = M_u; // 常数项
            coeffs_1[i] = x;   // 一次项

            // 计算 Mask: lam = u + x
            lams[i] = add_mod(u, x);
        }

        // 6. 批量发送 Mask (Batch Send)
        io->send_data(lams.data(), size * sizeof(uint64_t));

        // 7. 标记状态
        res.is_consumed = false;
        
        return res; // RVO 优化，无拷贝
    }

    // =========================================================
    // 5. Helper Function
    // =========================================================

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

        //std::cout << "[Prover] Prover naive VOLE res:";
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

    std::vector<PolyDelta> vole2pd(std::vector<__uint128_t>& vole_data, int size) override {
        std::vector<PolyDelta> res;
        res.resize(size);
        for (int i = 0; i < size; i++){
            res[i].is_consumed = false;
            res[i].degree = 1;
            res[i].coeffs.resize(2);
            res[i].coeffs[1] = HIGH64(vole_data[i]);
            // Note: this step is to make K = M + delta * x, instead of M = K + delta * x
            res[i].coeffs[0] = add_mod((PR - LOW64(vole_data[i])), 0);
        }
        return res;
    }

    void send_poly(const PolyDelta& pd) {
        // 1. 计算总大小
        // 结构: [degree (4 bytes)] + [real_val (8 bytes)] + [coeffs (N * 8 bytes)]
        size_t num_coeffs = pd.coeffs.size();
        size_t total_size = sizeof(int) + num_coeffs * sizeof(uint64_t);

        // 2. 准备缓冲区
        std::vector<uint8_t> buffer(total_size);
        uint8_t* ptr = buffer.data();

        // 3. 序列化 Degree
        memcpy(ptr, &pd.degree, sizeof(int));
        ptr += sizeof(int);

        // 4. 【新增】序列化 Real Value

        // 5. 序列化 Coefficients
        if (num_coeffs > 0) {
            memcpy(ptr, pd.coeffs.data(), num_coeffs * sizeof(uint64_t));
        }

        // 6. 发送
        io->send_data(buffer.data(), total_size);
    }

    void send_poly(const std::vector<PolyDelta>& pd_list) {
        size_t count = pd_list.size();

        // 1. 计算 Header 大小
        size_t total_size = sizeof(size_t); 

        // 2. 计算 Body 大小
        for (const auto& pd : pd_list) {
            // [Degree] + [RealVal] + [Coeffs]
            total_size += sizeof(int);
            total_size += pd.coeffs.size() * sizeof(uint64_t);
        }

        std::vector<uint8_t> buffer(total_size);
        uint8_t* ptr = buffer.data();

        // 3. 序列化 Count
        memcpy(ptr, &count, sizeof(size_t));
        ptr += sizeof(size_t);

        // 4. 序列化 Body
        for (const auto& pd : pd_list) {
            // 4.1 Degree
            memcpy(ptr, &pd.degree, sizeof(int));
            ptr += sizeof(int);

            // 4.2 【新增】Real Value

            // 4.3 Coefficients
            size_t num_coeffs = pd.coeffs.size();
            if (num_coeffs > 0) {
                memcpy(ptr, pd.coeffs.data(), num_coeffs * sizeof(uint64_t));
                ptr += num_coeffs * sizeof(uint64_t);
            }
        }

        io->send_data(buffer.data(), total_size);
    }

    std::vector<uint64_t> reveal(const PolyTensor& pt) override {
        // 1. 发送元数据 (长度和阶数)
        io->send_data(&pt.total_elements, sizeof(size_t));
        io->send_data(&pt.degree, sizeof(int));
        
        // 2. 发送所有系数 (flat_coeffs)
        // 注意：PolyTensor 采用 SoA 布局 [Deg0_Block][Deg1_Block]...
        size_t total_coeffs_size = pt.flat_coeffs.size();
        io->send_data(pt.flat_coeffs.data(), total_coeffs_size * sizeof(uint64_t));
        std::vector<uint64_t> placeholder;
        return placeholder;
    }

    // 打印 PolyDelta (Prover 视角: Val + Poly)
    void debug_print(const PolyDelta& pd, std::string name = "") override {
        std::cout << std::left;
        std::cout << "\033[32m[Prover] " << std::setw(15) << name << "\033[0m"; // 绿色
        std::cout << " (Deg=" << pd.degree << ") ";
        std::cout << "Val=" << std::setw(6) << pd.get_real_val() << " | Poly = ";
        
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

    // 打印 PolyTensor (Prover 视角: Val + Poly)
    void debug_print(const PolyTensor& pt, std::string name, int limit = 4) override {
        std::cout << "--------------------------------------------------------" << std::endl;
        size_t count = (size_t)limit < pt.total_elements ? (size_t)limit : pt.total_elements;

        for(size_t i=0; i<count; ++i) {
            std::string item_name = name + "[" + std::to_string(i) + "]";
            std::cout << "\033[32m[Prover] " << std::setw(15) << item_name << "\033[0m"; // 绿色
            std::cout << " (Deg=" << pt.degree << ") ";
            
            // 打印 Value
            if (i < pt.total_elements)
                std::cout << "Val=" << std::setw(6) << pt.get_real_vals_ptr()[i] << " | Poly = ";
            else 
                std::cout << "Val=ERR    | Poly = ";

            // 打印系数
            for (int d = 0; d <= pt.degree; ++d) {
                if (d > 0) std::cout << " + ";
                // 需要确保你有 get_coeffs_ptr 或者能访问成员
                // 假设继承或者 friend 关系能访问
                const uint64_t* ptr = pt.get_coeffs_ptr(d); 
                if (ptr) std::cout << ptr[i]; else std::cout << "?";
                
                if (d == 1) std::cout << "x";
                else if (d > 1) std::cout << "x^" << d;
            }
            std::cout << std::endl;
        }
        if (pt.total_elements > count) std::cout << " ... (" << (pt.total_elements - count) << " omitted)" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
    }

    bool debug_instant_check(const PolyDelta& pd) override {
        YELLOW("[Prover] Instant PD Check");
        debug_print(pd, "");
        send_poly(pd);
        return true;
    }

    bool debug_instant_check(const PolyTensor& pt) override {
        YELLOW("[Prover] Instant Tensor Check");
        debug_print(pt, "Sent Tensor");
        
        // 1. 发送元数据 (长度和阶数)
        io->send_data(&pt.total_elements, sizeof(size_t));
        io->send_data(&pt.degree, sizeof(int));
        
        // 2. 发送所有系数 (flat_coeffs)
        // 注意：PolyTensor 采用 SoA 布局 [Deg0_Block][Deg1_Block]...
        size_t total_coeffs_size = pt.flat_coeffs.size();
        io->send_data(pt.flat_coeffs.data(), total_coeffs_size * sizeof(uint64_t));
        
        return true;
    }


protected:
    // --- Helper 1: 处理 PolyDelta ---
    void accumulate_delta_buffer(PolyDelta& final_item) override {
        if (check_buffer.empty()) return;

        size_t size = check_buffer.size();
        std::vector<uint64_t> chi(size);
        this->prg.random_data(chi.data(), size * sizeof(uint64_t)); // 批量生成随机数

        for (size_t i = 0; i < size; i++){
            // final += chi[i] * item[i]
            // 复用 PolyDelta 的运算符重载
            chi[i] = chi[i] % PR;
            final_item += chi[i] * check_buffer[i];
            check_buffer[i].is_consumed = true;
        }
        check_buffer.clear();
    }

    // --- Helper 2: 处理 PolyTensor (SoA 极致性能版) ---
    /*
    void accumulate_tensor_buffer(PolyDelta& final_item) override {
        if (check_tensor_buffer.empty()) return;

        for (auto& tensor : check_tensor_buffer) {
            size_t len = tensor.total_elements;
            
            // 1. 为 Tensor 中的每个元素生成对应的随机系数
            std::vector<uint64_t> chi_vec(len);
            this->prg.random_data(chi_vec.data(), len * sizeof(uint64_t));

            #pragma omp parallel for if(len >= MVZK_OMP_SIZE_THRESHOLD)
            for(size_t i=0; i<len; ++i) {
                chi_vec[i] = chi_vec[i] % PR; 
            }

            // 2. 动态扩容 final_item
            if (final_item.degree < tensor.degree) {
                final_item.coeffs.resize(tensor.degree + 1, 0);
                final_item.degree = tensor.degree;
            }

            // 3. 累加 Real Values
            const uint64_t* real_ptr = tensor.get_real_vals_ptr();
            uint64_t sum_real = 0;
            
            // 单线程累加 (add_mod 不支持简单的 omp reduction)
            // 瓶颈通常在 mult_mod，如果需要可以用分块并行优化
            for (size_t i = 0; i < len; ++i) {
                uint64_t term = mult_mod(real_ptr[i], chi_vec[i]);
                sum_real = add_mod(sum_real, term);
            }
            final_item.real_val = add_mod(final_item.real_val, sum_real);

            // 4. 累加 Coefficients (按阶遍历 - SoA 优势)
            for (int d = 0; d <= tensor.degree; ++d) {
                const uint64_t* coeff_ptr = tensor.get_coeffs_ptr(d);
                uint64_t sum_coeff = 0;

                // 连续内存访问，对 Cache 非常友好
                for (size_t i = 0; i < len; ++i) {
                     uint64_t term = mult_mod(coeff_ptr[i], chi_vec[i]);
                     sum_coeff = add_mod(sum_coeff, term);
                }

                // 加到最终结果的对应阶上
                final_item.coeffs[d] = add_mod(final_item.coeffs[d], sum_coeff);
            }

            tensor.is_consumed = true;
        }
        check_tensor_buffer.clear();
    }*/

    // --- Helper 2: 处理 PolyTensor (SoA 极致性能版) ---
    void accumulate_tensor_buffer(PolyDelta& final_item) override {
        if (check_tensor_buffer.empty()) return;

        if (final_item.coeffs.empty()) {
            final_item.coeffs.resize(final_item.degree + 1, 0);
        }

        // 1. 【新增】扫描寻找全局最大阶数
        int max_deg = final_item.degree;
        for (auto& tensor : check_tensor_buffer) {
            max_deg = std::max(max_deg, tensor.degree);
        }

        // 2. 【新增】将已有的 final_item 拔高对齐
        if (final_item.degree < max_deg) {
            int offset_diff = max_deg - final_item.degree;
            std::vector<uint64_t> new_coeffs(max_deg + 1, 0);
            for (size_t i = 0; i <= final_item.degree; ++i) {
                new_coeffs[i + offset_diff] = final_item.coeffs[i];
            }
            final_item.coeffs = std::move(new_coeffs);
            final_item.degree = max_deg;
        }

        // 3. 遍历 Tensor 进行累加
        for (auto& tensor : check_tensor_buffer) {
            size_t len = tensor.total_elements;
            
            std::vector<uint64_t> chi_vec(len);
            this->prg.random_data(chi_vec.data(), len * sizeof(uint64_t));

            #pragma omp parallel for if(len >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(size_t i=0; i<len; ++i) {
                chi_vec[i] = chi_vec[i] % PR; 
            }

            // 【核心修复】：计算当前约束需要的移位量 (Offset)
            // 这保证了当前 tensor 的最高阶 tensor.degree 一定会被加到 max_deg 上！
            int offset = max_deg - tensor.degree;

            // 注意：这里必须是 <=，我们要包含明文！
            for (int d = 0; d <= tensor.degree; ++d) {
                const uint64_t* coeff_ptr = tensor.get_coeffs_ptr(d);
                uint64_t sum_coeff = 0;

                if (len >= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel()) {
                    #pragma omp parallel
                    {
                        uint64_t local_sum = 0;
                        #pragma omp for nowait
                        for (size_t i = 0; i < len; ++i) {
                            uint64_t term = mult_mod(coeff_ptr[i], chi_vec[i]);
                            local_sum = add_mod(local_sum, term);
                        }
                        #pragma omp critical
                        {
                            sum_coeff = add_mod(sum_coeff, local_sum);
                        }
                    }
                } else {
                    for (size_t i = 0; i < len; ++i) {
                        uint64_t term = mult_mod(coeff_ptr[i], chi_vec[i]);
                        sum_coeff = add_mod(sum_coeff, term);
                    }
                }
                
                // 【核心修复】：加上 offset！
                final_item.coeffs[d + offset] = add_mod(final_item.coeffs[d + offset], sum_coeff);
            }

            tensor.is_consumed = true;
        }
        check_tensor_buffer.clear();
    }

    // =========================================================
    // 5. LUT Section
    // =========================================================
    int register_lut_table(const std::vector<std::pair<uint64_t, uint64_t>>& data) override {
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
                LOG_INFO("LUT Table Deduplication: Reusing ID " << i);
                return (int)i;
            }
        }
        auto entry = std::make_unique<LUTData>(data);
        for (const auto& kv : data) {
            entry->version_tracker[kv] = 0;
        }
        lut_tables.push_back(std::move(entry));
        return (int)lut_tables.size() - 1;
    }

    void submit_lut_check(const PolyTensor& keys, const PolyTensor& vals, int table_id) override {
        // When submit to the lut, there is only the public table, read and a version tracker. Storing Wirte is not necessary.
        // 1. 安全检查：防止 ID 越界
        if (table_id < 0 || table_id >= lut_tables.size()) {
            LOG_ERROR("[ERROR] Invalid Table ID: " << table_id);
            return;
        }

        // 获取数据指针
        // Find the right LUT table instance.
        LUTData* table = lut_tables[table_id].get();

        // 2. 准备请求对象 (数据私有化)
        // 我们必须 Clone，因为 keys/vals 可能是用户栈上的临时变量，马上会被销毁
        LUTData::Request req;
        req.keys = keys.clone();
        req.vals = vals.clone();

        // =========================================================
        // 3. 【Prover 核心逻辑】预计算 Hint (Eager Evaluation)
        // =========================================================
        // 此时 keys/vals 数据还在 CPU L1/L2 缓存里，遍历它们非常快。
        size_t len = keys.total_elements;
        req.v_j_hint.reserve(len);
        
        // 获取裸指针加速访问
        const uint64_t* k_ptr = keys.get_real_vals_ptr();
        const uint64_t* v_ptr = vals.get_real_vals_ptr();

        for(size_t i = 0; i < len; ++i) {
            std::pair<uint64_t, uint64_t> item = {k_ptr[i], v_ptr[i]};
            
            // 逻辑：
            // 1. version_tracker[item] 返回当前版本 (例如 0)
            // 2. 将其填入 hint
            // 3. operator++ 将 tracker 里的值加 1 (变成 1)
            req.v_j_hint.push_back(table->version_tracker[item]++);
        }

        // 4. 入队 (Enqueue)
        table->buffer.push_back(std::move(req));
        table->current_buffer_size += len;

        keys.is_consumed = true;
        vals.is_consumed = true;

        // 5. 阈值检查 (Auto-Flush)
        // 如果积压太多，为了防止内存爆掉，先处理一波
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
        // Phase 1: Serialization & Commit (Prover 独有逻辑)
        // =========================================================
        
        // 1.1 准备 Hint (Query 时的版本号)
        std::vector<uint64_t> hints_flat;
        hints_flat.reserve(N_Query);
        for (const auto& req : table->buffer) {
            hints_flat.insert(hints_flat.end(), req.v_j_hint.begin(), req.v_j_hint.end());
        }

        // 1.2 准备 Final Version (表的最终状态)
        // 【安全核心】：必须遍历 table->data，不能遍历 version_tracker！
        // 这样确保了任何不在原始表里的非法 Key 无法进入 Final 集合
        std::vector<uint64_t> finals_flat;
        finals_flat.reserve(N_Table);
        for (const auto& kv : table->data) {
            // 利用 map 特性：如果之前没查过，tracker[key] 自动返回 0 (正确)
            finals_flat.push_back(table->version_tracker[kv]);
        }

        // 1.3 VOLE 提交 (网络交互)
        PolyTensor V_J = this->input({(int)N_Query, 1}, hints_flat);
        PolyTensor V_FINAL = this->input({(int)N_Table, 1}, finals_flat);
        // TODO: Set to one round send

        //debug_print(V_J, "V_j");
        //debug_print(V_FINAL, "V_Final");

        // =========================================================
        // Phase 2: Challenge (Prover 接收随机数)
        // =========================================================
        block seed;
        io->recv_data(&seed, sizeof(block));
        this->prg.reseed(&seed);

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
        // Phase 3: RLC & Term Construction
        // =========================================================

        // 3.1 拼接 Buffer (AoS -> SoA)

        auto [Big_Keys, Big_Vals] = concat_buffer_tensors(table);

        //debug_print(Big_Keys, "Big_Keys");
        //debug_print(Big_Vals, "Big_Vals");

        // 3.2 计算 Query Read 项
        // Term_Read = K*sx + V*sy + v*sv - r
        PolyTensor Term_Q_Read = (Big_Keys * sx) + (Big_Vals * sy) + (V_J * sv) - r;
        
        //debug_instant_check(Term_Q_Read);

        // 3.3 计算 Query Write 项 (隐式推导)
        // Term_Write = Term_Read + sv (即 v -> v+1)
        PolyTensor Term_Q_Write = Term_Q_Read + sv;

        //debug_instant_check(Term_Q_Write);

        // 3.4 计算 Final 项 和 Init 标量 (遍历 table->data)
        std::vector<uint64_t> table_consts;
        table_consts.reserve(N_Table);
        uint64_t P_Init_Scalar = 1; // 标量累乘器 (u64)

        for (const auto& kv : table->data) {
            // Process write const
            uint64_t key = kv.first;
            uint64_t val = kv.second;

            // 计算常数部分: K*sx + V*sy
            uint64_t term_val = add_mod(mult_mod(key, sx), mult_mod(val, sy));
            table_consts.push_back(term_val);

            // Init 项：Version=0，所以直接是 (term_val - r)
            // 这一步是【安全核心】：只有 data 里有的 Key 才能进入 Init 积
            P_Init_Scalar = mult_mod(P_Init_Scalar, add_mod(term_val, PR - r));
            //P_Init_Scalar *= (term_val - r);
        }

        // 构造 PolyTensor 形式的 Final 项
        // Term_Final = (K*sx + V*sy) + v_final*sv - r
        PolyTensor P_Table_Const = PolyTensor::from_public({(int)N_Table, 1}, table_consts);
        
        PolyTensor Term_Final = P_Table_Const + (V_FINAL * sv) - r;
        //debug_instant_check(Term_Final);

        // =========================================================
        // Phase 4: Grand Product (计算 LHS 和 RHS)
        // =========================================================

        // 4.1 准备 LHS (Read U Final)
        std::vector<PolyTensor> lhs_list = split_to_scalars(Term_Q_Read);
        std::vector<PolyTensor> final_list = split_to_scalars(Term_Final);
        
        // 合并列表，并行计算
        lhs_list.reserve(lhs_list.size() + final_list.size());
        lhs_list.insert(lhs_list.end(), 
                        std::make_move_iterator(final_list.begin()), 
                        std::make_move_iterator(final_list.end()));

        // TODO: Realize this, check split_to_scalars, verifier side lut. concat_buffer_tensors
        PolyTensor P_LHS = fast_tree_product(lhs_list);

        // 4.2 准备 RHS (Write Only)
        std::vector<PolyTensor> rhs_list = split_to_scalars(Term_Q_Write);
        PolyTensor P_RHS_Write = fast_tree_product(rhs_list);

        // =========================================================
        // Phase 5: Check Zero
        // =========================================================
        
        // Z = LHS - (Write_Product * Init_Scalar)
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

        this->submit_tensor_to_buffer(std::move(Z));
    }

    int register_range_check_table(const std::vector<uint64_t>& data) override {
        if (data[0] != real2fp(0)){
            LOG_ERROR("Not implemented: Range Table must start from 0 in F_p, range check with offset is not implemented currently.");
            exit(-1);
        }
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
                LOG_INFO("Range Check Table Deduplication: Reusing ID " << i);
                return (int)i;
            }
        }
        auto entry = std::make_unique<RangeCheckData>(data);
        /*
        for (const auto& kv : data) {
            entry->version_tracker[kv] = 0;
        }*/

        range_check_tables.push_back(std::move(entry));
        return (int)range_check_tables.size() - 1;
    }

    /*
    void submit_range_check(const PolyTensor& x, int table_id) override {
        // When submit to the range check, there is only the public table, read and a version tracker. Storing Write is not necessary.
        // 1. 安全检查：防止 ID 越界
        if (table_id < 0 || table_id >= range_check_tables.size()) {
            std::cerr << "[ERROR] Invalid Table ID: " << table_id << std::endl;
            return;
        }

        // 获取数据指针
        // Find the right LUT table instance.
        RangeCheckData* table = range_check_tables[table_id].get();

        // 2. 准备请求对象 (数据私有化)
        // 我们必须 Clone，因为 keys/vals 可能是用户栈上的临时变量，马上会被销毁
        RangeCheckData::Request req;
        req.vals = x.clone();

        // =========================================================
        // 3. 【Prover 核心逻辑】预计算 Hint (Eager Evaluation)
        // =========================================================
        // 此时 keys/vals 数据还在 CPU L1/L2 缓存里，遍历它们非常快。
        size_t len = x.total_elements;
        req.v_j_hint.reserve(len);
        
        // 获取裸指针加速访问
        const uint64_t* x_ptr = x.get_real_vals_ptr();

        for(size_t i = 0; i < len; ++i) {
            uint64_t item = x_ptr[i];
            
            // 逻辑：
            // 1. version_tracker[item] 返回当前版本 (例如 0)
            // 2. 将其填入 hint
            // 3. operator++ 将 tracker 里的值加 1 (变成 1)
            req.v_j_hint.push_back(table->version_tracker[item]++);
        }

        // 4. 入队 (Enqueue)
        table->buffer.push_back(std::move(req));
        table->current_buffer_size += len;

        // We set consumed here, but it is a little risky. However, it seems ok though.
        x.is_consumed = true;

        // 5. 阈值检查 (Auto-Flush)
        // 如果积压太多，为了防止内存爆掉，先处理一波
        if (table->current_buffer_size >= MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD) {
            process_range_check_batch(table); 
        }
    }
        */
    
    void submit_range_check(const PolyTensor& x, int table_id) override {
        if (table_id < 0 || table_id >= range_check_tables.size()) {
            LOG_ERROR("Invalid Table ID: " << table_id);
            return;
        }

        RangeCheckData* table = range_check_tables[table_id].get();
        size_t threshold = MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD;
        
        // 【优化】获取 Tracker 的裸指针，避免 vector 边界检查开销
        uint64_t* tracker_ptr = table->version_tracker.data();
        size_t tracker_size = table->version_tracker.size();

        // 【优化】获取 Input 的裸指针
        const uint64_t* x_ptr = x.get_real_vals_ptr();
        size_t len = x.total_elements;
        
        size_t offset = 0;

        // 循环切片逻辑 (保持你的逻辑，但优化内部实现)
        while (offset < len) {
            size_t capacity = threshold - table->current_buffer_size;
            size_t current_chunk = std::min(capacity, len - offset);

            // 1. 构造子张量 (保持不变)
            PolyTensor sub_x({(int)current_chunk}, x.degree);
            if (!x.flat_coeffs.empty()) {
                for (int d = 0; d <= x.degree; ++d) {
                    std::memcpy(sub_x.get_coeffs_ptr(d), x.get_coeffs_ptr(d) + offset, current_chunk * sizeof(uint64_t));
                }
            }

            // 2. 构造 Request
            RangeCheckData::Request req;
            req.v_j_hint.reserve(current_chunk);
            
            // 【核心优化】消灭 std::map 查找，改为数组索引
            // 注意：这里我们假设 x_ptr[i] 的值一定在 [0, tracker_size) 范围内
            // 对于 8-bit/16-bit 分解，这通常是成立的。
            const uint64_t* chunk_ptr = sub_x.get_real_vals_ptr();
            
            // 如果能保证安全性，这里甚至可以手动展开循环
            for(size_t i = 0; i < current_chunk; ++i) {
                uint64_t item = chunk_ptr[i];
                
                // Debug 模式下可以开这个检查
                #ifndef NDEBUG
                if (item >= tracker_size) {
                    LOG_ERROR("Range Check Value OOB! Val: " << item << " Size: " << tracker_size);
                    exit(-1);
                }
                #endif

                // O(1) 访问 + 自增
                req.v_j_hint.push_back(tracker_ptr[item]++);
            }

            req.vals = std::move(sub_x); 
            table->buffer.push_back(std::move(req));
            table->current_buffer_size += current_chunk;

            if (table->current_buffer_size >= threshold) {
                process_range_check_batch(table); 
            }
            offset += current_chunk;
        }
        x.is_consumed = true;
    }

    void process_range_check_batch(RangeCheckData* table) override {
        if (table->buffer.empty()) return;

        size_t N_Query = table->current_buffer_size;
        size_t N_Table = table->data.size(); // 注意：这里 data.size() == version_tracker.size()

        // Phase 1.1: Hint (保持不变)
        std::vector<uint64_t> hints_flat;
        hints_flat.reserve(N_Query);
        for (const auto& req : table->buffer) {
            hints_flat.insert(hints_flat.end(), req.v_j_hint.begin(), req.v_j_hint.end());
        }

        // Phase 1.2: Final Version (优化遍历)
        // 【修改】直接拷贝 vector，不用 map 迭代器
        // 这一步可以用 memcpy 或者 assign 进一步加速，但循环也足够快了
        // 逻辑：table->data 是 [0, 1, ..., N-1]，对应 version_tracker 的下标
        // 所以我们只需要把 version_tracker 里的值拷出来就行
        // 注意：这里假设 table->data 是顺序的 0..N-1。如果不是，需要用索引
        // 为了通用性，我们还是遍历 table->data，用它的值作为下标去取 tracker
        std::vector<uint64_t> finals_flat;
        finals_flat.reserve(N_Table);
        
        const uint64_t* tracker_ptr = table->version_tracker.data();
        for (const auto& kv : table->data) {
            // O(1) 取值
            finals_flat.push_back(tracker_ptr[kv]);
        }

        // Phase 1.3: VOLE (保持不变)
        PolyTensor V_J = this->input({(int)N_Query, 1}, hints_flat);
        PolyTensor V_FINAL = this->input({(int)N_Table, 1}, finals_flat);

        // Phase 2: Challenge (保持不变)
        block seed;
        io->recv_data(&seed, sizeof(block));
        this->prg.reseed(&seed);
        uint64_t sy, sv, r;
        this->prg.random_data(&sy, sizeof(uint64_t));
        this->prg.random_data(&sv, sizeof(uint64_t));
        this->prg.random_data(&r, sizeof(uint64_t));
        sy = sy % PR; sv = sv % PR; r = r % PR;

        // Phase 3: RLC (保持不变)
        auto Big_Vals = concat_buffer_tensors(table);
        PolyTensor Term_Q_Read = (Big_Vals * sy) + (V_J * sv) - r;
        PolyTensor Term_Q_Write = Term_Q_Read + sv;

        std::vector<uint64_t> table_consts;
        table_consts.reserve(N_Table);
        uint64_t P_Init_Scalar = 1;

        for (const auto& kv : table->data) {
            uint64_t term_val = mult_mod(kv, sy);
            table_consts.push_back(term_val);
            P_Init_Scalar = mult_mod(P_Init_Scalar, add_mod(term_val, PR - r));
        }

        PolyTensor P_Table_Const = PolyTensor::from_public({(int)N_Table, 1}, table_consts);
        PolyTensor Term_Final = P_Table_Const + (V_FINAL * sv) - r;

        // Phase 4: Grand Product
        std::vector<PolyTensor> lhs_list = this->split_to_scalars(Term_Q_Read);
        std::vector<PolyTensor> final_list = this->split_to_scalars(Term_Final);
        
        lhs_list.reserve(lhs_list.size() + final_list.size());
        lhs_list.insert(lhs_list.end(), 
                        std::make_move_iterator(final_list.begin()), 
                        std::make_move_iterator(final_list.end()));

        PolyTensor P_LHS = fast_tree_product(lhs_list);

        std::vector<PolyTensor> rhs_list = this->split_to_scalars(Term_Q_Write);
        PolyTensor P_RHS_Write = fast_tree_product(rhs_list);

        // Phase 5: Check Zero
        PolyTensor Z = P_LHS - (P_RHS_Write * P_Init_Scalar);
        Z.is_constraint = true;
        

        // Phase 6: Cleanup (优化)
        table->buffer.clear();
        table->current_buffer_size = 0;

        // 【修改】快速清零 vector，比 map 遍历删除快得多
        // std::fill 或者 memset 都可以
        std::fill(table->version_tracker.begin(), table->version_tracker.end(), 0);
        this->submit_tensor_to_buffer(std::move(Z));
    }
};

#endif