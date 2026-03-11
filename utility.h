#pragma once

#include <iostream>
#include <cmath> // for std::round
#include <vector>
#include <type_traits>
#include <algorithm>
#include "config.h"
#include "emp-zk/emp-vole/utility.h"
#include <utility>   // for std::pair
#include <functional> // for std::hash
#include <cstdint>   // for uint64_t
#include <cstddef>   // for size_t
#include <omp.h>
#include "data_type/PolyTensor.h"

// ==========================================
// 1. 自带换行的颜色输出 (std::cout << ... << \n)
// ==========================================

// --- 原有保留 ---
#define GREEN(STRING)  std::cout<<"\033[32m"<<STRING<<"\033[m\n"
#define RED(STRING)    std::cout<<"\033[31m"<<STRING<<"\033[m\n"
#define YELLOW(STRING) std::cout<<"\033[33m"<<STRING<<"\033[m\n"

// --- 新增颜色 ---
#define BLUE(STRING)    std::cout<<"\033[34m"<<STRING<<"\033[m\n"
#define MAGENTA(STRING) std::cout<<"\033[35m"<<STRING<<"\033[m\n" // 紫红色/洋红
#define CYAN(STRING)    std::cout<<"\033[36m"<<STRING<<"\033[m\n" // 青色/蓝绿
#define WHITE(STRING)   std::cout<<"\033[37m"<<STRING<<"\033[m\n"


// ==========================================
// 2. 不带换行符的颜色输出 (std::cout << ...)
// ==========================================

// --- 原有保留 ---
#define YELLOW_WITHOUTENTER(STRING) std::cout<<"\033[33m"<<STRING<<"\033[m"
#define GREEN_WITHOUTENTER(STRING)  std::cout<<"\033[32m"<<STRING<<"\033[m"
#define RED_WITHOUTENTER(STRING)    std::cout<<"\033[31m"<<STRING<<"\033[m"

// --- 新增颜色 ---
#define BLUE_WITHOUTENTER(STRING)    std::cout<<"\033[34m"<<STRING<<"\033[m"
#define MAGENTA_WITHOUTENTER(STRING) std::cout<<"\033[35m"<<STRING<<"\033[m"
#define CYAN_WITHOUTENTER(STRING)    std::cout<<"\033[36m"<<STRING<<"\033[m"
#define WHITE_WITHOUTENTER(STRING)   std::cout<<"\033[37m"<<STRING<<"\033[m"


// ==========================================
// 3. 单纯的颜色控制字符 (用于拼接字符串)
// ==========================================

// --- 原有保留 ---
#define COLOR_GREEN  "\033[32m"
#define COLOR_RED    "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_NORMAL "\033[m"

// --- 新增颜色 ---
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_WHITE   "\033[37m"

// ==========================================
// 4. 工业级日志分级系统 (Log Levels)
// ==========================================

// 定义日志等级 (数字越小，输出的内容越多)
enum LogLevel {
    LEVEL_DEBUG = 0, // 调试信息，极度啰嗦，排查 Bug 时开启
    LEVEL_INFO  = 1, // 关键流程信息，如网络层开始、耗时统计 (默认级别)
    LEVEL_WARN  = 2, // 警告，如 FFT 建议、Threshold 触发等
    LEVEL_ERROR = 3, // 致命错误，通常伴随 exit(-1)
    LEVEL_NONE  = 4  // 静默模式，压测极限性能时开启
};

// 全局日志等级状态 (使用 inline/static 保证 Header-only 且防止多重定义)
inline LogLevel& GlobalLogLevel() {
    // 默认设置为 INFO 级别。
    // 在正式跑 50 层网络测速时，可以改为 LEVEL_WARN 或 LEVEL_NONE
    static LogLevel level = MVZK_CONFIG_DEFAULT_LEVEL_INFO; 
    return level;
}

// 提供一个供 main 函数调用的接口
inline void SetLogLevel(LogLevel level) {
    GlobalLogLevel() = level;
}

// ==========================================
// 5. 日志输出宏 (支持 `<<` 流式拼接，且带性能阻断)
// ==========================================
// 核心技巧：do { if(...) { ... } } while(0)
// 如果当前级别高于宏的级别，内部的代码（包括耗时的参数运算）根本不会被执行！

#define LOG_DEBUG(MSG) \
    do { if (GlobalLogLevel() <= LEVEL_DEBUG) { \
        std::cout << COLOR_CYAN << "[DEBUG] " << MSG << COLOR_NORMAL << "\n"; \
    } } while(0)

#define LOG_INFO(MSG) \
    do { if (GlobalLogLevel() <= LEVEL_INFO) { \
        std::cout << COLOR_GREEN << "[INFO]  " << MSG << COLOR_NORMAL << "\n"; \
    } } while(0)

#define LOG_WARN(MSG) \
    do { if (GlobalLogLevel() <= LEVEL_WARN) { \
        std::cout << COLOR_YELLOW << "[WARN]  " << MSG << COLOR_NORMAL << "\n"; \
    } } while(0)

#define LOG_ERROR(MSG) \
    do { if (GlobalLogLevel() <= LEVEL_ERROR) { \
        std::cerr << COLOR_RED << "[ERROR] " << MSG << COLOR_NORMAL << "\n"; \
    } } while(0)

inline uint64_t pow_mod(uint64_t a, uint64_t deg) {
    assert(deg >= 0);
    uint64_t res = 1;
    uint64_t base = a;

    if (deg == 1) return a;
    
    while (deg > 0) {
        if (deg & 1) {
            res = mult_mod(res, base);
        }
        base = mult_mod(base, base);
        deg >>= 1;
    }
    return res;
}

// Comm Data
inline uint64_t comm(NetIO** io_arr, int num){
    uint64_t c = 0;
    for (int i = 0; i < num; i++){
        c += io_arr[i]->counter;
    }
    return c;
}

/*
LOG_WARN("FFT Optimization is recommended, deg = " << final_checked_item.degree);
LOG_DEBUG("Check buffer threshold reached, instant check now."); // 这种频繁触发的建议放 DEBUG
LOG_INFO("Executing Layer 1 (3 Blocks)...");
LOG_ERROR("Shape mismatch! Expected " << expected_size << ", got " << actual);
*/

// 功能：Dst += LHS * RHS
// 优化：IKJ Loop Reordering + OpenMP
/*
static void matrix_mul_acc_kernel(
    uint64_t* __restrict__ Dst, 
    const uint64_t* __restrict__ LHS, 
    const uint64_t* __restrict__ RHS,
    int M, int K, int N
) {

    size_t total_work = (size_t)M * N * K;
    // 外层循环并行化 (按行分块)
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        // [优化关键] 这里的循环顺序是 i -> k -> j
        // 这被称为 "IKJ" 优化，它将内存访问线性化了
        for (int k = 0; k < K; ++k) {
            
            // 1. 提取 LHS[i, k] 的值 (标量)
            // 在这一轮内部循环中，它是常量，放入寄存器
            uint64_t a_val = LHS[i * K + k];
            
            // 如果 A 的元素是 0，跳过整行计算 (稀疏优化)
            if (a_val == 0) continue;

            // 2. 遍历 RHS 的第 k 行 (对应 Row k, Col 0...N)
            // 此时 RHS 的访问是连续的 (stride = 1)
            // Dst 的访问也是连续的 (stride = 1)
            for (int j = 0; j < N; ++j) {
                uint64_t b_val = RHS[k * N + j];
                
                // 计算乘积 (模乘)
                // Dst[i, j] += A[i, k] * B[k, j]
                uint64_t prod = mult_mod(a_val, b_val);
                
                // 累加到目标矩阵 (Cumulative)
                // 这里必须是 add_mod (+=)，因为 Prover 的卷积逻辑依赖累加
                Dst[i * N + j] = add_mod(Dst[i * N + j], prod);
            }
        }
    }
}
    */

// ... existing code ...

static void matrix_mul_acc_kernel(
    uint64_t* __restrict__ Dst, 
    const uint64_t* __restrict__ LHS, 
    const uint64_t* __restrict__ RHS,
    int M, int K, int N
) {
    // 估算总工作量
    size_t total_work = (size_t)M * N * K;

    // 如果工作量太小，强制串行，避免线程开销
    if (total_work < MVZK_OMP_SIZE_THRESHOLD) {
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                uint64_t a_val = LHS[i * K + k];
                if (a_val == 0) continue;
                for (int j = 0; j < N; ++j) {
                    uint64_t prod = mult_mod(a_val, RHS[k * N + j]);
                    Dst[i * N + j] = add_mod(Dst[i * N + j], prod);
                }
            }
        }
        return;
    }

    // 如果工作量足够，且 M 足够大，按行并行
    if (M >= 4 && !omp_in_parallel()) { // 至少每个线程能分到一点点行
        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                uint64_t a_val = LHS[i * K + k];
                if (a_val == 0) continue;
                for (int j = 0; j < N; ++j) {
                    uint64_t prod = mult_mod(a_val, RHS[k * N + j]);
                    Dst[i * N + j] = add_mod(Dst[i * N + j], prod);
                }
            }
        }
    } 
    // [高级优化] 如果 M=1 但 N 很大 (向量 x 矩阵)，我们需要并行化 N (内层循环)
    else {
        // M 既然很小，就不在 i 循环上并行了
        for (int i = 0; i < M; ++i) {
            // 对 N 维度进行并行
            #pragma omp parallel for if (!omp_in_parallel())
            for (int j = 0; j < N; ++j) {
                uint64_t sum = 0; // 私有累加器
                for (int k = 0; k < K; ++k) {
                    // Dst[i, j] += LHS[i, k] * RHS[k, j]
                    uint64_t term = mult_mod(LHS[i * K + k], RHS[k * N + j]);
                    sum = add_mod(sum, term);
                }
                // 原子加或者非冲突写入
                // 因为我们这里是直接覆盖写或者 +=，要注意 Dst 是否初始化
                // 原始代码逻辑是 Dst += ... 所以这里需要小心
                // 但由于 matrix_mul_acc_kernel 语义是 Accumulate，
                // 且不同 j 对应不同内存地址，这里并行 j 是安全的！
                Dst[i * N + j] = add_mod(Dst[i * N + j], sum);
            }
        }
    }
}

// 通用 Hash Combine 辅助
inline void hash_combine(size_t& seed, size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// 模板化 Hash 函数
template <typename T>
inline size_t compute_vector_hash(const std::vector<T>& vec) {
    size_t seed = 0;
    std::hash<T> hasher;
    for (const auto& item : vec) {
        hash_combine(seed, hasher(item));
    }
    return seed;
}

// 特化：针对 pair<uint64_t, uint64_t>
// 如果上面的模板不方便直接用，可以重载一个版本
inline size_t compute_vector_hash(const std::vector<std::pair<uint64_t, uint64_t>>& vec) {
    size_t seed = 0;
    std::hash<uint64_t> hasher;
    for (const auto& item : vec) {
        hash_combine(seed, hasher(item.first));
        hash_combine(seed, hasher(item.second));
    }
    return seed;
}

inline PolyTensor fast_tree_product(std::vector<PolyTensor>& items) {
    // 1. 边界检查
    if (items.empty()) {
        LOG_ERROR("Empty PolyTensor received at fast_tree_product!");
        exit(-1);
    }
    
    // 如果只有一个元素，直接返回（移交所有权）
    if (items.size() == 1) {
        return std::move(items[0]);
    }

    // 2. 接管所有权 (Take Ownership)
    // 把输入 items 移动到本地 current_layer，避免修改外部传入的 vector 导致副作用
    std::vector<PolyTensor> current_layer = std::move(items);

    // 3. 树状归约循环 (Ping-Pong 模式)
    while (current_layer.size() > 1) {
        size_t current_size = current_layer.size();
        size_t next_size = (current_size + 1) / 2;

        // 【关键 1】创建下一层的 Buffer (零拷贝开销)
        std::vector<PolyTensor> next_layer(next_size);

        // ==========================================================
        // 【关键 2】多线程并行计算 (纯数学)
        // 读: current_layer (只读) | 写: next_layer (无冲突)
        // 绝对不在并发域里做网络通信，彻底杜绝死锁！
        // ==========================================================
        #pragma omp parallel for schedule(guided) if(current_size >= MVZK_CONFIG_OMP_FAST_TREE_PRODUCT_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < current_size / 2; ++i) {
            next_layer[i] = current_layer[2 * i] * current_layer[2 * i + 1];
        }

        // 处理奇数个元素，直接晋级
        if (current_size % 2 != 0) {
            next_layer.back() = std::move(current_layer.back());
        }

        // ==========================================================
        // 【关键 3】动态阶数检查与降维打击 (主线程串行)
        // 此时已退出 OMP 并发域。安全检查并触发多态降阶！
        // ==========================================================
        for (size_t i = 0; i < next_size; ++i) {
            // 如果度数超标，立即阻断！
            if (next_layer[i].degree >= MVZK_CONFIG_MULT_PRODUCT_THRESHOLD) {
                
                //std::string check_name = "Dyn_Degree_Breaker_L" + std::to_string(current_size) + "_I" + std::to_string(i);
                
                // 触发多态接口：
                // 内部会自动处理 Prover/Verifier 的不同行为，并调用 store_relation 强制零检查
                // 返回的崭新 1 阶张量直接覆盖掉原来的高阶张量！
                next_layer[i] = next_layer[i].refresh_degree();
            }
        }

        // 【关键 4】交换 Buffer (Ping-Pong)
        // 旧的 current_layer 变成空壳自动析构，next_layer 成为新一轮的输入
        current_layer = std::move(next_layer);
    }

    // 5. 返回最终结果 (也就是 current_layer[0])
    return std::move(current_layer[0]);
}

inline uint64_t sub_mod(const uint64_t& a, const uint64_t& b) {
    return add_mod(a, PR-b);
}

inline void print_u128(__uint128_t n) {
    if (n == 0) {
        std::cout << "0";
        return;
    }

    std::string s;
    while (n > 0) {
        s += (char)('0' + (n % 10));
        n /= 10;
    }
    std::reverse(s.begin(), s.end());
    std::cout << s;
}

// ==========================================
// Conv2D 加速内核 (OpenMP Optimized)
// ==========================================

// 1. Im2Col Kernel: 将图片转为矩阵行
// ==========================================
// Conv2D 加速内核 (OpenMP Optimized)
// ==========================================

// 1. Im2Col Kernel: 支持 Padding 和 Stride
inline void im2col_kernel(
    const uint64_t* __restrict__ src, 
    uint64_t* __restrict__ dst,
    int N, int C, int H, int W,
    int kH, int kW, 
    int padding, int stride, int dilation, // 新增参数
    int H_out, int W_out
) {
    // 目标矩阵维度: (N * H_out * W_out) x (C * kH * kW)
    // 这里的 K_dim 是 Im2Col 矩阵的列数，也是 GEMM 的 K 维
    int K_dim = C * kH * kW;

    // 并行化策略: 对 Batch 和 Output Pixels 并行
    // collapse(2) 意味着把 N 和 h_out 两个循环合并并行，增加并行度
    size_t total_parallel_tasks = (size_t)N * H_out * W_out;
    #pragma omp parallel for collapse(2) if(total_parallel_tasks >= (MVZK_OMP_SIZE_THRESHOLD / 2) && !omp_in_parallel())
    for (int n = 0; n < N; ++n) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                
                // [GEMM M 维索引] 当前输出像素在 Im2Col 矩阵中的行号
                // Row layout: [Batch 0 (Pixels...)] [Batch 1 (Pixels...)] ...
                size_t dst_row = ((size_t)n * H_out + h_out) * W_out + w_out;
                
                // 遍历卷积核体积 (Channel -> KernelH -> KernelW)
                // 这个顺序必须与 Weight Transpose 的顺序一致
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            
                            // [关键逻辑] 计算在输入图(src)上的绝对坐标
                            // 公式: Input_Coord = Output_Coord * Stride + Kernel_Offset * Dilation - Padding
                            int h_in = h_out * stride + kh * dilation - padding;
                            int w_in = w_out * stride + kw * dilation - padding;
                            
                            // [GEMM K 维索引] 当前 Im2Col 矩阵的列号
                            int dst_col = (c * kH + kh) * kW + kw;
                            
                            // 目标绝对索引 (Row * Width + Col)
                            size_t dst_idx = dst_row * K_dim + dst_col;

                            // 边界检查 (Padding Logic)
                            // 如果坐标在 [0, H) 和 [0, W) 范围内，拷贝数据；否则填 0
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                // 源数据索引 (NCHW 格式)
                                size_t src_idx = ((size_t)n * C + c) * H * W + h_in * W + w_in;
                                dst[dst_idx] = src[src_idx];
                            } else {
                                // Padding 区域填 0
                                dst[dst_idx] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

// 2. Weight Transpose Kernel: (C_out, C_in, kH, kW) -> (C_in * kH * kW, C_out)
// 用于适配 GEMM: Im2Col_Mat * Weight_Mat
inline void transpose_weight_kernel(
    const uint64_t* __restrict__ src, 
    uint64_t* __restrict__ dst,
    int C_out, int C_in, int kH, int kW
) {
    int K = C_in * kH * kW; // GEMM 的 K 维度 (Weight 的行数)
    
    // Src Layout: C_out (最外层) -> K (内层打平)
    // Dst Layout: K (最外层) -> C_out (内层)
    size_t total_elements = (size_t)C_out * K;
    
    #pragma omp parallel for collapse(2) if(total_elements >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (int co = 0; co < C_out; ++co) {
        for (int k = 0; k < K; ++k) {
            dst[k * C_out + co] = src[co * K + k];
        }
    }
}

// 3. Permute & Add Bias Kernel: (N, H, W, C) -> (N, C, H, W) + Bias
// MatMul 出来的结果通常是 Row-Major 的像素排列，即 NHWC 格式，需要转回 NCHW
// 增加了 bias_scale 参数，用于支持 Verifier 的 Delta 移位逻辑
inline void permute_and_add_bias_kernel(
    const uint64_t* __restrict__ src, 
    uint64_t* __restrict__ dst, 
    const uint64_t* __restrict__ bias, // 可以是 nullptr
    uint64_t bias_scale,               // 【新增】缩放因子
    int N, int H_out, int W_out, int C_out
) {
    bool use_scale = (bias_scale != 1);

    size_t total_elements = (size_t)N * H_out * W_out * C_out;

    #pragma omp parallel for collapse(3) if(total_elements >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_out; ++c) {
            for (int h = 0; h < H_out; ++h) {
                
                // 预计算 bias_val (Hoisting)
                uint64_t final_bias_val = 0;
                if (bias) {
                    final_bias_val = bias[c];
                    if (use_scale) {
                        final_bias_val = mult_mod(final_bias_val, bias_scale);
                    }
                }

                for (int w = 0; w < W_out; ++w) {
                    size_t src_idx = (((size_t)n * H_out + h) * W_out + w) * C_out + c;
                    size_t dst_idx = (((size_t)n * C_out + c) * H_out + h) * W_out + w;

                    uint64_t val = src[src_idx];
                    
                    if (bias) {
                        val = add_mod(val, final_bias_val);
                    }
                    
                    dst[dst_idx] = val;
                }
            }
        }
    }
}

// ==========================================
// Linear Layer 加速内核
// ==========================================

// 4. Matrix Transpose Kernel: (Rows, Cols) -> (Cols, Rows)
// 用于 Linear 层: W (Out, In) -> W^T (In, Out)
inline void transpose_matrix_kernel(
    const uint64_t* __restrict__ src, 
    uint64_t* __restrict__ dst,
    int Rows, int Cols
) {

    size_t total_elements = (size_t)Rows * Cols;
    
    // 【修正】
    #pragma omp parallel for collapse(2) if(total_elements >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) {
            // Src: [r, c]
            // Dst: [c, r]
            dst[c * Rows + r] = src[r * Cols + c];
        }
    }
}

// 5. Add Bias Row Broadcast Kernel: Dst[row][col] += Bias[col]
// 增加了 bias_scale 参数，支持 Verifier 高阶对齐逻辑
inline void add_bias_row_broadcast_kernel(
    uint64_t* __restrict__ data, 
    const uint64_t* __restrict__ bias, 
    uint64_t bias_scale, 
    int Rows, int Cols
) {
    bool use_scale = (bias_scale != 1);

    size_t total_elements = (size_t)Rows * Cols;
    
    // 【修正】
    #pragma omp parallel for if(total_elements >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) {
            uint64_t b_val = bias[c];
            
            if (use_scale) {
                b_val = mult_mod(b_val, bias_scale);
            }
            
            // data[r, c] += bias[c]
            int idx = r * Cols + c;
            data[idx] = add_mod(data[idx], b_val);
        }
    }
}

// ==========================================
// Conv1D 加速内核
// ==========================================

// 6. Im2Col 1D Kernel
// Input: (N, C, L_in) -> Output Matrix: (N * L_out, C * K)
inline void im2col_1d_kernel(
    const uint64_t* __restrict__ src, 
    uint64_t* __restrict__ dst,
    int N, int C, int L_in,
    int K_size, 
    int padding, int stride, int dilation,
    int L_out
) {
    int K_dim = C * K_size; // GEMM K 维大小

    // 并行策略: Batch 和 Length 并行
    #pragma omp parallel for collapse(2) if (!omp_in_parallel())
    for (int n = 0; n < N; ++n) {
        for (int l_out = 0; l_out < L_out; ++l_out) {
            
            // Dst Row Index: [Batch, L_out]
            size_t dst_row = (size_t)n * L_out + l_out;

            for (int c = 0; c < C; ++c) {
                for (int k = 0; k < K_size; ++k) {
                    
                    // 1D 坐标变换
                    int l_in = l_out * stride + k * dilation - padding;
                    
                    // Dst Col Index
                    int dst_col = c * K_size + k;
                    size_t dst_idx = dst_row * K_dim + dst_col;

                    if (l_in >= 0 && l_in < L_in) {
                        size_t src_idx = ((size_t)n * C + c) * L_in + l_in;
                        dst[dst_idx] = src[src_idx];
                    } else {
                        dst[dst_idx] = 0; // Padding
                    }
                }
            }
        }
    }
}

// 7. Permute & Add Bias 1D Kernel
// Src: (N * L_out, C_out) -> logical (N, L_out, C_out)
// Dst: (N, C_out, L_out)
// Bias: (C_out)
inline void permute_and_add_bias_1d_kernel(
    const uint64_t* __restrict__ src, 
    uint64_t* __restrict__ dst, 
    const uint64_t* __restrict__ bias, 
    uint64_t bias_scale,
    int N, int L_out, int C_out
) {
    // 预检查 bias 是否存在，避免循环内判断
    // 注意：这里的 bias 指针已经在 helper 层判空过了，这里主要为了 scale
    bool use_scale = (bias_scale != 1);

    #pragma omp parallel for collapse(2) if (!omp_in_parallel())
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_out; ++c) {
            
            // Hoisting Bias Calculation
            uint64_t final_bias = 0;
            if (bias) {
                final_bias = bias[c];
                if (use_scale) final_bias = mult_mod(final_bias, bias_scale);
            }

            // 连续写入优化: L_out 是最内层循环
            for (int l = 0; l < L_out; ++l) {
                // Src Index (Row-Major from MatMul): (n, l, c)
                size_t src_idx = ((size_t)n * L_out + l) * C_out + c;
                
                // Dst Index (Target NCL): (n, c, l)
                size_t dst_idx = ((size_t)n * C_out + c) * L_out + l;

                uint64_t val = src[src_idx];
                if (bias) {
                    val = add_mod(val, final_bias);
                }
                dst[dst_idx] = val;
            }
        }
    }
}

inline uint64_t real2fp(double x) {
    // 预计算 Scale，使用 double 避免重复类型转换
    const double scale_factor = static_cast<double>(1ULL << MVZK_CONFIG_SCALE);
    
    // 四舍五入，转为有符号 64 位整数
    int64_t val = std::round(x * scale_factor);

    // 快速模运算处理负数
    // C++ 中负数取模结果可能为负 (例如 -5 % 3 = -2)，需要修正为正
    int64_t rem = val % (int64_t)PR;
    if (rem < 0) {
        rem += (int64_t)PR;
    }
    return (uint64_t)rem;
}

// ==========================================
// 2. 向量/矩阵版本 (高性能重载)
// ==========================================
// 支持任意维度的扁平化数据输入
inline std::vector<uint64_t> real2fp(const std::vector<float>& input) {
    size_t n = input.size();
    
    // 1. 内存预分配 (关键性能点)
    // 直接构造指定大小的 vector，避免 realloc
    std::vector<uint64_t> output(n);

    // 2. 预计算常量
    const double scale_factor = static_cast<double>(1ULL << MVZK_CONFIG_SCALE);
    // 强转 PR 为有符号，避免循环内重复 cast
    const int64_t signed_PR = (int64_t)PR;

    // 获取原始指针，帮助编译器进行自动向量化 (Auto-Vectorization)
    const float* __restrict__ in_ptr = input.data();
    uint64_t* __restrict__ out_ptr = output.data();

    // 3. OpenMP 并行化
    // static 调度适合这种每个元素计算量完全相同的场景
    #pragma omp parallel for schedule(static) if(n >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (size_t i = 0; i < n; ++i) {
        // 核心转换逻辑 (手动内联以确保性能)
        
        // Step A: 乘法与取整
        // 显式转为 double 乘法，提高精度
        double raw_val = (double)in_ptr[i] * scale_factor;
        int64_t val = std::round(raw_val);

        // Step B: 模运算 (处理负数)
        // 使用 signed_PR 进行取模
        int64_t rem = val % signed_PR;
        
        // 消除分支预测失败的风险 (Branchless 优化尝试)
        // 这种写法比 if (rem < 0) rem += PR 更利于流水线，
        // 但编译器通常能很好地优化 if，这里保留 if 版本可读性更好，
        // 现代 CPU 的分支预测对这种随机性不强的判断处理得很快。
        if (rem < 0) {
            rem += signed_PR;
        }

        out_ptr[i] = (uint64_t)rem;
    }

    return output;
}

// ==========================================
// 3. 泛型递归包装器 (支持任意维度 vector)
// ==========================================

// 辅助：计算总元素数
template <typename T> size_t get_total_elements(const T& val) { return 1; }
template <typename T> size_t get_total_elements(const std::vector<T>& vec) {
    if (vec.empty()) return 0;
    return vec.size() * get_total_elements(vec[0]);
}

// 辅助：递归展平
template <typename T> void flatten_recursive(const T& val, std::vector<float>& buffer) {
    buffer.push_back(static_cast<float>(val));
}
template <typename T> void flatten_recursive(const std::vector<T>& vec, std::vector<float>& buffer) {
    for (const auto& item : vec) flatten_recursive(item, buffer);
}

// 主入口：real2fp_ndim
// 无论输入是 vector<float> 还是 vector<vector<vector<float>>>, 都能处理
template <typename T>
inline std::vector<uint64_t> real2fp_ndim(const std::vector<T>& input) {
    size_t total = get_total_elements(input);
    std::vector<float> flat_floats;
    flat_floats.reserve(total);
    flatten_recursive(input, flat_floats);
    
    // 调用你刚才写对的那个高性能 1D 版本
    return real2fp(flat_floats); 
}

// ==========================================
// 高性能多项式乘法引擎 (Karatsuba + OpenMP)
// ==========================================

// [基础乘法器]：双层 for 循环，作为递归终点
/*
inline void base_mul_core(
    const uint64_t* __restrict__ A, int dA, 
    const uint64_t* __restrict__ B, int dB, 
    uint64_t* __restrict__ Res, size_t size) 
{
    int dRes = dA + dB;
    size_t total_ops = (size_t)(dRes + 1) * size;
    
    // 初始化 Res 数组为 0
    if (total_ops >= MVZK_OMP_SIZE_THRESHOLD) {
        #pragma omp parallel for
        for (size_t i = 0; i < total_ops; ++i) Res[i] = 0;
    } else {
        std::memset(Res, 0, total_ops * sizeof(uint64_t));
    }

    if (size >= MVZK_OMP_SIZE_THRESHOLD) {
        for (int k = 0; k <= dRes; ++k) {
            uint64_t* res_ptr = Res + k * size;
            int start_i = std::max(0, k - dB);
            int end_i = std::min(dA, k);

            #pragma omp parallel for
            for (size_t idx = 0; idx < size; ++idx) {
                uint64_t sum = 0;
                for (int i = start_i; i <= end_i; ++i) {
                    int j = k - i;
                    uint64_t term = mult_mod(A[i * size + idx], B[j * size + idx]);
                    sum = add_mod(sum, term);
                }
                res_ptr[idx] = sum;
            }
        }
    } else {
        //#pragma omp parallel for if(dRes >= MVZK_OMP_DEGREE_THRESHOLD)
        for (int k = 0; k <= dRes; ++k) {
            uint64_t* res_ptr = Res + k * size;
            int start_i = std::max(0, k - dB);
            int end_i = std::min(dA, k);

            for (size_t idx = 0; idx < size; ++idx) {
                uint64_t sum = 0;
                for (int i = start_i; i <= end_i; ++i) {
                    int j = k - i;
                    uint64_t term = mult_mod(A[i * size + idx], B[j * size + idx]);
                    sum = add_mod(sum, term);
                }
                res_ptr[idx] = sum;
            }
        }
    }
}*/

inline void base_mul_core(
    const uint64_t* __restrict__ A, int dA, 
    const uint64_t* __restrict__ B, int dB, 
    uint64_t* __restrict__ Res, size_t size) 
{
    int dRes = dA + dB;
    size_t total_ops = (size_t)(dRes + 1) * size;
    
    // 1. 初始化 Res 数组为 0 (保持原逻辑)
    if (total_ops >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel()) {
        #pragma omp parallel for
        for (size_t i = 0; i < total_ops; ++i) Res[i] = 0;
    } else {
        std::memset(Res, 0, total_ops * sizeof(uint64_t));
    }

    // 2. 核心计算优化：将 OpenMP 并行域外提
    if (size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel()) {
        // 【关键修改】开启一次并行域，所有线程在此待命，不再反复创建销毁
        #pragma omp parallel 
        {
            // 所有线程共享外层 k 的循环逻辑
            for (int k = 0; k <= dRes; ++k) {
                uint64_t* res_ptr = Res + k * size;
                int start_i = std::max(0, k - dB);
                int end_i = std::min(dA, k);

                // 【关键修改】使用 #pragma omp for 分配任务
                // 注意：这里保留了隐式 Barrier（即没有加 nowait），
                // 确保所有线程算完第 k 阶后，再一起进入第 k+1 阶，保证依赖安全（虽然此处无依赖，但为了逻辑清晰）
                #pragma omp for 
                for (size_t idx = 0; idx < size; ++idx) {
                    uint64_t sum = 0;
                    for (int i = start_i; i <= end_i; ++i) {
                        int j = k - i;
                        // 注意：此处指针偏移量计算由编译器优化，直接访问
                        uint64_t term = mult_mod(A[i * size + idx], B[j * size + idx]);
                        sum = add_mod(sum, term);
                    }
                    res_ptr[idx] = sum;
                }
            }
        }
    } else {
        // 串行版本 (处理小规模数据)
        for (int k = 0; k <= dRes; ++k) {
            uint64_t* res_ptr = Res + k * size;
            int start_i = std::max(0, k - dB);
            int end_i = std::min(dA, k);

            for (size_t idx = 0; idx < size; ++idx) {
                uint64_t sum = 0;
                for (int i = start_i; i <= end_i; ++i) {
                    int j = k - i;
                    uint64_t term = mult_mod(A[i * size + idx], B[j * size + idx]);
                    sum = add_mod(sum, term);
                }
                res_ptr[idx] = sum;
            }
        }
    }
}

// [Karatsuba 核心]：递归分治魔术
inline void karatsuba_core(
    const uint64_t* __restrict__ A, int dA, 
    const uint64_t* __restrict__ B, int dB, 
    uint64_t* __restrict__ Res, size_t size) 
{
#if MVZK_CONFIG_ENABLE_KARATSUBA_OPT
    if (dA < MVZK_CONFIG_KARATSUBA_THRESHOLD || dB < MVZK_CONFIG_KARATSUBA_THRESHOLD || 
        std::max(dA, dB) > 2 * std::min(dA, dB)) {
        base_mul_core(A, dA, B, dB, Res, size);
        return;
    }

    int m = (std::max(dA, dB) + 1) / 2;

    int dA_lo = std::min(dA, m - 1);
    int dA_hi = (dA >= m) ? (dA - m) : -1;
    
    int dB_lo = std::min(dB, m - 1);
    int dB_hi = (dB >= m) ? (dB - m) : -1;

    int dAsum = std::max(dA_lo, dA_hi);
    int dBsum = std::max(dB_lo, dB_hi);

    std::vector<uint64_t> Asum((dAsum + 1) * size, 0);
    std::vector<uint64_t> Bsum((dBsum + 1) * size, 0);

    for (int i = 0; i <= dAsum; ++i) {
        const uint64_t* p_lo = (i <= dA_lo) ? (A + i * size) : nullptr;
        const uint64_t* p_hi = (i <= dA_hi) ? (A + (m + i) * size) : nullptr;
        uint64_t* p_sum = Asum.data() + i * size;
        
        //#pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD)
        for (size_t idx = 0; idx < size; ++idx) {
            uint64_t val_lo = p_lo ? p_lo[idx] : 0;
            uint64_t val_hi = p_hi ? p_hi[idx] : 0;
            p_sum[idx] = add_mod(val_lo, val_hi);
        }
    }
    for (int i = 0; i <= dBsum; ++i) {
        const uint64_t* p_lo = (i <= dB_lo) ? (B + i * size) : nullptr;
        const uint64_t* p_hi = (i <= dB_hi) ? (B + (m + i) * size) : nullptr;
        uint64_t* p_sum = Bsum.data() + i * size;
        
        //#pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD)
        for (size_t idx = 0; idx < size; ++idx) {
            uint64_t val_lo = p_lo ? p_lo[idx] : 0;
            uint64_t val_hi = p_hi ? p_hi[idx] : 0;
            p_sum[idx] = add_mod(val_lo, val_hi);
        }
    }

    int dP1 = dA_lo + dB_lo;
    int dP2 = (dA_hi >= 0 && dB_hi >= 0) ? (dA_hi + dB_hi) : -1;
    int dP3 = dAsum + dBsum;

    std::vector<uint64_t> P1((dP1 + 1) * size, 0);
    std::vector<uint64_t> P3((dP3 + 1) * size, 0);
    
    karatsuba_core(A, dA_lo, B, dB_lo, P1.data(), size);
    karatsuba_core(Asum.data(), dAsum, Bsum.data(), dBsum, P3.data(), size);
    
    std::vector<uint64_t> P2;
    if (dP2 >= 0) {
        P2.resize((dP2 + 1) * size, 0);
        karatsuba_core(A + m * size, dA_hi, B + m * size, dB_hi, P2.data(), size);
    }

    int dRes = dA + dB;
    //#pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD)
    for (size_t i = 0; i < (size_t)(dRes + 1) * size; ++i) Res[i] = 0;

    for (int i = 0; i <= dP1; ++i) {
        uint64_t* dst = Res + i * size;
        uint64_t* src = P1.data() + i * size;
        //#pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD)
        for (size_t idx = 0; idx < size; ++idx) dst[idx] = add_mod(dst[idx], src[idx]);
    }
    if (dP2 >= 0) {
        for (int i = 0; i <= dP2; ++i) {
            uint64_t* dst = Res + (2 * m + i) * size;
            uint64_t* src = P2.data() + i * size;
            //#pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD)
            for (size_t idx = 0; idx < size; ++idx) dst[idx] = add_mod(dst[idx], src[idx]);
        }
    }
    for (int i = 0; i <= dP3; ++i) {
        uint64_t* dst = Res + (m + i) * size;
        uint64_t* src3 = P3.data() + i * size;
        const uint64_t* src1 = (i <= dP1) ? (P1.data() + i * size) : nullptr;
        const uint64_t* src2 = (dP2 >= 0 && i <= dP2) ? (P2.data() + i * size) : nullptr;

        //#pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD)
        for (size_t idx = 0; idx < size; ++idx) {
            uint64_t val = src3[idx];
            if (src1) val = sub_mod(val, src1[idx]); 
            if (src2) val = sub_mod(val, src2[idx]);
            dst[idx] = add_mod(dst[idx], val);
        }
    }
#else
    base_mul_core(A, dA, B, dB, Res, size);
#endif
}