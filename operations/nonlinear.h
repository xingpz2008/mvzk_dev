#pragma once

#include "../data_type/PolyDelta.h"
#include "../data_type/PolyTensor.h"
#include "../utility.h"
#include "../config.h"

PolyTensor ReLU(PolyTensor& x, uint64_t bitlen = MVZK_CONFIG_DEFAULT_BITLEN, 
    uint64_t digdec_k = MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K, 
    bool do_truncation = MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION,
    uint64_t scale = MVZK_CONFIG_SCALE);

PolyTensor MaxPool2D(PolyTensor& x, 
    int kernel_size, 
    int stride, 
    int padding, 
    uint64_t bitlen = MVZK_CONFIG_DEFAULT_BITLEN, 
    uint64_t digdec_k = MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K,
    uint64_t scale = MVZK_CONFIG_SCALE);

PolyTensor IntegratedNL(PolyTensor& x, 
    int kernel_size, 
    int stride, 
    int padding, 
    uint64_t bitlen = MVZK_CONFIG_DEFAULT_BITLEN, 
    uint64_t digdec_k = MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K,
    bool do_truncation = MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION,
    uint64_t scale = MVZK_CONFIG_SCALE);

// ==========================================
// Plaintext Helper Functions (for Debug/Check)
// ==========================================

/**
 * @brief [Plaintext Helper] Digit Decomposition
 * 将 l 比特的数据分解为 k 段。
 * 策略：如果 l 不能被 k 整除，视为在高位补 0 直到能被整除。
 * * * @param src     输入向量 (N)
 * @param k       分解段数
 * @param l       有效数据总比特数 (l_bits)
 * * 计算：每段比特数 bits = ceil(l / k)
 * 例如：l=62, k=4 -> bits=16. (实际处理 64 bits, 高位自动为 0)
 * * @return std::vector<uint64_t> 扁平化的输出，大小为 N * k
 * Layout: [x0_seg0, ..., x0_seg{k-1}, x1_seg0, ...]
 */
inline std::vector<uint64_t> helper_plaintext_digit_decomposition(
    const std::vector<uint64_t>& src, 
    int k,
    int l = MVZK_CONFIG_DEFAULT_BITLEN 
) {
    if (k <= 0) return {};
    size_t n = src.size();
    
    // 1. 自动计算每段需要的 bits (向上取整)
    // 公式：ceil(l / k) => (l + k - 1) / k
    int bits_per_seg = (l + k - 1) / k;
    
    // 2. 预计算 Mask
    // 比如 bits=16, mask = 0xFFFF
    uint64_t mask = (bits_per_seg >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << bits_per_seg) - 1);

    std::vector<uint64_t> output(n * k);
    const uint64_t* __restrict__ in_ptr = src.data();
    uint64_t* __restrict__ out_ptr = output.data();

    // 3. OpenMP 并行化
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        uint64_t val = in_ptr[i];
        
        // 分解为 k 段
        for (int j = 0; j < k; ++j) {
            int shift = j * bits_per_seg;
            uint64_t segment = 0;
            
            // 安全移位检查
            // 只要 shift < 64，我们就可以移位。
            // 因为 val 是 uint64_t，超出 l 的高位本身就是 0，
            // 所以这自动实现了 "高位补 0" 的逻辑。
            if (shift < 64) {
                segment = (val >> shift) & mask;
            }
            // 如果 shift >= 64 (比如 l=100, k=2, bits=50, 第二段 shift=50 ok, 第三段若有则为0)
            // 这里 shift >= 64 时 segment 保持为 0，符合补零逻辑。
            
            out_ptr[i * k + j] = segment;
        }
    }

    return output;
}

/**
 * @brief [Plaintext Helper] Sign Determination (Boolean Mode)
 * 严格对应 Figure 9 的 Step 1。
 * 逻辑：
 * - 负数 (Val > Zero): b_Q = 0 (Integer)
 * - 正数 (Val <= Zero): b_Q = 1 (Integer)
 * 返回值是 Unscaled Integer，用于 Step 6/7 的布尔选择。
 */
inline std::vector<uint64_t> helper_plaintext_sign(
    const std::vector<uint64_t>& src, 
    uint64_t zero_point = 0 
) {
    size_t n = src.size();
    std::vector<uint64_t> output(n);

    // 默认零点 PR/2
    if (zero_point == 0) zero_point = PR >> 1; 

    const uint64_t* __restrict__ in_ptr = src.data();
    uint64_t* __restrict__ out_ptr = output.data();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        // 图 9 协议暗示：正数 b=1, 负数 b=0
        if (in_ptr[i] > zero_point) {
            out_ptr[i] = 0; // Negative
        } else {
            out_ptr[i] = 1; // Positive
        }
    }
    return output;
}

inline bool helper_plaintext_fp_greater(uint64_t a, uint64_t b, uint64_t zero_point = 0) {
    if (zero_point == 0) zero_point = PR >> 1;

    bool a_is_pos = (a <= zero_point);
    bool b_is_pos = (b <= zero_point);

    // Case 1 & 2: 正数 vs 负数
    if (a_is_pos && !b_is_pos) return true;  // a 正 b 负 -> a > b
    if (!a_is_pos && b_is_pos) return false; // a 负 b 正 -> a < b

    // Case 3 & 4: 符号相同（同为正或同为负）
    // 负数在 uint64 下比如 P-3 > P-5，等价于 -3 > -5，直接比较依然成立！
    return a > b;
}