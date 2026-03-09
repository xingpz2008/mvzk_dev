#ifndef MVZK_CONFIG_H__
#define MVZK_CONFIG_H__

// =================================================================
// 1. Core System & Execution Configs (核心系统与执行配置)
// =================================================================
#define MVZK_CONFIG_THREADS_NUM 1
#define MVZK_CONFIG_DEFAULT_LEVEL_INFO LEVEL_ERROR
#define MVZK_CONFIG_INSTANT_ABORT false

// =================================================================
// 2. Base Cryptography & Data Representation (基础密码学与数据表示)
// =================================================================
static constexpr char fix_key[] = "\x61\x7e\x8d\xa2\xa0\x51\x1e\x96\x5e\x41\xc2\x9b\x15\x3f\xc7\x7a";
#define MVZK_CONFIG_SCALE 12
#define MVZK_CONFIG_DEFAULT_BITLEN 32

// =================================================================
// 3. Machine Learning Model & Non-Linear Ops (机器学习模型与非线性算子)
// =================================================================
#define MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K 8
#define MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION true
#define MVZK_CONFIG_MODEL_USING_INTEGRATED_NON_LINEAR true

// =================================================================
// 4. Verification & Buffer Limits (零知识证明约束与缓存阈值)
// =================================================================
//#define MVZK_CONFIG_AUTO_CHECK_LIMIT 1<<20
#define MVZK_MULT_CHECK_CNT 1024*1024
#define MVZK_CONFIG_TENSOR_CHECK_CNT 2000
#define MVZK_CONFIG_MATMUL_TENSOR_CHECK_CNT 2000
#define MVZK_CONFIG_LUT_PUBLIC_TABLE_SIZE 1 << 20

//#define MVZK_CONFIG_LUT_REQUEST_BUFFER_THRESHOLD 260000
//#define MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD 260000
#define MVZK_CONFIG_LUT_REQUEST_BUFFER_THRESHOLD 32768
#define MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD 32768

// =================================================================
// 5. Math Optimizations (多项式底层数学优化: Karatsuba & FFT)
// =================================================================
#define MVZK_FFT_WARNING_LIMIT 256
//#define MVZK_CONFIG_FFT_WARNING_ON false
#define MVZK_CONFIG_ENABLE_KARATSUBA_OPT 1       // 1 开启 Karatsuba 优化，0 关闭
#define MVZK_CONFIG_KARATSUBA_THRESHOLD 64       // 触发 Karatsuba 的最小阶数 (建议 64)

// =================================================================
// 6. 动态 OpenMP 调度阈值 (应对 Thread Thrashing 线程自杀问题)
// =================================================================
// 1. 空间维度：当 Tensor 的 total_elements >= 此值时，才开启内部 size 级并行
#define MVZK_OMP_SIZE_THRESHOLD 8192

// 2. 阶数维度：针对 size 极小但阶数爆炸的连乘，当 degree >= 此值时，开启外部 k 级并行
#define MVZK_OMP_DEGREE_THRESHOLD 64

#define MVZK_CONFIG_OMP_FAST_TREE_PRODUCT_SIZE_THRESHOLD 4

#endif