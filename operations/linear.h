#ifndef MVZK_LINEAR_H__
#define MVZK_LINEAR_H__

#include "../data_type/PolyTensor.h"

/**
 * @brief 2D 卷积层接口 (类似 PyTorch Conv2d)
 * * @param input   输入张量, Shape: (N, C_in, H_in, W_in)
 * @param weight  权重张量, Shape: (C_out, C_in, kH, kW)
 * @param bias    偏置张量, Shape: (C_out). 可以为空 Tensor (total_elements=0)
 * @param stride  步长 (默认为 1)
 * @param padding 填充 (默认为 0)
 * @param dilation 膨胀 (暂不支持，预留接口，默认为 1)
 * @return PolyTensor 输出张量 (N, C_out, H_out, W_out)
 */
PolyTensor Conv2D(
    const PolyTensor& input, 
    const PolyTensor& weight, 
    const PolyTensor& bias, 
    int stride = 1, 
    int padding = 0,
    int dilation = 1 
);

// Fully Conncected Layer
PolyTensor Linear(
    const PolyTensor& input, 
    const PolyTensor& weight, 
    const PolyTensor& bias
);

/**
 * @brief 1D 卷积层接口 (PyTorch Conv1d)
 * @param input   Shape: (N, C_in, L_in)
 * @param weight  Shape: (C_out, C_in, K)
 * @param bias    Shape: (C_out)
 */
PolyTensor Conv1D(
    const PolyTensor& input, 
    const PolyTensor& weight, 
    const PolyTensor& bias, 
    int stride = 1, 
    int padding = 0,
    int dilation = 1 
);

PolyTensor AvgPool2D(PolyTensor& x, 
    int kernel_size, 
    int stride = -1, 
    int padding = 0,
    bool back_to_sum_pool = false);


// last 4 parameters are public
[[deprecated("WARNING: DO NOT USE Standalone BatchNorm2D. Please use offline Conv-BN folding instead.")]]
PolyTensor BatchNorm2D(
    PolyTensor& x, 
    const std::vector<double>& weight, 
    const std::vector<double>& bias, 
    const std::vector<double>& running_mean, 
    const std::vector<double>& running_var, 
    double eps = 1e-5
);

#endif