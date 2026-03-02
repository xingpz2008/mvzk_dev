#pragma once

#include <vector>
// 只引入需要的数据结构
#include "../data_type/PolyTensor.h"
#include "../config.h"

// ==========================================
// 1. Bottleneck 模块权重定义
// ==========================================
struct BottleneckWeights {
    static const int expansion = 4;

    PolyTensor conv1_w, conv1_b; 
    PolyTensor conv2_w, conv2_b; 
    PolyTensor conv3_w, conv3_b; 
    
    bool has_downsample;         
    PolyTensor downsample_w, downsample_b; 
};

// ==========================================
// 2. ResNet-50 全局权重定义
// ==========================================
struct ResNet50Weights {
    PolyTensor conv1_w, conv1_b; 
    
    std::vector<BottleneckWeights> layer1; 
    std::vector<BottleneckWeights> layer2; 
    std::vector<BottleneckWeights> layer3; 
    std::vector<BottleneckWeights> layer4; 

    PolyTensor fc_w, fc_b; 
};

// ==========================================
// 3. 函数声明
// ==========================================
PolyTensor Bottleneck_Forward(
    PolyTensor& x, 
    BottleneckWeights& block, 
    int stride, 
    uint64_t bitlen, 
    uint64_t digdec_k, 
    bool do_truncation
);

PolyTensor ResNet50_Forward(
    PolyTensor& image, 
    ResNet50Weights& net,
    uint64_t bitlen, 
    uint64_t digdec_k, 
    bool do_truncation,
    bool using_integrated_nl = MVZK_CONFIG_MODEL_USING_INTEGRATED_NON_LINEAR
);