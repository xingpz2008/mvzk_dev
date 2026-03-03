/*
 * Usage: 
 * ./resnet50_test <party> <port> <N> <H> <W>
 *
 * Example (CIFAR-10, Recommended 32x32 for fast verification):
 * [Prover]   ./resnet50_test 1 12345 1 32 32
 * [Verifier] ./resnet50_test 2 12345 1 32 32
 * 
 * Example (ImageNet):
 * [Prover]   ./resnet50_test 1 12345 1 224 224
 * [Verifier] ./resnet50_test 2 12345 1 224 224
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "utility.h"       
#include "emp-tool/emp-tool.h"

// 引入你写好的模型结构定义与前向传播逻辑
#include "local_models/resnet50.h"

using namespace std;
using namespace emp;
using namespace std::chrono;

MVZKExec* MVZKExec::mvzk_exec = nullptr;
const int PARTY_PROVER = ALICE;
const int PARTY_VERIFIER = BOB;

// ==========================================
// 辅助工具：时间测量
// ==========================================
template <typename Func>
long long measure_time(Func&& func, string step_name) {
    auto start = high_resolution_clock::now();
    func();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "[TIME] " << left << setw(45) << step_name << ": " << duration.count() << " ms" << endl;
    return duration.count();
}

// ==========================================
// 辅助工具：张量随机生成器 (Dummy Data)
// ==========================================
PolyTensor create_dummy_tensor(MVZKExec* exec, int party, const vector<int>& shape) {
    size_t total_size = 1;
    for (int s : shape) total_size *= s;

    vector<uint64_t> data(total_size, 0);
    if (party == PARTY_PROVER) {
        static mt19937 gen(12345);
        uniform_int_distribution<int64_t> dis(-5, 5); // 小范围随机数，防止未经 LayerNorm/BN 的 50 层网络直接溢出
        for (size_t i = 0; i < total_size; ++i) {
            float f_val = static_cast<float>(dis(gen)) / 100.0f;
            data[i] = real2fp(f_val);
        }
    }
    return exec->input(shape, data);
}

// ==========================================
// PyTorch 范式：模型实例化工厂 (Model Builder)
// ==========================================

// 对标 PyTorch 源码里的 Bottleneck.__init__
BottleneckWeights _make_bottleneck(MVZKExec* exec, int party, int in_channels, int base_channels, bool downsample) {
    BottleneckWeights bw;
    int out_channels = base_channels * 4;

    bw.conv1_w = create_dummy_tensor(exec, party, {base_channels, in_channels, 1, 1});
    bw.conv1_b = create_dummy_tensor(exec, party, {base_channels});

    bw.conv2_w = create_dummy_tensor(exec, party, {base_channels, base_channels, 3, 3});
    bw.conv2_b = create_dummy_tensor(exec, party, {base_channels});

    bw.conv3_w = create_dummy_tensor(exec, party, {out_channels, base_channels, 1, 1});
    bw.conv3_b = create_dummy_tensor(exec, party, {out_channels});

    bw.has_downsample = downsample;
    if (downsample) {
        bw.downsample_w = create_dummy_tensor(exec, party, {out_channels, in_channels, 1, 1});
        bw.downsample_b = create_dummy_tensor(exec, party, {out_channels});
    }
    return bw;
}

// 对标 PyTorch 源码里的 ResNet._make_layer
vector<BottleneckWeights> _make_layer(MVZKExec* exec, int party, int in_channels, int base_channels, int blocks) {
    vector<BottleneckWeights> layer;
    // Layer 的第一个 block 负责通道扩增和降采样
    layer.push_back(_make_bottleneck(exec, party, in_channels, base_channels, true));
    
    // 后续的 block 通道数已经对齐，不需要 downsample
    int next_in_channels = base_channels * 4;
    for (int i = 1; i < blocks; ++i) {
        layer.push_back(_make_bottleneck(exec, party, next_in_channels, base_channels, false));
    }
    return layer;
}

// 极其清爽的顶层 API：获取完整的 ResNet50 模型实例
ResNet50Weights resnet50(MVZKExec* exec, int party) {
    ResNet50Weights model;
    
    // Stem
    model.conv1_w = create_dummy_tensor(exec, party, {64, 3, 7, 7});
    model.conv1_b = create_dummy_tensor(exec, party, {64});

    // 4 个 Stage 的堆叠 (自动推导维度并申请内存)
    model.layer1 = _make_layer(exec, party, 64,   64,  3); // 输出通道: 256
    model.layer2 = _make_layer(exec, party, 256,  128, 4); // 输出通道: 512
    model.layer3 = _make_layer(exec, party, 512,  256, 6); // 输出通道: 1024
    model.layer4 = _make_layer(exec, party, 1024, 512, 3); // 输出通道: 2048

    // FC Layer
    model.fc_w = create_dummy_tensor(exec, party, {1000, 2048});
    model.fc_b = create_dummy_tensor(exec, party, {1000});

    return model;
}

// ==========================================
// 主函数：极其优雅的用户侧体验
// ==========================================
int main(int argc, char** argv) {
    SetLogLevel(LEVEL_WARN);
    if (argc < 6) {
        cout << "Usage: " << argv[0] << " <party> <port> <N> <H> <W>" << endl;
        return 1;
    }

    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    int N = atoi(argv[3]), H = atoi(argv[4]), W = atoi(argv[5]);

    uint64_t bitlen = MVZK_CONFIG_DEFAULT_BITLEN;
    uint64_t digdec_k = MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K;
    bool do_truncation = MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION;

    cout << "=================================================" << endl;
    cout << "ResNet-50 Full ZK Graph Test" << endl;
    cout << "Role: " << (party == PARTY_PROVER ? "PROVER" : "VERIFIER") << endl;
    cout << "=================================================" << endl;

    /*
    NetIO *io = new NetIO(party == PARTY_PROVER ? nullptr : "127.0.0.1", port);
    NetIO *io_arr[1] = {io};*/

    int num_threads = MVZK_CONFIG_THREADS_NUM;
    NetIO **io_arr = new NetIO*[num_threads];
    for (int i = 0; i < num_threads; ++i) {
        // 每个线程分配一个专属的连续端口 (port, port+1, port+2...)
        io_arr[i] = new NetIO(party == PARTY_PROVER ? nullptr : "127.0.0.1", port + i);
    }

    MVZKExec *exec = nullptr;
    if (party == PARTY_PROVER) exec = new MVZKExecProver<NetIO>(io_arr);
    else exec = new MVZKExecVerifier<NetIO>(io_arr);

    try {
        auto start_zk = high_resolution_clock::now();

        // 1. 生成数据
        cout << "[TEST] Loading Dummy Image..." << endl;
        PolyTensor image = create_dummy_tensor(exec, party, {N, 3, H, W});

        // 2. 实例化模型 (对标 PyTorch: model = resnet50())
        cout << "[TEST] Instantiating ResNet-50 Model Weights..." << endl;
        ResNet50Weights model = resnet50(exec, party);

        // 3. 前向传播 (对标 PyTorch: output = model(image))
        PolyTensor output;
        measure_time([&](){
            output = ResNet50_Forward(image, model, bitlen, digdec_k, do_truncation);
        }, "ResNet50 Forward Pass Execution");

        cout << "[TEST] Output Shape: (" 
             << output.shape[0] << ", " << output.shape[1] << ", " 
             << output.shape[2] << ", " << output.shape[3] << ")" << endl;

        cout << "[TEST] Securing final logits via Self-Relation..." << endl;
        PolyTensor::store_self_relation(output, "ResNet50_Final_Logits_Check");

        // 4. 密码学协议约束校验
        cout << "\n[TEST] Triggering ZK Constraints Verification..." << endl;
        measure_time([&](){
            exec->check_all();
        }, "Check All MACs and Circuits");

        auto stop_zk = high_resolution_clock::now();
        auto total_ms = duration_cast<milliseconds>(stop_zk - start_zk).count();
        long long h  = total_ms / 3600000;
        long long m  = (total_ms % 3600000) / 60000;
        long long s  = (total_ms % 60000) / 1000;
        long long ms = total_ms % 1000;

        cout << "\n=================================================" << endl;
        cout << "[TIME] TOTAL PIPELINE: " << total_ms << " ms" << endl;
        
        // 额外输出直观的格式
        cout << "[TIME] FORMATTED TIME: ";
        if (h > 0) cout << h << "h ";
        if (m > 0 || h > 0) cout << m << "m ";
        if (s > 0 || m > 0 || h > 0) cout << s << "s ";
        cout << ms << "ms" << endl;
        cout << "=================================================\n" << endl;
        cout << "\033[32m[SUCCESS] ResNet-50 Test Passed Beautifully!\033[0m" << endl;

    } catch (const std::exception& e) {
        cerr << "\n\033[31m[FATAL ERROR] " << e.what() << "\033[0m" << endl;
    }

    if(exec) delete exec;
    delete io_arr;
    return 0;
}