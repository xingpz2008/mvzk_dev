/*
 * Usage: 
 * ./vgg16_test <party> <port> <N> <H> <W>
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
#include "local_models/vgg16.h"

using namespace std;
using namespace emp;
using namespace std::chrono;

MVZKExec* MVZKExec::mvzk_exec = nullptr;
const int PARTY_PROVER = ALICE;
const int PARTY_VERIFIER = BOB;

// ==========================================
// 辅助工具：张量随机生成器 (Dummy Data)
// ==========================================
PolyTensor create_dummy_tensor(MVZKExec* exec, int party, const vector<int>& shape) {
    size_t total_size = 1;
    for (int s : shape) total_size *= s;

    vector<uint64_t> data(total_size, 0);
    if (party == PARTY_PROVER) {
        static mt19937 gen(12345);
        uniform_int_distribution<int64_t> dis(-5, 5);
        for (size_t i = 0; i < total_size; ++i) {
            float f_val = static_cast<float>(dis(gen)) / 100.0f;
            data[i] = real2fp(f_val);
        }
    }
    return exec->input(shape, data);
}

// ==========================================
// 模型实例化工厂 (Model Builder)
// ==========================================
VGG16Weights vgg16(MVZKExec* exec, int party, int input_H, int input_W) {
    VGG16Weights model;
    
    model.conv1_1_w = create_dummy_tensor(exec, party, {64, 3, 3, 3});
    model.conv1_1_b = create_dummy_tensor(exec, party, {64});
    model.conv1_2_w = create_dummy_tensor(exec, party, {64, 64, 3, 3});
    model.conv1_2_b = create_dummy_tensor(exec, party, {64});

    model.conv2_1_w = create_dummy_tensor(exec, party, {128, 64, 3, 3});
    model.conv2_1_b = create_dummy_tensor(exec, party, {128});
    model.conv2_2_w = create_dummy_tensor(exec, party, {128, 128, 3, 3});
    model.conv2_2_b = create_dummy_tensor(exec, party, {128});

    model.conv3_1_w = create_dummy_tensor(exec, party, {256, 128, 3, 3});
    model.conv3_1_b = create_dummy_tensor(exec, party, {256});
    model.conv3_2_w = create_dummy_tensor(exec, party, {256, 256, 3, 3});
    model.conv3_2_b = create_dummy_tensor(exec, party, {256});
    model.conv3_3_w = create_dummy_tensor(exec, party, {256, 256, 3, 3});
    model.conv3_3_b = create_dummy_tensor(exec, party, {256});

    model.conv4_1_w = create_dummy_tensor(exec, party, {512, 256, 3, 3});
    model.conv4_1_b = create_dummy_tensor(exec, party, {512});
    model.conv4_2_w = create_dummy_tensor(exec, party, {512, 512, 3, 3});
    model.conv4_2_b = create_dummy_tensor(exec, party, {512});
    model.conv4_3_w = create_dummy_tensor(exec, party, {512, 512, 3, 3});
    model.conv4_3_b = create_dummy_tensor(exec, party, {512});

    model.conv5_1_w = create_dummy_tensor(exec, party, {512, 512, 3, 3});
    model.conv5_1_b = create_dummy_tensor(exec, party, {512});
    model.conv5_2_w = create_dummy_tensor(exec, party, {512, 512, 3, 3});
    model.conv5_2_b = create_dummy_tensor(exec, party, {512});
    model.conv5_3_w = create_dummy_tensor(exec, party, {512, 512, 3, 3});
    model.conv5_3_b = create_dummy_tensor(exec, party, {512});

    int pooled_H = std::max(1, input_H / 32);
    int pooled_W = std::max(1, input_W / 32);
    int flattened_size = 512 * pooled_H * pooled_W;

    model.fc1_w = create_dummy_tensor(exec, party, {4096, flattened_size});
    model.fc1_b = create_dummy_tensor(exec, party, {4096});
    
    model.fc2_w = create_dummy_tensor(exec, party, {4096, 4096});
    model.fc2_b = create_dummy_tensor(exec, party, {4096});
    
    model.fc3_w = create_dummy_tensor(exec, party, {1000, 4096});
    model.fc3_b = create_dummy_tensor(exec, party, {1000});

    return model;
}

// ==========================================
// 主函数
// ==========================================
int main(int argc, char** argv) {
    SetLogLevel(LEVEL_ERROR);
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
    cout << "VGG-16 Full ZK Graph Test" << endl;
    cout << "Role: " << (party == PARTY_PROVER ? "PROVER" : "VERIFIER") << endl;
    cout << "=================================================" << endl;

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
        // 记录端到端总时间起点
        auto start_zk = high_resolution_clock::now();

        cout << "[TEST] Loading Dummy Image..." << endl;
        PolyTensor image = create_dummy_tensor(exec, party, {N, 3, H, W});

        cout << "[TEST] Instantiating VGG-16 Model Weights..." << endl;
        VGG16Weights model = vgg16(exec, party, H, W);

        // 前向传播
        PolyTensor output = VGG16_Forward(image, model, bitlen, digdec_k, do_truncation);

        cout << "[TEST] Output Shape: (" 
             << output.shape[0] << ", " << output.shape[1] << ")" << endl;

        cout << "[TEST] Securing final logits via Self-Relation..." << endl;
        PolyTensor::store_self_relation(output, "VGG16_Final_Logits_Check");

        // 密码学约束校验
        cout << "\n[TEST] Triggering ZK Constraints Verification..." << endl;
        exec->check_all();

        // 记录端到端总时间终点
        auto stop_zk = high_resolution_clock::now();
        auto total_ms = duration_cast<milliseconds>(stop_zk - start_zk).count();
        long long h  = total_ms / 3600000;
        long long m  = (total_ms % 3600000) / 60000;
        long long s  = (total_ms % 60000) / 1000;
        long long ms = total_ms % 1000;

        cout << "\n=================================================" << endl;
        cout << "[TIME] TOTAL PIPELINE: " << total_ms << " ms" << endl;
        cout << "[TIME] FORMATTED TIME: ";
        if (h > 0) cout << h << "h ";
        if (m > 0 || h > 0) cout << m << "m ";
        if (s > 0 || m > 0 || h > 0) cout << s << "s ";
        cout << ms << "ms" << endl;
        cout << "=================================================\n" << endl;
        cout << "\033[32m[SUCCESS] VGG-16 Test Passed Beautifully!\033[0m" << endl;

    } catch (const std::exception& e) {
        cerr << "\n\033[31m[FATAL ERROR] " << e.what() << "\033[0m" << endl;
    }

    if(exec) delete exec;
    delete io_arr;
    return 0;
}