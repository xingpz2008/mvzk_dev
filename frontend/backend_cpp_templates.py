"""
================================================================================
ZK-ML Compiler Configuration & Templates
================================================================================
This file contains the Operator Registry mapping and the static C++ templates.
Modify this file if the underlying C++ MPC framework's API changes.
================================================================================
"""

# ====================================================================
# MODULE 1: OPERATOR REGISTRY (CPP API Mapping)
# ====================================================================
CPP_API_MAP = {
    "conv2d": "PolyTensor {out_var} = Conv2D({in_var}, poly_weights.{node_name}_w, poly_weights.{node_name}_b, {stride}, {padding});",
    "linear": "PolyTensor {out_var} = Linear({in_var}, poly_weights.{node_name}_w, poly_weights.{node_name}_b);",
    "relu": "PolyTensor {out_var} = ReLU({in_var}, bitlen, digdec_k, do_truncation);",
    "integrated_nl": "PolyTensor {out_var} = IntegratedNL({in_var}, {kernel_size}, {stride}, {padding}, bitlen, digdec_k, do_truncation);",
    "max_pool2d": "PolyTensor {out_var} = MaxPool2D({in_var}, {kernel_size}, {stride}, {padding}, bitlen, digdec_k);",
    "avgpool2d": "PolyTensor {out_var} = AvgPool2D({in_var}, {kernel_size}, {stride}, {padding});",
    "global_avg_pool2d": "PolyTensor {out_var} = AvgPool2D({in_var}, {in_var}.shape[2], 1, 0);",
    "func_add": "PolyTensor {out_var} = {in_var_0} + {in_var_1};",
    "func_sub": "PolyTensor {out_var} = {in_var_0} - {in_var_1};",
    "func_mul": "PolyTensor {out_var} = {in_var_0} * {in_var_1};",
    "flatten": "PolyTensor {out_var} = {in_var}.flatten();",
    "unknown": "// [WARNING] Unmapped operation: {node_type} for {out_var}"
}

# ====================================================================
# MODULE 2: STATIC TEMPLATES
# ====================================================================

TEMPLATE_HEADER = """#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "data_type/PolyTensor.h"

// ==========================================
// Phase 1: Offline Host Weights (RAM Storage)
// ==========================================
struct {model_name}HostWeights {{
{host_fields}
}};

// ==========================================
// Phase 2: Online PolyTensor Weights (ZK Engine)
// ==========================================
struct {model_name}PolyWeights {{
{poly_fields}
}};

// ==========================================
// Function Declarations
// ==========================================
{model_name}HostWeights load_{model_name_lower}_host_weights(int party, const std::string& bin_dir);
{model_name}PolyWeights inject_{model_name_lower}_weights_to_zk(MVZKExec* exec, const {model_name}HostWeights& hw);
PolyTensor {model_name}_Forward(
    PolyTensor& {input_name}, 
    {model_name}PolyWeights& poly_weights,
    uint64_t bitlen, 
    uint64_t digdec_k, 
    bool do_truncation
);
"""

TEMPLATE_CPP = """#include "{model_name}.h"
#include "data_type/tensor_loader.h"
#include "operations/linear.h"
#include "operations/nonlinear.h"
#include "exec/MVZKExec.h"
#include <iostream>

{model_name}HostWeights load_{model_name_lower}_host_weights(int party, const std::string& bin_dir) {{
    {model_name}HostWeights hw;
{load_statements}
    return hw;
}}

{model_name}PolyWeights inject_{model_name_lower}_weights_to_zk(MVZKExec* exec, const {model_name}HostWeights& hw) {{
    {model_name}PolyWeights pw;
{inject_statements}
    return pw;
}}

PolyTensor {model_name}_Forward(PolyTensor& {input_name}, {model_name}PolyWeights& poly_weights, uint64_t bitlen, uint64_t digdec_k, bool do_truncation) {{
{forward_statements}
    return {output_name};
}}
"""

# 全量恢复：所有 C++ 大括号已安全双写为 {{ }}
TEMPLATE_TEST = """/*
 * Usage: 
 * ./model_test <party> <port> <N> <H> <W> <weights_dir>
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctime>
#include <omp.h>
#include <cstdlib>

#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "utility.h"       
#include "emp-tool/emp-tool.h"

// 引入自动生成的模型头文件
#include "{model_name}.h"

using namespace std;
using namespace emp;
using namespace std::chrono;

MVZKExec* MVZKExec::mvzk_exec = nullptr;
const int PARTY_PROVER = ALICE;
const int PARTY_VERIFIER = BOB;

// ==========================================
// 辅助工具：时间测量 (全量恢复)
// ==========================================
template <typename Func>
long long measure_time(Func&& func, string step_name) {{
    auto start = high_resolution_clock::now();
    func();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "[TIME] " << left << setw(45) << step_name << ": " << duration.count() << " ms" << endl;
    return duration.count();
}}

// ==========================================
// 辅助工具：张量随机生成器 (Dummy Data)
// ==========================================
PolyTensor create_dummy_tensor(MVZKExec* exec, int party, const vector<int>& shape) {{
    size_t total_size = 1;
    for (int s : shape) total_size *= s;

    vector<uint64_t> data(total_size, 0);
    if (party == PARTY_PROVER) {{
        static mt19937 gen(12345);
        uniform_int_distribution<int64_t> dis(-5, 5);
        for (size_t i = 0; i < total_size; ++i) {{
            float f_val = static_cast<float>(dis(gen)) / 100.0f;
            data[i] = real2fp(f_val);
        }}
    }}
    return exec->input(shape, data);
}}

// ==========================================
// 辅助工具：生成、打印并保存最终测试报告 (全量恢复)
// ==========================================
string strip_ansi(const string& input) {{
    string output;
    bool in_escape = false;
    for (char c : input) {{
        if (c == '\\033') {{
            in_escape = true;
        }} else if (in_escape) {{
            if (c == 'm') in_escape = false;
        }} else {{
            output += c;
        }}
    }}
    return output;
}}

void print_test_report(const string& model_name, int party, int N, int H, int W, 
                       uint64_t bitlen, uint64_t digdec_k, bool do_truncation, 
                       int num_threads, int omp_threads, uint64_t net_io_counter_start, uint64_t net_io_counter_end, 
                       const string& start_time_str, const string& stop_time_str, time_t stop_time_t, 
                       long long total_ms, const string& profiler_str) {{
    
    // 1. 换算时间格式
    long long h  = total_ms / 3600000;
    long long m  = (total_ms % 3600000) / 60000;
    long long s  = (total_ms % 60000) / 1000;
    long long ms = total_ms % 1000;

    // 2. 通信量计算
    uint64_t total_comm_bytes = net_io_counter_end - net_io_counter_start;
    double total_comm_kb = (double)total_comm_bytes / 1024.0;
    double total_comm_mb = (double)total_comm_bytes / (1024.0 * 1024.0);
    double total_comm_gb = total_comm_mb / 1024.0;

    // 3. 使用 stringstream 统一构建报告内容
    ostringstream oss;
    oss << "\\n=================================================\\n";
    oss << "                 FINAL TEST REPORT               \\n";
    oss << "=================================================\\n";
    oss << "[Model & Task Info]\\n";
    oss << "  - Model Name   : " << model_name << "\\n";
    oss << "  - Input Shape  : (N=" << N << ", C=3, H=" << H << ", W=" << W << ")\\n";
    oss << "  - Role         : " << (party == PARTY_PROVER ? "PROVER (Alice)" : "VERIFIER (Bob)") << "\\n";
    
    // 运行时传入的动态参数
    oss << "\\n[Test Runtime Parameters]\\n";
    oss << "  - Run: Bit Length   : " << bitlen << "\\n";
    oss << "  - Run: DigDec_k     : " << digdec_k << "\\n";
    oss << "  - Run: Truncation   : " << (do_truncation ? "Enabled" : "Disabled") << "\\n";
    oss << "  - Run: Threads (IO) : " << num_threads << "\\n";
    oss << "  - Run: Threads (OMP): " << omp_threads << "\\n";
    
    // config.h 中硬编码的系统底层静态配置
    oss << "\\n[Compiled System Configs (config.h)]\\n";
    oss << "  - MVZK_CONFIG_THREADS_NUM     : " << MVZK_CONFIG_THREADS_NUM << "\\n";
    oss << "  - MVZK_CONFIG_SCALE           : " << MVZK_CONFIG_SCALE << "\\n";
    oss << "  - MVZK_CONFIG_DEFAULT_BITLEN  : " << MVZK_CONFIG_DEFAULT_BITLEN << "\\n";
    oss << "  - MVZK_CONFIG_RELU_DIGDEC_K   : " << MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K << "\\n";
    oss << "  - MVZK_OMP_SIZE_THRESHOLD     : " << MVZK_OMP_SIZE_THRESHOLD << "\\n";
    oss << "  - MVZK_MULT_CHECK_CNT         : " << MVZK_MULT_CHECK_CNT << "\\n";
    oss << "  - MVZK_CONFIG_TENSOR_CHK_CNT  : " << MVZK_CONFIG_TENSOR_CHECK_CNT << "\\n";
    oss << "  - MVZK_CONFIG_RANGE_CHK_BUF   : " << MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD << "\\n";

    oss << "\\n[Execution Time]\\n";
    oss << "  - Start Time   : " << start_time_str << "\\n";
    oss << "  - End Time     : " << stop_time_str << "\\n";
    oss << "  - Total Time   : ";
    if (h > 0) oss << h << "h ";
    if (m > 0 || h > 0) oss << m << "m ";
    oss << s << "s " << ms << "ms\\n";
    
    oss << "\\n[Communication]\\n";
    oss << "  - Data Sent    : " << fixed << setprecision(2) << total_comm_mb << " MB";
    if (total_comm_gb > 1.0) oss << " (" << total_comm_gb << " GB)";
    oss << "\\n=================================================\\n";

    // 4. 打印到终端
    string report_str = oss.str();
    cout << report_str << endl; 
    cout << profiler_str;       

    // 5. 写入本地文件
    string role_str = (party == PARTY_PROVER) ? "PROVER" : "VERIFIER";
    
    char time_buf[64];
    strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", localtime(&stop_time_t));
    string safe_time_str(time_buf);

    string log_dir = "log/"; 
    system(("mkdir -p " + log_dir).c_str()); 

    string filename = log_dir + "report_" + model_name + "_" + role_str + "_" 
                      + to_string(N) + "x" + to_string(H) + "x" + to_string(W) 
                      + "_" + to_string(bitlen) + "bit_" 
                      + safe_time_str + ".log";
    
    ofstream outfile(filename);
    if (outfile.is_open()) {{
        outfile << report_str;
        outfile << strip_ansi(profiler_str);
        outfile.close();
        cout << "[INFO] Report successfully saved to: \\033[36m" << filename << "\\033[0m" << endl;
    }} else {{
        cerr << "\\033[31m[ERROR] Failed to save report to " << filename << "\\033[0m" << endl;
    }}
}}

// ==========================================
// 主函数
// ==========================================
int main(int argc, char** argv) {{
    SetLogLevel(LEVEL_ERROR);
    if (argc < 6) {{
        cout << "Usage: " << argv[0] << " <party> <port> <N> <H> <W> <weights_dir>" << endl;
        return 1;
    }}

    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    int N = atoi(argv[3]), H = atoi(argv[4]), W = atoi(argv[5]);
    std::string bin_dir = (argc >= 7) ? argv[6] : "NO_WEIGHTS_PROVIDED";

    uint64_t bitlen = MVZK_CONFIG_DEFAULT_BITLEN;
    uint64_t digdec_k = MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K;
    bool do_truncation = MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION;

    cout << "=================================================" << endl;
    cout << "{model_name} Full ZK Graph Test" << endl;
    cout << "Role: " << (party == PARTY_PROVER ? "PROVER" : "VERIFIER") << endl;
    cout << "=================================================" << endl;

    int num_threads = MVZK_CONFIG_THREADS_NUM;
    int omp_threads = omp_get_max_threads(); 

    NetIO **io_arr = new NetIO*[num_threads];
    for (int i = 0; i < num_threads; ++i) {{
        io_arr[i] = new NetIO(party == PARTY_PROVER ? nullptr : "127.0.0.1", port + i);
    }}

    MVZKExec *exec = nullptr;
    if (party == PARTY_PROVER) exec = new MVZKExecProver<NetIO>(io_arr);
    else exec = new MVZKExecVerifier<NetIO>(io_arr);

    try {{
        // 【两段式解耦设计】
        cout << "\\n[TEST] Phase 1: Offline Loading Weights from Disk (NOT TIMED)..." << endl;
        {model_name}HostWeights host_weights = load_{model_name_lower}_host_weights(party, bin_dir);

        uint64_t net_io_start = comm(io_arr, num_threads);
        
        // --------------------------------------------------
        // [起点时刻]：分别获取高精度测速时钟和系统日历时钟
        // --------------------------------------------------
        auto start_zk = high_resolution_clock::now(); 
        
        time_t start_time_t = system_clock::to_time_t(system_clock::now()); 
        string start_time_str = ctime(&start_time_t);
        start_time_str.pop_back(); 

        cout << "[TEST] Phase 2: Online Injecting Weights into ZK Engine..." << endl;
        PolyTensor image = create_dummy_tensor(exec, party, {{N, 3, H, W}});
        {model_name}PolyWeights poly_weights = inject_{model_name_lower}_weights_to_zk(exec, host_weights);

        cout << "[TEST] Phase 3: Executing {model_name} ZK Forward Pass..." << endl;
        PolyTensor output = {model_name}_Forward(image, poly_weights, bitlen, digdec_k, do_truncation);

        PolyTensor::store_self_relation(output, "{model_name}_Final_Logits_Check");
        exec->check_all();
        
        // --------------------------------------------------
        // [终点时刻]：分别获取高精度测速时钟和系统日历时钟
        // --------------------------------------------------
        auto stop_zk = high_resolution_clock::now(); 
        
        time_t stop_time_t = system_clock::to_time_t(system_clock::now()); 
        string stop_time_str = ctime(&stop_time_t);
        stop_time_str.pop_back(); 

        uint64_t net_io_end = comm(io_arr, num_threads);
        auto total_ms = duration_cast<milliseconds>(stop_zk - start_zk).count();

        // --------------------------------------------------
        // 流重定向拦截：捕获 Profiler 的终端打印结果
        // --------------------------------------------------
        stringstream profiler_buffer;
        streambuf* old_cout = cout.rdbuf(profiler_buffer.rdbuf()); 
        exec->print_profiler_report(); 
        cout.rdbuf(old_cout); 
        string captured_profiler_str = profiler_buffer.str(); 

        // --------------------------------------------------
        // 输出格式化报告
        // --------------------------------------------------
        print_test_report("{model_name}", party, N, H, W, bitlen, digdec_k, do_truncation, 
                          num_threads, omp_threads, net_io_start, net_io_end, start_time_str, stop_time_str, 
                          stop_time_t, total_ms, captured_profiler_str);

        // ==================================================
        // 用户需求：在屏幕最底部，额外打印一次醒目的总时间
        // ==================================================
        long long h  = total_ms / 3600000;
        long long m  = (total_ms % 3600000) / 60000;
        long long s  = (total_ms % 60000) / 1000;
        long long ms = total_ms % 1000;

        cout << "\\n\\033[1;36m[====> TOTAL EXECUTION TIME <====]\\033[0m" << endl;
        cout << "  ";
        if (h > 0) cout << h << "h ";
        if (m > 0 || h > 0) cout << m << "m ";
        cout << s << "s " << ms << "ms  (" << total_ms << " ms)\\n" << endl;
        cout << "=================================================\\n" << endl;
        
        cout << "\\033[32m[SUCCESS] {model_name} Test Passed Beautifully!\\033[0m" << endl;

    }} catch (const std::exception& e) {{
        cerr << "\\n\\033[31m[FATAL ERROR] " << e.what() << "\\033[0m" << endl;
    }}

    if(exec) delete exec;

    for (int i = 0; i < num_threads; ++i) {{
        delete io_arr[i]; 
    }}
    delete[] io_arr;

    return 0;
}}
"""