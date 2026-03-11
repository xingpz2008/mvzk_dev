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

# ====================================================================
# [全量覆盖] 替换 backend_cpp_templates.py 底部的 TEMPLATE_TEST 变量
# ====================================================================

TEMPLATE_TEST = """/*
 * Usage: 
 *  
 * ./<binary_name> <party> <port> <model_dir>
 *
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
#include <limits>

#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "utility.h"       
#include "emp-tool/emp-tool.h"
#include "data_type/tensor_loader.h"

// 引入自动生成的模型头文件
#include "{model_name}.h"

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
long long measure_time(Func&& func, string step_name) {{
    auto start = high_resolution_clock::now();
    func();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "[TIME] " << left << setw(45) << step_name << ": " << duration.count() << " ms" << endl;
    return duration.count();
}}

// ==========================================
// 辅助工具：脱敏 ANSI 颜色码
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

// ==========================================
// 辅助工具：在内存中排版基础测试报告
// ==========================================
string format_base_report(const string& model_name, int party, int N, int C, int H, int W, 
                       uint64_t bitlen, uint64_t digdec_k, bool do_truncation, 
                       int num_threads, int omp_threads, uint64_t net_io_counter_start, uint64_t net_io_counter_end, 
                       const string& start_time_str, const string& stop_time_str, long long total_ms) {{
    
    long long h  = total_ms / 3600000;
    long long m  = (total_ms % 3600000) / 60000;
    long long s  = (total_ms % 60000) / 1000;
    long long ms = total_ms % 1000;

    uint64_t total_comm_bytes = net_io_counter_end - net_io_counter_start;
    double total_comm_kb = (double)total_comm_bytes / 1024.0;
    double total_comm_mb = (double)total_comm_bytes / (1024.0 * 1024.0);
    double total_comm_gb = total_comm_mb / 1024.0;

    ostringstream oss;
    oss << "\\n=================================================\\n";
    oss << "                 FINAL TEST REPORT               \\n";
    oss << "=================================================\\n";
    oss << "[Model & Task Info]\\n";
    oss << "  - Model Name   : " << model_name << "\\n";
    oss << "  - Input Shape  : (N=" << N << ", C=" << C << ", H=" << H << ", W=" << W << ")\\n";
    oss << "  - Role         : " << (party == PARTY_PROVER ? "PROVER (Alice)" : "VERIFIER (Bob)") << "\\n";
    oss << "  - Source       : Frontend Generated \\n";
    
    oss << "\\n[Test Runtime Parameters]\\n";
    oss << "  - Run: Bit Length   : " << bitlen << "\\n";
    oss << "  - Run: DigDec_k     : " << digdec_k << "\\n";
    oss << "  - Run: Truncation   : " << (do_truncation ? "Enabled" : "Disabled") << "\\n";
    oss << "  - Run: Threads (IO) : " << num_threads << "\\n";
    oss << "  - Run: Threads (OMP): " << omp_threads << "\\n";
    
    oss << "\\n[Compiled System Configs (config.h)]\\n";
    oss << "  - MVZK_CONFIG_THREADS_NUM      : " << MVZK_CONFIG_THREADS_NUM << "\\n";
    oss << "  - MVZK_CONFIG_SCALE            : " << MVZK_CONFIG_SCALE << "\\n";
    oss << "  - MVZK_CONFIG_DEFAULT_BITLEN   : " << MVZK_CONFIG_DEFAULT_BITLEN << "\\n";
    oss << "  - MVZK_CONFIG_RELU_DIGDEC_K    : " << MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K << "\\n";
    oss << "  - MVZK_OMP_SIZE_THRESHOLD      : " << MVZK_OMP_SIZE_THRESHOLD << "\\n";
    oss << "  - MVZK_MULT_CHECK_CNT          : " << MVZK_MULT_CHECK_CNT << "\\n";
    oss << "  - MVZK_CONFIG_TENSOR_CHK_CNT   : " << MVZK_CONFIG_TENSOR_CHECK_CNT << "\\n";
    oss << "  - MVZK_CONFIG_RANGE_CHK_BUF    : " << MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD << "\\n";
    oss << "  - MVZK_CONFIG_MULT_PRODUCT_THR : " << MVZK_CONFIG_MULT_PRODUCT_THRESHOLD << "\\n";

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

    return oss.str();
}}

// ==========================================
// 主函数
// ==========================================
int main(int argc, char** argv) {{
    SetLogLevel(LEVEL_ERROR);
    // [极简传参] 现在只需要 4 个参数：程序名、身份、端口、模型主目录
    if (argc < 4) {{
        cout << "Usage: " << argv[0] << " <party> <port> <model_dir>" << endl;
        return 1;
    }}
    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    std::string model_dir = argv[3];

    // [路径自动推导] C++ 自己拼接出权重和输入的路径，告别冗长的命令行！
    std::string bin_dir = model_dir + "/weights";
    std::string input_bin_path = model_dir + "/test_cases/case_1/input_0.bin";

    // [状态管理] SSOT: 尺寸信息已被 Python AOT 编译器硬编码注入，无需再通过命令行手动传入！
    const int N = {N};
    const int C = {C};
    const int H = {H};
    const int W = {W};

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
        cout << "\\n[TEST] Phase 1: Offline Loading Weights and Inputs from Disk (NOT TIMED)..." << endl;
        {model_name}HostWeights host_weights = load_{model_name_lower}_host_weights(party, bin_dir);
        std::vector<uint64_t> host_input_data = load_raw_data_from_bin(party, {{N, C, H, W}}, input_bin_path);

        uint64_t net_io_start = comm(io_arr, num_threads);
        auto start_zk = high_resolution_clock::now(); 
        time_t start_time_t = system_clock::to_time_t(system_clock::now()); 
        string start_time_str = ctime(&start_time_t);
        start_time_str.pop_back(); 

        cout << "[TEST] Phase 2: Commit weights and inputs with ZK backend." << endl;
        {model_name}PolyWeights poly_weights = inject_{model_name_lower}_weights_to_zk(exec, host_weights);
        PolyTensor image = exec->input({{N, C, H, W}}, host_input_data);

        cout << "[TEST] Phase 3: Executing {model_name} ZK Forward Pass..." << endl;
        PolyTensor output = {model_name}_Forward(image, poly_weights, bitlen, digdec_k, do_truncation);

        PolyTensor::store_self_relation(output, "{model_name}_Final_Logits_Check");
        exec->check_all();
        
        auto stop_zk = high_resolution_clock::now(); 
        time_t stop_time_t = system_clock::to_time_t(system_clock::now()); 
        string stop_time_str = ctime(&stop_time_t);
        stop_time_str.pop_back(); 

        uint64_t net_io_end = comm(io_arr, num_threads);
        auto total_ms = duration_cast<milliseconds>(stop_zk - start_zk).count();

        // --------------------------------------------------
        // 在内存中构建所有报告字符串
        // --------------------------------------------------
        // 1. 截获 Profiler 输出
        stringstream profiler_buffer;
        streambuf* old_cout = cout.rdbuf(profiler_buffer.rdbuf()); 
        exec->print_profiler_report(); 
        cout.rdbuf(old_cout); 
        string captured_profiler_str = profiler_buffer.str(); 

        // 2. 生成基础执行报告
        string base_report_str = format_base_report("{model_name}", party, N, C, H, W, bitlen, digdec_k, do_truncation, 
                                                   num_threads, omp_threads, net_io_start, net_io_end, start_time_str, stop_time_str, total_ms);

        // 实时打印基础报告与 Profiler 到终端
        cout << base_report_str << captured_profiler_str;

        long long h  = total_ms / 3600000;
        long long m  = (total_ms % 3600000) / 60000;
        long long s  = (total_ms % 60000) / 1000;
        long long ms = total_ms % 1000;

        double total_comm_mb = (double)(net_io_end - net_io_start) / (1024.0 * 1024.0);
        
        cout << "\\n\\033[1;36m[====> ZK PERFORMANCE METRICS <====]\\033[0m" << endl;
        cout << "  - Total Time  : ";
        if (h > 0) cout << h << "h ";
        if (m > 0 || h > 0) cout << m << "m ";
        cout << s << "s " << ms << "ms  (" << total_ms << " ms)\\n";
        cout << "  - Total Comm  : " << fixed << setprecision(2) << total_comm_mb << " MB\\n";
        cout << "=================================================\\n" << endl;
        
        // ==================================================
        // Phase 4: 明文提取与 Ground Truth 精度对齐
        // ==================================================
        string align_report_str = ""; // 用于存放对齐日志

        cout << "[TEST] Phase 4: Revealing and Verifying Output against Oracle..." << endl;
        std::vector<uint64_t> raw_logits = exec->reveal(output);

        if (party == PARTY_VERIFIER && !raw_logits.empty()) {{
            size_t last_slash = input_bin_path.find_last_of('/');
            std::string expected_bin_path = (last_slash == std::string::npos) ? "expected_logits.bin" : input_bin_path.substr(0, last_slash) + "/expected_logits.bin";

            std::ifstream ifs(expected_bin_path, std::ios::binary);
            std::vector<float> expected_logits;
            if(ifs.is_open()) {{
                ifs.seekg(0, std::ios::end);
                size_t size = ifs.tellg();
                ifs.seekg(0, std::ios::beg);
                expected_logits.resize(size / sizeof(float));
                ifs.read(reinterpret_cast<char*>(expected_logits.data()), size);
                ifs.close();
            }}

            if (!expected_logits.empty()) {{
                if (raw_logits.size() != expected_logits.size()) {{
                    cout << "\\n\\033[1;31m[FATAL ERROR] Dimension mismatch! ZK output size (" 
                         << raw_logits.size() << ") != Oracle expected size (" 
                         << expected_logits.size() << ").\\033[0m" << endl;
                    exit(-1);
                }}
                double mse = 0.0;
                int match_count = 0;
                size_t num_classes = raw_logits.size() / N; 
                
                ostringstream align_stream;
                
                for (int b = 0; b < N; ++b) {{
                    int zk_argmax = 0;
                    int pt_argmax = 0;
                    double zk_max = std::numeric_limits<double>::lowest(); 
                    float pt_max = std::numeric_limits<float>::lowest();

                    const uint64_t PRIME = PR;
                    const uint64_t HALF_PRIME = PRIME / 2;
                    
                    ostringstream img_debug;
                    img_debug << "\\n\\033[1;33m[DEBUG] --- Logits Comparison (Image " << b << ", First 15 Classes) ---\\033[0m\\n";
                    img_debug << left << setw(10) << "Class" 
                              << setw(22) << "Raw Int"
                              << setw(15) << "ZK Output" 
                              << setw(15) << "Oracle" 
                              << "Abs Diff\\n";
                    
                    for(size_t c = 0; c < num_classes; ++c) {{
                        size_t i = b * num_classes + c; 
                        uint64_t field_val = raw_logits[i];
                        int64_t signed_val;
                        
                        if (field_val > HALF_PRIME) {{
                            signed_val = static_cast<int64_t>(field_val) - static_cast<int64_t>(PRIME);
                        }} else {{
                            signed_val = static_cast<int64_t>(field_val);
                        }}
                        
                        double pt_val = expected_logits[i];

                        double scale_factor = static_cast<double>(1ULL << MVZK_CONFIG_SCALE);
                        double total_scale = 1.0;
                        for(int s = 0; s < {final_scale_power}; ++s) {{
                            total_scale *= scale_factor;
                        }}
                        double zk_val = (double)signed_val / total_scale;
                        
                        double err = zk_val - pt_val;
                        mse += err * err;

                        if (zk_val > zk_max) {{
                            zk_max = zk_val;
                            zk_argmax = c;
                        }}
                        if (pt_val > pt_max) {{
                            pt_max = pt_val;
                            pt_argmax = c;
                        }}

                        if (c < 15) {{
                            img_debug << left << setw(10) << c 
                                      << setw(22) << signed_val 
                                      << setw(15) << fixed << setprecision(6) << zk_val 
                                      << setw(15) << pt_val 
                                      << abs(zk_val - pt_val) << "\\n";
                        }}
                    }}
                    
                    bool is_match = (zk_argmax == pt_argmax);
                    if (is_match) {{
                        match_count++;
                    }}
                    
                    align_stream << "\\n\\033[1;36m[PREDICT] Image " << b << " | ZK Argmax = " << zk_argmax << " (Val: " << zk_max 
                                 << ") || Oracle Argmax = " << pt_argmax << " (Val: " << pt_max << ")\\033[0m\\n";
                    
                    if (!is_match) {{
                        align_stream << img_debug.str();
                    }}
                }}
                
                mse /= raw_logits.size();

                align_stream << "\\n\\033[1;35m[====> GROUND TRUTH ALIGNMENT <====]\\033[0m\\n";
                align_stream << "  - Oracle Answer Loaded : " << expected_bin_path << "\\n";
                align_stream << "  - Mean Squared Error   : " << scientific << mse << fixed << "\\n";
                if (match_count == N) {{
                    align_stream << "  - Classification       : \\033[1;32m[ MATCH! (" << match_count << "/" << N << ") ]\\033[0m\\n";
                }} else {{
                    align_stream << "  - Classification       : \\033[1;31m[ MISMATCH! (" << match_count << "/" << N << ") ]\\033[0m\\n";
                }}
                align_stream << "=================================================\\n\\n";
                
                // 打印对齐报告到终端，并将其保存到最终的日志字符串中
                align_report_str = align_stream.str();
                cout << align_report_str;
                
            }} else {{
                cout << "\\033[33m[WARNING] expected_logits.bin not found at " << expected_bin_path << ". Skipping MSE check.\\033[0m\\n" << endl;
            }}
        }}

        // ==================================================
        // 终极单次磁盘 I/O (The Ultimate Atomic File Write)
        // ==================================================
        string role_str = (party == PARTY_PROVER) ? "PROVER" : "VERIFIER";
        char time_buf[64];
        strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", localtime(&stop_time_t));
        string safe_time_str(time_buf);

        string log_dir = "log/"; 
        system(("mkdir -p " + log_dir).c_str()); 

        string filename = log_dir + "report_{model_name}_" + role_str + "_" 
                          + to_string(N) + "x" + to_string(H) + "x" + to_string(W) 
                          + "_" + to_string(bitlen) + "bit_" 
                          + safe_time_str + ".log";

        // 将基础信息、性能信息、对齐信息拼接在一起，一次性写入！
        string full_report = base_report_str + captured_profiler_str + align_report_str;
        
        ofstream outfile(filename);
        if (outfile.is_open()) {{
            outfile << strip_ansi(full_report);
            outfile.close();
            cout << "[INFO] Final complete report successfully saved to: \\033[36m" << filename << "\\033[0m\\n" << endl;
        }} else {{
            cerr << "\\033[31m[ERROR] Failed to save report to " << filename << "\\033[0m\\n" << endl;
        }}
        
        cout << "\\033[32m[SUCCESS] {model_name} Test Execution Complete!\\033[0m" << endl;

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