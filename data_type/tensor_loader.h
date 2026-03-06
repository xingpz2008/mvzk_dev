#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <iostream>

// 注意：这里不再需要引入 PolyTensor.h 和 MVZKExec.h！彻底与 ZK 引擎解耦
#include "../config.h"
#include "../utility.h" // 包含 ALICE, BOB, real2fp 等定义

// ====================================================================
// Weight Data Type Enum: To differentiate the underlying format of .bin files
// ====================================================================
enum class TensorDataType {
    FP32,  // Unquantized: Raw 32-bit float (requires real2fp conversion)
    INT8,  // Quantized: 8-bit integer (e.g., from PTQ, direct conversion)
    INT32  // Quantized: 32-bit integer (typically for quantized biases)
};

// ====================================================================
// Offline Loader: Reads weights from disk and converts them to fixed-point 
// integers in standard C++ memory. DOES NOT trigger any ZK protocol.
// ====================================================================
inline std::vector<uint64_t> load_raw_data_from_bin(
    int party, 
    const std::vector<int>& shape, 
    const std::string& filepath, 
    TensorDataType dtype = TensorDataType::FP32) 
{
    // 1. Calculate the flattened size of the tensor
    size_t total_size = 1;
    for (int s : shape) total_size *= s;

    // 2. Initialize host memory container (defaults to all zeros)
    std::vector<uint64_t> data(total_size, 0);

    // 3. Only Prover (Alice) reads from the hard drive
    if (party == ALICE) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("[ERROR] Prover cannot open weight file: " + filepath);
        }

        // --- Branch A: Handle unquantized floats ---
        if (dtype == TensorDataType::FP32) {
            std::vector<float> buffer(total_size);
            file.read(reinterpret_cast<char*>(buffer.data()), total_size * sizeof(float));
            
            for (size_t i = 0; i < total_size; ++i) {
                // Convert float to finite field fixed-point integer
                data[i] = real2fp(buffer[i]); 
            }
        } 
        // --- Branch B: Handle INT8 quantized models ---
        else if (dtype == TensorDataType::INT8) {
            std::vector<int8_t> buffer(total_size);
            file.read(reinterpret_cast<char*>(buffer.data()), total_size * sizeof(int8_t));
            
            for (size_t i = 0; i < total_size; ++i) {
                int64_t val = static_cast<int64_t>(buffer[i]);
                data[i] = static_cast<uint64_t>(val); 
            }
        }
        // --- Branch C: Handle INT32 quantized models ---
        else if (dtype == TensorDataType::INT32) {
            std::vector<int32_t> buffer(total_size);
            file.read(reinterpret_cast<char*>(buffer.data()), total_size * sizeof(int32_t));
            for (size_t i = 0; i < total_size; ++i) {
                data[i] = static_cast<uint64_t>(static_cast<int64_t>(buffer[i]));
            }
        }

        file.close();
    }
    
    // Verifier (Bob) skips the file I/O and simply returns the zeroed vector
    
    // 4. Return standard C++ memory (No ZK protocol is invoked here)
    return data;
}