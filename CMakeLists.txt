# ==============================================================================
# MVZK 工程 CMake 构建脚本 & 使用指南
# ==============================================================================
#
# 【基础编译】
#   mkdir build && cd build
#   cmake .. 
#   make -j
#   * 默认以 Release 模式编译（开启 LTO 和 -O3 优化）。
#
# 【调试模式编译】
#   cmake .. -DCMAKE_BUILD_TYPE=Debug
#
# 
#   随着工程增大，全量编译极其耗时。本脚本支持通过 -DTEST_SELECT 参数精准指定编译目标。
#   
#   用法示例: cmake .. -DTEST_SELECT=<模式>
#
#   支持的模式列表：
#   - "ALL"        (默认): 编译所有手写测试 (test/) + 所有自动生成的模型 (generated_model/)
#   - "DEV"        : 仅编译底层开发测试 (test/dev_test/*.cpp)
#   - "FUNC"       : 仅编译算子功能验证测试 (test/functional_test/*.cpp)
#   - "LOCALMODEL" : 仅编译手动编写的本地模型测试 (test/local_model_test/*.cpp)
#   - "GEN"        : 仅编译 Python 工具链自动生成的模型 (generated_model/*_test.cpp)。
#                    (自动将测试框架与算子实现文件联合编译)
#   - "<特定文件名>" : 支持模糊匹配。如 `cmake .. -DTEST_SELECT=conv2d` 将只编译含 conv2d 的测试文件。
#
# 【底层依赖与隔离】
#   - 默认全局搜寻并启用 jemalloc 以优化内存碎片，若无则静默回退系统分配器。
#   - 严格的源码扫描隔离：自动屏蔽 frontend/ 目录，防止引入 Python 虚拟环境中的 C++ 源码。
#
# ==============================================================================

cmake_minimum_required(VERSION 3.14)
project(MVZK LANGUAGES CXX)

# ==========================================
# 1. macOS 环境配置 (OpenMP & OpenSSL)
# ==========================================
if(APPLE)
    # 自动寻找 Homebrew 安装路径
    if(EXISTS "/opt/homebrew") # Apple Silicon (M1/M2/M3)
        set(HOMEBREW_PREFIX "/opt/homebrew")
    elseif(EXISTS "/usr/local") # Intel
        set(HOMEBREW_PREFIX "/usr/local")
    endif()

    if(DEFINED HOMEBREW_PREFIX)
        list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")
        include_directories("${HOMEBREW_PREFIX}/include")
        link_directories("${HOMEBREW_PREFIX}/lib")
        
        # 特殊处理 OpenSSL (因为 brew 版 openssl 不在标准路径)
        execute_process(COMMAND brew --prefix openssl@3 
                        OUTPUT_VARIABLE OPENSSL_ROOT_DIR 
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        list(APPEND CMAKE_PREFIX_PATH "${OPENSSL_ROOT_DIR}")
    endif()
endif()

# ==========================================
# 2. 基础编译选项 (极致性能版)
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认构建类型设置为 Release (防止忘记加 -D 参数)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# 通用 Flag (架构相关)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(ARCH_FLAGS -Wno-ignored-attributes -Wno-unused-result)
else()
    # 针对 AWS Intel 芯片的极限配置 (-unroll-loops 展开循环进一步压榨流水线)
    set(ARCH_FLAGS -maes -msse4.1 -mpclmul -march=native -funroll-loops -Wno-ignored-attributes -Wno-unused-result)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Build type: DEBUG (Optimizations OFF)")
    add_compile_options(${ARCH_FLAGS} -O0 -g -ggdb -fno-inline)
else()
    message(STATUS "Build type: RELEASE (Optimizations ON + LTO)")
    add_compile_options(${ARCH_FLAGS} -O3 -g -fno-omit-frame-pointer)
    
    # 开启 LTO 链接期优化 (神级性能开关)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT error)
    if(ipo_supported)
        message(STATUS "IPO / LTO supported and enabled.")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO / LTO not supported: ${error}")
    endif()
endif()

# ==========================================
# 3. 查找依赖库
# ==========================================
if(APPLE)
    if(EXISTS "${HOMEBREW_PREFIX}/opt/libomp")
        set(OpenMP_ROOT "${HOMEBREW_PREFIX}/opt/libomp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${OpenMP_ROOT}/lib/libomp.dylib")
        
        include_directories("${OpenMP_ROOT}/include")
        link_directories("${OpenMP_ROOT}/lib")
    endif()
endif()

find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)

# 查找 EMP-Tool
find_path(EMP_TOOL_INC "emp-tool/emp-tool.h")
find_library(EMP_TOOL_LIB NAMES emp-tool)
if (NOT EMP_TOOL_LIB)
    message(WARNING "emp-tool not found! Please ensure it is installed.")
endif()

# ==========================================
# [性能优化选项]：默认全局尝试开启 jemalloc
# ==========================================
# 优雅生成 ANSI 颜色控制符的前缀 (ASCII 码 27 就是 ESC)
string(ASCII 27 Esc)

option(USE_JEMALLOC "Use jemalloc for memory allocation" ON)

if(USE_JEMALLOC)
    # 自动跨平台寻找 jemalloc，加入了 Mac Homebrew 的常见路径以防万一
    find_library(JEMALLOC_LIB NAMES jemalloc
        PATHS
        /opt/homebrew/lib   # Apple Silicon Mac (M1/M2/M3)
        /usr/local/lib      # Intel Mac / Linux 源码编译
        /usr/lib            # Linux apt-get 安装路径
        /usr/lib/x86_64-linux-gnu # Ubuntu 常见路径
    )
    
    if(JEMALLOC_LIB)
        # 找到了：绿字提示，并全局注入 (使用 ${Esc} 替代 \033)
        message(STATUS "${Esc}[32m[Performance Opt] Found jemalloc at ${JEMALLOC_LIB}. Enabling globally!${Esc}[0m")
        link_libraries(${JEMALLOC_LIB}) 
    else()
        # 没找到：黄字提示，直接忽略，绝不中断编译！
        message(WARNING "${Esc}[33m[Performance Opt] jemalloc not found. Falling back to default system allocator.${Esc}[0m")
    endif()
else()
    # 手动关闭：蓝字提示
    message(STATUS "${Esc}[34m[Performance Opt] USE_JEMALLOC is OFF. Using default system allocator.${Esc}[0m")
endif()

# ==========================================
# 4. 自动扫描源文件 (安全版，防虚拟环境入侵)
# ==========================================

# 步骤 1：递归扫描 (GLOB_RECURSE) 明确安全的纯 C++ 子文件夹
file(GLOB_RECURSE LIBRARY_SOURCES 
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/data_type/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/operations/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/exec/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/local_models/*.cpp"
)

# 步骤 2：非递归扫描 (GLOB) 根目录下的孤立 .cpp 文件（比如 config.cpp 等）
# 绝对不要在这里用 GLOB_RECURSE，否则会把 python 虚拟环境里的 cpp 全拉进来！
file(GLOB ROOT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
if(ROOT_SOURCES)
    list(APPEND LIBRARY_SOURCES ${ROOT_SOURCES})
endif()

# 步骤 3：从列表中剔除 test 文件夹下的文件 (保持原样)
file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
if(TEST_SOURCES)
    list(REMOVE_ITEM LIBRARY_SOURCES ${TEST_SOURCES})
endif()
# 调试：打印一下到底编译了哪些文件 (可选)
# message(STATUS "Library Sources: ${LIBRARY_SOURCES}")

# ==========================================
# 5. 生成核心静态库 (mvzk_core)
# ==========================================
add_library(mvzk_core STATIC ${LIBRARY_SOURCES})

# 设置头文件包含路径
target_include_directories(mvzk_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}        # 允许 #include "data_type/..."
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ # 允许 #include "emp-zk/emp-mvzk/..." (向上两级)
    ${EMP_TOOL_INC}
    ${OpenMP_CXX_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

# 链接依赖
target_link_libraries(mvzk_core PUBLIC
    OpenMP::OpenMP_CXX
    ${EMP_TOOL_LIB}
    OpenSSL::Crypto
    OpenSSL::SSL
    pthread
)

# ==========================================
# 6. 生成测试可执行文件 (从 test/ 文件夹)
# ==========================================

# ==============================================================================
# Testing Configuration
# ==============================================================================

# 1. 定义控制变量 TEST_SELECT
# 默认值为 "ALL"
# 选项: 
#   "ALL"  : 编译 test/ 下所有子文件夹的所有常规测试 + generated_model/ 下所有自动生成的模型
#   "DEV"  : 只编译 test/dev_test/ 下的 .cpp
#   "FUNC" : 只编译 test/functional_test/ 下的 .cpp
#   "LOCALMODEL": 只编译 test/local_model_test/ 下的 .cpp
#   "GEN"  : 只编译 generated_model/ 下 Python 自动生成的 ZK 模型
#   "文件名": 编译指定的文件 (例如 "test_operators")
set(TEST_SELECT "ALL" CACHE STRING "Select tests to build: ALL, DEV, FUNC, LOCALMODEL, GEN, or specific filename")

message(STATUS "------------------------------------------------")
message(STATUS "Build Test Mode: ${TEST_SELECT}")

# 2. 准备两个独立的列表
set(TEST_SOURCES "")     # 用于存放常规的单文件测试
set(GEN_TEST_SOURCES "") # 用于存放自动生成的双文件模型测试

if(TEST_SELECT STREQUAL "ALL")
    # 【模式：ALL】同时抓取传统的 test 和新生成的 generated_model
    file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
    file(GLOB_RECURSE GEN_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/generated_model/*_test.cpp")

elseif(TEST_SELECT STREQUAL "DEV")
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/dev_test/*.cpp")

elseif(TEST_SELECT STREQUAL "FUNC")
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/functional_test/*.cpp")

elseif(TEST_SELECT STREQUAL "LOCALMODEL")
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/local_model_test/*.cpp")

elseif(TEST_SELECT STREQUAL "GEN")
    # 【模式：GEN】只抓取 generated_model 下的文件
    file(GLOB_RECURSE GEN_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/generated_model/*_test.cpp")

else()
    # 自定义文件名匹配 (只在常规 test 目录里找)
    file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*${TEST_SELECT}*.cpp")
endif()

# ==============================================================================
# 3. 编译流程 A：处理常规的单文件测试 (来自 test/ 目录)
# ==============================================================================
if(TEST_SOURCES)
    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_target_name ${test_src} NAME_WE)
        message(STATUS "  [+] Adding standard test target: ${test_target_name}")

        add_executable(${test_target_name} ${test_src})
        target_link_libraries(${test_target_name} PRIVATE mvzk_core)
    endforeach()
endif()

# ==============================================================================
# 4. 编译流程 B：处理自动生成的双文件模型 (来自 generated_model/ 目录)
# ==============================================================================
# ==============================================================================
# 4. 编译流程 B：处理自动生成的双文件模型 (来自 generated_model/ 目录)
# ==============================================================================
if(GEN_TEST_SOURCES)
    foreach(test_src ${GEN_TEST_SOURCES})
        # 获取基础文件名 (例如 ResNet18_test)
        get_filename_component(base_name ${test_src} NAME_WE)
        
        # 获取父级文件夹的绝对路径 (例如 .../generated_model/resnet18_20260309_141521)
        get_filename_component(parent_dir ${test_src} DIRECTORY)
        # 提取最后一级文件夹名
        get_filename_component(folder_name ${parent_dir} NAME)
        
        # 拼接成独一无二的 Target 名字！防止 CMake 内部重名报错
        set(test_target_name "${folder_name}_${base_name}")
        
        # 自动推导它的“亲兄弟”实现文件
        string(REPLACE "_test.cpp" ".cpp" model_src ${test_src})

        message(STATUS "  [+] Adding generated model target: ${test_target_name} -> Output to ${parent_dir}")

        # 编译时使用全局唯一的名字
        add_executable(${test_target_name} ${test_src} ${model_src})

        # 链接核心库，并强行把根目录加入头文件搜索路径
        target_link_libraries(${test_target_name} PRIVATE mvzk_core)
        target_include_directories(${test_target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

        # 1. RUNTIME_OUTPUT_DIRECTORY: 将编译好的二进制文件直接放进 generated_model/xxx 文件夹
        # 2. OUTPUT_NAME: 把长长的唯一 Target 名字重命名回清爽的 "ResNet18_test"
        set_target_properties(${test_target_name} PROPERTIES 
            RUNTIME_OUTPUT_DIRECTORY "${parent_dir}"
            OUTPUT_NAME "${base_name}"
        )
    endforeach()
endif()

# 5. 如果啥都没找到的报警
if(NOT TEST_SOURCES AND NOT GEN_TEST_SOURCES)
    message(WARNING "  [-] No test source files found matching: ${TEST_SELECT}")
endif()

message(STATUS "------------------------------------------------")