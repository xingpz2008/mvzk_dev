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
        # 找到了：绿字提示，并全局注入
        message(STATUS "\033[32m[Performance Opt] Found jemalloc at ${JEMALLOC_LIB}. Enabling globally!\033[0m")
        link_libraries(${JEMALLOC_LIB}) 
    else()
        # 没找到：黄字提示，直接忽略，绝不中断编译！
        message(WARNING "\033[33m[Performance Opt] jemalloc not found. Falling back to default system allocator.\033[0m")
    endif()
else()
    # 手动关闭：蓝字提示
    message(STATUS "\033[34m[Performance Opt] USE_JEMALLOC is OFF. Using default system allocator.\033[0m")
endif()

# ==========================================
# 4. 自动扫描源文件 (排除 test 文件夹)
# ==========================================

# 1. 扫描 data_type, operations 以及当前目录下的所有 .cpp
file(GLOB_RECURSE LIBRARY_SOURCES 
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/data_type/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/operations/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" # 比如 config.cpp 如果有的话
)

# 2. 【关键】从列表中剔除 test 文件夹下的文件
# 这样库文件里就不会包含 main 函数，避免冲突
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
# 默认值为 "ALL"，你可以通过命令行 -DTEST_SELECT=... 来修改
# 选项: 
#   "ALL"  : 编译 test/ 下所有子文件夹的所有 .cpp
#   "DEV"  : 只编译 test/dev_test/ 下的 .cpp
#   "FUNC" : 只编译 test/functional_test/ 下的 .cpp
#   "文件名": 编译指定的文件 (例如 "test_operators" 或 "tmp_file.cpp")
set(TEST_SELECT "ALL" CACHE STRING "Select tests to build: ALL, DEV, FUNC, or specific filename")

message(STATUS "------------------------------------------------")
message(STATUS "Build Test Mode: ${TEST_SELECT}")

# 2. 根据 TEST_SELECT 查找源文件
set(TEST_SOURCES "")

if(TEST_SELECT STREQUAL "ALL")
    # 递归查找 test 目录下所有的 .cpp 文件
    file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")

elseif(TEST_SELECT STREQUAL "DEV")
    # 只查找 dev_test 目录
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/dev_test/*.cpp")

elseif(TEST_SELECT STREQUAL "FUNC")
    # 只查找 functional_test 目录
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/functional_test/*.cpp")

elseif(TEST_SELECT STREQUAL "LOCALMODEL")
    # 只查找 local_model_test 目录
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/local_model_test/*.cpp")

else()
    # 假设用户输入的是特定的文件名 (例如 "test_operators")
    # 我们在 test 目录下递归搜索匹配该名字的文件
    # 加上 * 使得输入 "test_op" 也能匹配 "test_operators.cpp"
    file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*${TEST_SELECT}_test*.cpp")
endif()

# 3. 遍历找到的源文件，为每一个文件生成一个独立的可执行程序
if(TEST_SOURCES)
    foreach(test_src ${TEST_SOURCES})
        # 获取文件名 (不带路径和后缀)，用作 Target 名字
        # 例如: .../test/dev_test/test_operators.cpp -> test_operators
        get_filename_component(test_target_name ${test_src} NAME_WE)

        message(STATUS "  [+] Adding test target: ${test_target_name}")

        # 生成可执行文件
        add_executable(${test_target_name} ${test_src})

        # 链接核心库 (mvzk_core)
        # 如果你有其他的库依赖 (比如 OpenMP, SSL)，确保 mvzk_core 已经 PUBLIC 链接了它们，或者在这里添加
        target_link_libraries(${test_target_name} PRIVATE mvzk_core)
        
    endforeach()
else()
    message(WARNING "  [-] No test source files found matching: ${TEST_SELECT}")
endif()

message(STATUS "------------------------------------------------")