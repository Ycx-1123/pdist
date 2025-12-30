import os
import subprocess
import re
import sys

# =========================================================
# 配置区域
# =========================================================
BINARY_PATH = "./build/main"  # C++ 可执行文件路径
TIMEOUT_SEC = 300             # 每个用例的超时时间 (秒)

# 测试用例定义: (N, M, P, DType_Enum)
# DType: 0=FP32, 1=FP16
TEST_CASES = [
    # --- 基础功能测试 ---
    {"name": "Case01_Base",    "args": [1024, 128, 2.0, 0]}, # FP32, P=2
    {"name": "Case02_FP16",    "args": [1024, 128, 2.0, 1]}, # FP16, P=2
    {"name": "Case03_Manhat",  "args": [1024, 128, 1.0, 0]}, # FP32, P=1 (曼哈顿)
    {"name": "Case04_Inf",     "args": [1024, 128, float('inf'), 0]}, # FP32, P=inf (切比雪夫)
    
    # --- 特殊 P 值测试 ---
    {"name": "Case05_P_3.0",   "args": [512, 128, 3.0, 0]},  # FP32, P=3 (通用p值)
    {"name": "Case06_P_0.5",   "args": [512, 128, 0.5, 0]},  # FP32, P=0.5
    
    # --- 特殊 Shape (奇数/对齐测试) ---
    {"name": "Case07_Odd_M",   "args": [128, 33, 2.0, 0]},   # M=33 (非32对齐)
    {"name": "Case08_Odd_N",   "args": [33, 128, 2.0, 0]},   # N=33 (小N)
    {"name": "Case09_Small",   "args": [16, 16, 2.0, 0]},    # 极小数据
    
    # --- 大规模压力测试 (Performance) ---
    {"name": "Case10_Tall",    "args": [4096, 32, 2.0, 0]},  # 瘦高矩阵
    {"name": "Case11_Wide",    "args": [256, 4096, 2.0, 0]}, # 矮胖矩阵
    {"name": "Case12_Large",   "args": [2048, 3008, 2.0, 1]} # 大规模 FP16 (重点跑分项)
]

def compile_cpp():
    """自动编译 C++ 代码"""
    print(">>> [Setup] Compiling TestPdist/main.cpp ...")
    if not os.path.exists("build"):
        os.makedirs("build")
    
    # 简单的 cmake 构建流程
    cmd = "cd build && cmake .. && make -j4"
    ret = os.system(cmd)
    if ret != 0:
        print("❌ Compilation Failed! Please check CMakeLists.txt and main.cpp")
        sys.exit(1)
    
    if not os.path.exists(BINARY_PATH):
        print(f"❌ Binary not found at {BINARY_PATH}")
        sys.exit(1)
    print("✅ Compilation Success!\n")

def run_single_case(case_info):
    """运行单个测试用例并解析输出"""
    name = case_info['name']
    args = [str(a) for a in case_info['args']]
    cmd = [BINARY_PATH] + args
    
    print(f"Running {name} args={args} ... ", end='', flush=True)
    
    try:
        # 运行 C++ 程序并捕获输出
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=TIMEOUT_SEC
        )
        output = result.stdout
        
        # 1. 检查是否 PASS
        is_pass = "PASS" in output
        status = "\033[32mPASS\033[0m" if is_pass else "\033[31mFAIL\033[0m"
        
        # 2. 提取 CPU 时间 (Regex)
        cpu_match = re.search(r"CPU Time:\s+([0-9\.]+)\s+ms", output)
        cpu_time = float(cpu_match.group(1)) if cpu_match else 0.0
        
        # 3. 提取 NPU 时间 (Regex)
        npu_match = re.search(r"NPU Time:\s+([0-9\.]+)\s+ms", output)
        npu_time = float(npu_match.group(1)) if npu_match else 0.0
        
        # 4. 计算加速比
        speedup = 0.0
        if npu_time > 0 and cpu_time > 0:
            speedup = cpu_time / npu_time
            
        print(f"{status}")
        return {
            "name": name,
            "shape": f"{args[0]}x{args[1]}",
            "cpu_ms": cpu_time,
            "npu_ms": npu_time,
            "speedup": speedup,
            "pass": is_pass
        }

    except subprocess.TimeoutExpired:
        print("\033[31mTIMEOUT\033[0m")
        return None
    except Exception as e:
        print(f"\033[31mERROR: {e}\033[0m")
        return None

def print_summary(results):
    """打印最终汇总表格"""
    print("\n" + "="*85)
    print(f"{'Case Name':<15} | {'Shape':<12} | {'CPU (ms)':>10} | {'NPU (ms)':>10} | {'Speedup':>8} | {'Result':<6}")
    print("-" * 85)
    
    for res in results:
        if res is None:
            continue
        
        # 格式化输出
        cpu_str = f"{res['cpu_ms']:.2f}" if res['cpu_ms'] > 0 else "N/A"
        npu_str = f"{res['npu_ms']:.3f}"
        speed_str = f"{res['speedup']:.1f}x" if res['speedup'] > 0 else "-"
        status = "PASS" if res['pass'] else "FAIL"
        
        # 给加速比加点颜色 (高亮显示 >50x 的)
        if res['speedup'] > 50:
            speed_str = f"\033[1;33m{speed_str}\033[0m"
            
        print(f"{res['name']:<15} | {res['shape']:<12} | {cpu_str:>10} | {npu_str:>10} | {speed_str:>18} | {status:<6}")
    print("="*85 + "\n")

if __name__ == "__main__":
    # 1. 编译
    compile_cpp()
    
    # 2. 跑测试
    all_results = []
    for case in TEST_CASES:
        res = run_single_case(case)
        all_results.append(res)
        
    # 3. 打印汇总
    print_summary(all_results)