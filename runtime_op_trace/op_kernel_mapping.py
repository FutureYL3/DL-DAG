import json
import sys
import os
from collections import defaultdict

# 获取命令行参数指定的目录，默认为当前目录
trace_dir = sys.argv[1] if len(sys.argv) > 1 else "."
trace_path = os.path.join(os.path.abspath(trace_dir), "trace.json")
output_path = os.path.join(os.path.abspath(trace_dir), "op_kernel_mapping.json")

print(f"Reading trace from: {trace_path}")

with open(trace_path, "r") as f:
    data = json.load(f)

events = data["traceEvents"]

launchAPIs = [
    "cudaLaunchKernel",
    "cudaLaunchCooperativeKernel",
    "cudaLaunchCooperativeKernelMultiDevice",
    "cudaLaunchDevice",
    "cudaLaunchKernelExC",
    "cudaLaunchKernelEx",
    "cudaGraphLaunch",
]

# 1. 按类别分类
cpu_ops = []
launches = []
kernels = []

for e in events:
    cat = e.get("cat")
    if cat == "cpu_op" and e.get("ph") == "X" and e.get("name", "").startswith("aten::"):
        cpu_ops.append(e)
    elif cat == "cuda_runtime" and e.get("name") in launchAPIs and e.get("ph") == "X":
        launches.append(e)
    elif cat == "kernel" and e.get("ph") == "X":
        kernels.append(e)

# 2. 先把 kernel 按 correlation 建索引
kernels_by_corr = defaultdict(list)
for k in kernels:
    corr = k.get("args", {}).get("correlation")
    if corr is not None:
        kernels_by_corr[corr].append(k)

# 3. 把 CPU op 按 pid 分组，并按时间排序，方便后面查包含关系
cpu_ops_by_pid = defaultdict(list)
for op in cpu_ops:
    cpu_ops_by_pid[op["pid"]].append(op)

for pid in cpu_ops_by_pid:
    cpu_ops_by_pid[pid].sort(key=lambda e: e["ts"])

def find_inner_cpu_op(pid, ts_launch):
    """
    在给定 pid 中，找到时间上包住 ts_launch 的所有内层 CPU op
    """
    candidates = [
        op for op in cpu_ops_by_pid.get(pid, [])
        if op["ts"] <= ts_launch <= op["ts"] + op["dur"]
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda e: e["dur"], reverse=True)
    return candidates

# 4. 构造 operator <-> kernel 的对应关系
op_to_kernels = defaultdict(list)

for launch in launches:
    corr = launch.get("args", {}).get("correlation")
    if corr is None:
        continue

    pid = launch["pid"]
    ts_launch = launch["ts"]

    ops = find_inner_cpu_op(pid, ts_launch)
    if ops is None:
        continue

    op_name = ops[0]["name"]
    for k in kernels_by_corr.get(corr, []):
        op_to_kernels[op_name].append({
            "kernel_name": k["name"],
            "kernel_ts": k["ts"],
            "kernel_dur": k["dur"],
            "device": k["args"].get("device"),
            "stream": k["args"].get("stream"),
        })

# 5. 输出为一个精简版 JSON，只保留算子 <-> kernel 对应关系
mapping = []
for op_name, ks in op_to_kernels.items():
    mapping.append({
        "op_name": op_name,
        "num_kernels": len(ks),
        "kernels": ks,
    })

with open(output_path, "w") as f:
    json.dump(mapping, f, indent=2)

print(f"wrote {len(mapping)} operators to {output_path}")
