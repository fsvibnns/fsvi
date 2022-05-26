from typing import NamedTuple

import nvidia_smi


def get_gpu_memory_usage() -> NamedTuple:
    """
    @return: a namedtuple that has three fields used, total, free
    """
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    nvidia_smi.nvmlShutdown()
    return info


def log_gpu_usage(desc: str = ""):
    info = get_gpu_memory_usage()
    used_gb = info.used / (1024 ** 3)
    total_gb = info.total / (1024 ** 3)
    perc = used_gb / total_gb * 100
    print(f"{desc}, {used_gb:.2f}/{total_gb:.2f}GB ({perc:.2f}%) GPU memory used")
