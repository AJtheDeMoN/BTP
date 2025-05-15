from diffusers import AutoPipelineForText2Image
import time
import torch
import psutil
import threading
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetHandleByIndex
)

class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipeline = None

    def monitor_system_usage(self, interval=0.1):
        self.cpu_samples = []
        self.gpu_memory_samples = []
        self.gpu_util_samples = []
        self.ram_samples = []

        self.net_start = psutil.net_io_counters()
        self.process = psutil.Process()
        self.done = False

        if torch.cuda.is_available():
            nvmlInit()
            self.nvml_handle = nvmlDeviceGetHandleByIndex(0)

        def sample():
            while not self.done:
                # CPU usage for current process
                self.cpu_samples.append(self.process.cpu_percent(interval=interval))
                # RAM memory
                self.ram_samples.append(self.process.memory_info().rss)

                if torch.cuda.is_available():
                    # GPU memory used by this process
                    mem = torch.cuda.memory_allocated(device="cuda:1")
                    self.gpu_memory_samples.append(mem)

                    # GPU utilization
                    util = nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    self.gpu_util_samples.append(util.gpu)

        self.monitor_thread = threading.Thread(target=sample)
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.done = True
        self.monitor_thread.join()
        self.net_end = psutil.net_io_counters()
        if torch.cuda.is_available():
            nvmlShutdown()

    def load_model(self):
        # Start monitoring before loading the model
        self.monitor_system_usage()
        
        # Record start time right before loading the model
        start_time = time.time()

        try:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            ).to("cuda:1")
        except Exception as e:
            print(f"Error loading model: {e}")
            return {}, None
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.4f} seconds")
        self.stop_monitoring()

        # Averages
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        avg_gpu_mem = sum(self.gpu_memory_samples) / len(self.gpu_memory_samples) if self.gpu_memory_samples else 0
        avg_gpu_util = sum(self.gpu_util_samples) / len(self.gpu_util_samples) if self.gpu_util_samples else 0
        avg_ram = sum(self.ram_samples) / len(self.ram_samples) if self.ram_samples else 0

        # Normalize CPU over all cores
        num_cores = psutil.cpu_count(logical=True)
        normalized_cpu = avg_cpu / num_cores if num_cores else 0

        # Network usage in MB
        net_used_bytes = (
            (self.net_end.bytes_recv - self.net_start.bytes_recv) +
            (self.net_end.bytes_sent - self.net_start.bytes_sent)
        )
        net_used_mb = net_used_bytes / (1024 * 1024)

        stats = {
            'load_model_time_sec': round(load_time, 4),
            'avg_cpu_util_percent': round(normalized_cpu, 2),
            'avg_gpu_util_percent': round(avg_gpu_util, 2),
            'avg_ram_memory_mb': round(avg_ram / (1024 ** 2), 2),
            'avg_gpu_memory_mb': round(avg_gpu_mem / (1024 ** 2), 2),
            'network_used_mb': round(net_used_mb, 2)
        }

        return stats, self.pipeline
