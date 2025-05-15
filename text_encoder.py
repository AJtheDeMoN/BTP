import time
import psutil
import torch
import threading
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetHandleByIndex
)

class TextEncoder:
    def __init__(self, pipeline, inputs):
        self.pipeline = pipeline
        self.inputs = inputs

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
                # CPU
                self.cpu_samples.append(self.process.cpu_percent(interval=interval))
                # RAM
                self.ram_samples.append(self.process.memory_info().rss)

                if torch.cuda.is_available():
                    # GPU mem used by process
                    mem = torch.cuda.memory_allocated(device="cuda:1")
                    self.gpu_memory_samples.append(mem)
                    # GPU util
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

    def encode_text(self):
        self.monitor_system_usage()
        start_time = time.time()

        text_embeddings = self.pipeline.text_encoder(self.inputs.input_ids.to("cuda:1"))

        encoding_time = time.time() - start_time
        self.stop_monitoring()

        # Averages
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        avg_gpu_mem = sum(self.gpu_memory_samples) / len(self.gpu_memory_samples) if self.gpu_memory_samples else 0
        avg_gpu_util = sum(self.gpu_util_samples) / len(self.gpu_util_samples) if self.gpu_util_samples else 0
        avg_ram = sum(self.ram_samples) / len(self.ram_samples) if self.ram_samples else 0

        # Normalize CPU
        num_cores = psutil.cpu_count(logical=True)
        normalized_cpu = avg_cpu / num_cores if num_cores else 0

        # Network in MB
        net_used_bytes = (
            (self.net_end.bytes_recv - self.net_start.bytes_recv) +
            (self.net_end.bytes_sent - self.net_start.bytes_sent)
        )
        net_used_mb = net_used_bytes / (1024 * 1024)

        stats = {
            'text_encoding_time_sec': round(encoding_time, 4),
            'avg_cpu_util_percent': round(normalized_cpu, 2),
            'avg_gpu_util_percent': round(avg_gpu_util, 2),
            'avg_ram_memory_mb': round(avg_ram / (1024 ** 2), 2),
            'avg_gpu_memory_mb': round(avg_gpu_mem / (1024 ** 2), 2),
            'network_used_mb': round(net_used_mb, 2)
        }

        return text_embeddings, stats
