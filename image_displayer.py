import matplotlib.pyplot as plt
import time
import os
import psutil
import torch
import threading
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetHandleByIndex
)

class ImageDisplayer:
    def __init__(self, image, model_name, prompt):
        self.image = image
        self.model_name = model_name.replace('/', '_')
        self.prompt = prompt

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
                self.cpu_samples.append(self.process.cpu_percent(interval=interval))
                self.ram_samples.append(self.process.memory_info().rss)

                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated(device="cuda:1")
                    self.gpu_memory_samples.append(mem)
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

    def display_image(self):
        self.monitor_system_usage()
        start_time = time.time()

        # Create directory if it doesn't exist
        save_dir = os.path.join("photos", self.model_name)
        os.makedirs(save_dir, exist_ok=True)

        # Sanitize filename
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in self.prompt).strip()
        filename = os.path.join(save_dir, f"{safe_prompt}.png")

        # Display and save
        fig, ax = plt.subplots(figsize=(self.image.width / 100, self.image.height / 100))
        ax.imshow(self.image)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(filename, bbox_inches=None, pad_inches=0, dpi=100)
        plt.close(fig)

        display_time = time.time() - start_time
        self.stop_monitoring()

        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        avg_gpu_mem = sum(self.gpu_memory_samples) / len(self.gpu_memory_samples) if self.gpu_memory_samples else 0
        avg_gpu_util = sum(self.gpu_util_samples) / len(self.gpu_util_samples) if self.gpu_util_samples else 0
        avg_ram = sum(self.ram_samples) / len(self.ram_samples) if self.ram_samples else 0

        num_cores = psutil.cpu_count(logical=True)
        normalized_cpu = avg_cpu / num_cores if num_cores else 0

        net_used_bytes = (
            (self.net_end.bytes_recv - self.net_start.bytes_recv) +
            (self.net_end.bytes_sent - self.net_start.bytes_sent)
        )
        net_used_mb = net_used_bytes / (1024 * 1024)

        stats = {
            'image_display_time_sec': round(display_time, 4),
            'avg_cpu_util_percent': round(normalized_cpu, 2),
            'avg_gpu_util_percent': round(avg_gpu_util, 2),
            'avg_ram_memory_mb': round(avg_ram / (1024 ** 2), 2),
            'avg_gpu_memory_mb': round(avg_gpu_mem / (1024 ** 2), 2),
            'network_used_mb': round(net_used_mb, 2)
        }

        return stats
