class DiffusionCalculator:
    def __init__(self, token_stats, text_stats, sampling_stats, generation_stats):
        self.token_stats = token_stats
        self.text_stats = text_stats
        self.sampling_stats = sampling_stats
        self.generation_stats = generation_stats

    def _avg(self, key):
        values = [
            self.token_stats.get(key, 0),
            self.text_stats.get(key, 0),
            self.sampling_stats.get(key, 0),
            self.generation_stats.get(key, 0)
        ]
        return sum(values) / len(values)

    def calculate_diffusion_stats(self):
        # Estimate time as difference between sampling end and generation start
        diffusion_time = self.generation_stats.get('start_time', 0) - self.sampling_stats.get('end_time', 0)

        return {
            "diffusion_time_sec": round(diffusion_time, 4),
            "diffusion_cpu_util_percent": round(self._avg("avg_cpu_util_percent"), 2),
            "diffusion_gpu_util_percent": round(self._avg("avg_gpu_util_percent"), 2),
            "diffusion_ram_usage_mb": round(self._avg("avg_ram_memory_mb"), 2),
            "diffusion_gpu_memory_mb": round(self._avg("avg_gpu_memory_mb"), 2),
            "diffusion_network_used_mb": round(self._avg("network_used_mb"), 2),
        }
