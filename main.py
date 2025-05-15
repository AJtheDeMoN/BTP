import warnings
import os
import csv
# Suppress the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

import time
from arg_parser import get_args
from clear_cache import clear_cache
from model_loader import ModelLoader
from tokenization import Tokenization
from text_encoder import TextEncoder
from latent_space_sampler import LatentSpaceSampler
from image_generator import ImageGenerator
from diffusion import DiffusionCalculator
from image_displayer import ImageDisplayer

def main():
    # clear_cache()
    args = get_args()

    start_t = time.time()

    # Step 1: Model Loading
    loader = ModelLoader(args.model)
    load_stats, pipeline = loader.load_model()

    # Step 2: Tokenization
    tokenizer = Tokenization(pipeline, args.prompt)
    inputs, token_stats = tokenizer.tokenize()

    # Step 3: Text Encoding
    encoder = TextEncoder(pipeline, inputs)
    text_embeddings, text_stats = encoder.encode_text()

    # Step 4: Latent Space Sampling
    sampler = LatentSpaceSampler(pipeline)
    latents, sampling_stats = sampler.sample_latents()

    # Step 5: Image Generation
    generator = ImageGenerator(pipeline, args.prompt)
    image, generation_stats = generator.generate_image()

    # Step 6: Image Display
    displayer = ImageDisplayer(image, args.model, args.prompt)
    display_stats = displayer.display_image()

    # Step 7: Diffusion Calculation
    diffusion_calculator = DiffusionCalculator(
        token_stats=token_stats,
        text_stats=text_stats,
        sampling_stats=sampling_stats,
        generation_stats=generation_stats
    )
    diffusion_stats = diffusion_calculator.calculate_diffusion_stats()

    total_time = time.time() - start_t

    
    # Output to CSV with full system stats

    csv_file = "output.csv"
    headers = [
        "Model", "Prompt", "Total_Generation_Time",

        "Model_time_sec", "Model_CPU_util", "Model_GPU_util", "Model_RAM_MB", "Model_GPU_Mem_MB", "Model_Network_MB",
        "Token_time_sec", "Token_CPU_util", "Token_GPU_util", "Token_RAM_MB", "Token_GPU_Mem_MB", "Token_Network_MB",
        "Encoding_time_sec", "Encoding_CPU_util", "Encoding_GPU_util", "Encoding_RAM_MB", "Encoding_GPU_Mem_MB", "Encoding_Network_MB",
        "Latent_time_sec", "Latent_CPU_util", "Latent_GPU_util", "Latent_RAM_MB", "Latent_GPU_Mem_MB", "Latent_Network_MB",
        "Diffusion_time_sec", "Diffusion_CPU_util", "Diffusion_GPU_util", "Diffusion_RAM_MB", "Diffusion_GPU_Mem_MB", "Diffusion_Network_MB",
        "Generation_time_sec", "Generation_CPU_util", "Generation_GPU_util", "Generation_RAM_MB", "Generation_GPU_Mem_MB", "Generation_Network_MB",
        "Display_time_sec", "Display_CPU_util", "Display_GPU_util", "Display_RAM_MB", "Display_GPU_Mem_MB", "Display_Network_MB"
    ]

    new_row = {
        "Model": args.model,
        "Prompt": args.prompt,
        "Total_Generation_Time": f"{total_time:.4f}",

        "Model_time_sec": f"{load_stats.get('load_model_time_sec', 0):.4f}",
        "Model_CPU_util": f"{load_stats.get('avg_cpu_util_percent', 0):.2f}",
        "Model_GPU_util": f"{load_stats.get('avg_gpu_util_percent', 0):.2f}",
        "Model_RAM_MB": f"{load_stats.get('avg_ram_memory_mb', 0):.2f}",
        "Model_GPU_Mem_MB": f"{load_stats.get('avg_gpu_memory_mb', 0):.2f}",
        "Model_Network_MB": f"{load_stats.get('network_used_mb', 0):.2f}",

        "Token_time_sec": f"{token_stats.get('tokenization_time_sec', 0):.4f}",
        "Token_CPU_util": f"{token_stats.get('avg_cpu_util_percent', 0):.2f}",
        "Token_GPU_util": f"{token_stats.get('avg_gpu_util_percent', 0):.2f}",
        "Token_RAM_MB": f"{token_stats.get('avg_ram_memory_mb', 0):.2f}",
        "Token_GPU_Mem_MB": f"{token_stats.get('avg_gpu_memory_mb', 0):.2f}",
        "Token_Network_MB": f"{token_stats.get('network_used_mb', 0):.2f}",

        "Encoding_time_sec": f"{text_stats.get('text_encoding_time_sec', 0):.4f}",
        "Encoding_CPU_util": f"{text_stats.get('avg_cpu_util_percent', 0):.2f}",
        "Encoding_GPU_util": f"{text_stats.get('avg_gpu_util_percent', 0):.2f}",
        "Encoding_RAM_MB": f"{text_stats.get('avg_ram_memory_mb', 0):.2f}",
        "Encoding_GPU_Mem_MB": f"{text_stats.get('avg_gpu_memory_mb', 0):.2f}",
        "Encoding_Network_MB": f"{text_stats.get('network_used_mb', 0):.2f}",

        "Latent_time_sec": f"{sampling_stats.get('latent_sampling_time_sec', 0):.4f}",
        "Latent_CPU_util": f"{sampling_stats.get('avg_cpu_util_percent', 0):.2f}",
        "Latent_GPU_util": f"{sampling_stats.get('avg_gpu_util_percent', 0):.2f}",
        "Latent_RAM_MB": f"{sampling_stats.get('avg_ram_memory_mb', 0):.2f}",
        "Latent_GPU_Mem_MB": f"{sampling_stats.get('avg_gpu_memory_mb', 0):.2f}",
        "Latent_Network_MB": f"{sampling_stats.get('network_used_mb', 0):.2f}",

        "Diffusion_time_sec": f"{diffusion_stats.get('diffusion_time_sec', 0):.4f}",
        "Diffusion_CPU_util": f"{diffusion_stats.get('diffusion_cpu_util_percent', 0):.2f}",
        "Diffusion_GPU_util": f"{diffusion_stats.get('diffusion_gpu_util_percent', 0):.2f}",
        "Diffusion_RAM_MB": f"{diffusion_stats.get('diffusion_ram_usage_mb', 0):.2f}",
        "Diffusion_GPU_Mem_MB": f"{diffusion_stats.get('diffusion_gpu_memory_mb', 0):.2f}",
        "Diffusion_Network_MB": f"{diffusion_stats.get('diffusion_network_used_mb', 0):.2f}",

        "Generation_time_sec": f"{generation_stats.get('image_generation_time_sec', 0):.4f}",
        "Generation_CPU_util": f"{generation_stats.get('avg_cpu_util_percent', 0):.2f}",
        "Generation_GPU_util": f"{generation_stats.get('avg_gpu_util_percent', 0):.2f}",
        "Generation_RAM_MB": f"{generation_stats.get('avg_ram_memory_mb', 0):.2f}",
        "Generation_GPU_Mem_MB": f"{generation_stats.get('avg_gpu_memory_mb', 0):.2f}",
        "Generation_Network_MB": f"{generation_stats.get('network_used_mb', 0):.2f}",

        "Display_time_sec": f"{display_stats.get('image_display_time_sec', 0):.4f}",
        "Display_CPU_util": f"{display_stats.get('avg_cpu_util_percent', 0):.2f}",
        "Display_GPU_util": f"{display_stats.get('avg_gpu_util_percent', 0):.2f}",
        "Display_RAM_MB": f"{display_stats.get('avg_ram_memory_mb', 0):.2f}",
        "Display_GPU_Mem_MB": f"{display_stats.get('avg_gpu_memory_mb', 0):.2f}",
        "Display_Network_MB": f"{display_stats.get('network_used_mb', 0):.2f}"
    }

    try:
        rows = []
        found = False

        if os.path.exists(csv_file):
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row["Model"] == new_row["Model"] and row["Prompt"] == new_row["Prompt"]:
                        print("Same model and prompt found. Updating row.")
                        row = new_row
                        found = True
                    rows.append(row)

        if not found:
            rows.append(new_row)

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n✅ Data written to {csv_file}.")

    except Exception as e:
        print(f"\n Error writing to CSV: {e}")


if __name__ == "__main__":
    main()
