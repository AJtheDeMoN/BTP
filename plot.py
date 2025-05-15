import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("output.csv")

# Define the stats you want to plot
stats = [
    "Model_CPU_util",
    "Model_GPU_util",
    "Model_RAM_MB",
    "Model_GPU_Mem_MB",
    "Model_Network_MB",
    "Model_time_sec"
]

# Unique prompts and models
prompts = df["Prompt"].unique()
models = df["Model"].unique()

# Loop through each stat to generate a separate line graph
for stat in stats:
    plt.figure(figsize=(12, 6))
    
    for model in models:
        # Filter data for this model
        model_df = df[df["Model"] == model]
        
        # Sort by prompt to keep X-axis consistent
        model_df = model_df.set_index("Prompt").loc[prompts].reset_index()
        
        plt.plot(model_df["Prompt"], model_df[stat], marker="o", label=model)
    
    plt.title(f"{stat.replace('_', ' ')} across Prompts")
    plt.xlabel("Prompt")
    plt.ylabel(stat.replace('_', ' '))
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Model")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{stat}_line_graph.png")
    plt.show()
