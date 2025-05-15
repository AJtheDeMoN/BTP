import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("output.csv")

# Get unique models and sort prompts
models = df["Model"].unique()
prompts = df["Prompt"].unique()

# Assign numbers 1–9 for prompt indices
prompt_numbers = list(range(1, len(prompts) + 1))
prompt_map = dict(zip(prompts, prompt_numbers))
df["Prompt_Number"] = df["Prompt"].map(prompt_map)

# Plot
plt.figure(figsize=(10, 6))
for model in models:
    model_df = df[df["Model"] == model].sort_values("Prompt_Number")
    plt.plot(
        model_df["Prompt_Number"],
        model_df["Model_time_sec"],
        marker="o",
        label=model
    )

plt.title("Generation Time across Prompts")
plt.xlabel("Prompt Number")
plt.ylabel("Generation Time (sec)")
plt.xticks(prompt_numbers)
plt.legend(title="Model")
plt.grid(True)
plt.tight_layout()
plt.savefig("generation_time_cleaned.png")
plt.show()
