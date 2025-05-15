import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Timing Program")
    parser.add_argument('--model', type=str, required=True, help='Model name for the pipeline')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image generation')
    return parser.parse_args()
