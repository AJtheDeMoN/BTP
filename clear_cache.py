import shutil
import os

def clear_cache():
    # Specify the directory you want to delete
    dir_path = "/tmp/stable_diffusion_cache"

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Directory '{dir_path}' has been deleted.")
    else:
        print(f"Directory '{dir_path}' does not exist.")
        