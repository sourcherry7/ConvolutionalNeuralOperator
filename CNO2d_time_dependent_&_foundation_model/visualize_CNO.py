import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import the existing tools from the repo
from test_and_fine_tune_utils.test_utils import _load_dict, _initialize_model, _load_data

# --- 1. CONFIGURATION (ABSOLUTE PATHS) ---
# Based on your screenshot, your files are directly in 'results/checkpoints'
PROJECT_ROOT = "/cluster/home/svisnjic/Thalassa"
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "results/checkpoints")

# The exact files seen in your screenshot
TRAIN_FILE = os.path.join(CHECKPOINTS_DIR, "training_properties.txt")
NET_FILE = os.path.join(CHECKPOINTS_DIR, "net_architecture.txt")
CKPT_PATH = os.path.join(CHECKPOINTS_DIR, "model_0601", "epoch=33-step=9248.ckpt")

WHICH_EXAMPLE = "wave_gauss"

def visualize():
    print(f"--- Configuration ---")
    print(f"Loading Properties: {TRAIN_FILE}")
    print(f"Loading Checkpoint: {CKPT_PATH}")

    # 1. LOAD CONFIGURATION
    # We reuse the repo's logic to parse the text files
    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: Could not find {TRAIN_FILE}")
        return

    loader_dict = _load_dict(files=[TRAIN_FILE, NET_FILE], 
                             which_example=WHICH_EXAMPLE, 
                             steps=15) # Wave Gauss uses 15 steps

    if "exp" not in loader_dict:
        loader_dict["exp"] = 1
    
    # 2. LOAD ONE SAMPLE OF DATA
    # This automatically finds the .nc file using the paths in training_properties.txt
    print("Loading test data...")
    loader = _load_data(which=WHICH_EXAMPLE, 
                        loader_dict=loader_dict, 
                        batch_size=1, 
                        which_loader="test",
                        in_dist=True)

    # 3. INITIALIZE MODEL
    print("Initializing model architecture...")
    model = _initialize_model(loader_dict)
    
    # 4. LOAD WEIGHTS
    print("Loading weights...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        checkpoint = torch.load(CKPT_PATH, map_location=device)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {CKPT_PATH}")
        return

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # 5. RUN INFERENCE
    print("Running Inference...")
    # Get first batch
    batch = next(iter(loader))
    
    # CNO Data Loader returns: [Time, Input, Output]
    # We need to move them to GPU
    time_batch = batch[0].to(device)   # The time coordinate
    input_batch = batch[1].to(device)  # The input wave (t=0)
    target_batch = batch[2].to(device) # The target wave (t=15)

    with torch.no_grad():
        # The model predicts the target state based on input + time
        prediction = model(input_batch, time_batch)

    # 6. PLOT RESULTS
    print("Plotting...")
    
    # Extract the first sample (index 0) and first channel (index 0) for 2D plotting
    # Convert to CPU numpy arrays
    input_img = input_batch[0, 0].cpu().numpy()
    target_img = target_batch[0, 0].cpu().numpy()
    pred_img = prediction[0, 0].cpu().numpy()
    
    # Compute Absolute Error
    abs_error = np.abs(target_img - pred_img)

    # Create Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    titles = ["Input (t=0)", "Ground Truth (t=15)", "CNO Prediction", "Absolute Error"]
    images = [input_img, target_img, pred_img, abs_error]
    cmaps = ['viridis', 'viridis', 'viridis', 'magma'] # Magma makes errors pop
    
    for i, ax in enumerate(axes):
        im = ax.imshow(images[i], cmap=cmaps[i], origin='lower')
        ax.set_title(titles[i], fontsize=14)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    output_filename = "wave_gauss_results.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nSUCCESS! Visualization saved to: {os.path.abspath(output_filename)}")
    print("Open this file in VS Code to see your results.")

if __name__ == "__main__":
    visualize()