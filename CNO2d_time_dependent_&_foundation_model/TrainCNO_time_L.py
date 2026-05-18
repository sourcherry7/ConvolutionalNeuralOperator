from pytorch_lightning import Trainer
from CNO_timeModule_CIN import CNO_time
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import copy
import json
import os
import sys
import shutil

import pandas as pd
import torch

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback

class RolloutVisualCallback(Callback):
    def __init__(self, save_dir, frequency=10):
        super().__init__()
        self.save_dir = save_dir
        self.frequency = frequency
        os.makedirs(save_dir, exist_ok=True)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.frequency == 0 and trainer.current_epoch > 0:
            pl_module.eval()
            try:
                # 1. Get a raw trajectory from the validation dataset
                # We need the full sequence, not just random pairs
                val_dataset = trainer.val_dataloaders[0].dataset
                
                # We use the dataset's internal reader to get a clean 16-frame sequence
                # Assuming WaveGaussians structure: [u] is [N, 16, 128, 128]
                # We'll take the first trajectory in the validation set
                idx = val_dataset.start 
                u_gt_phys = val_dataset.reader['u'][idx] # [16, 128, 128]
                c_phys = val_dataset.reader['c'][idx]    # [128, 128]
                
                # 2. Setup for Neural Rollout
                # We need to normalize using the dataset constants
                stats = val_dataset.constants
                device = pl_module.device
                
                # Initial wavefield normalized
                u_curr = (torch.from_numpy(u_gt_phys[0]).float().to(device) - stats["mean"]) / stats["std"]
                u_curr = u_curr.view(1, 1, 128, 128)
                
                # Velocity field normalized (static)
                c_norm = (torch.from_numpy(c_phys).float().to(device) - stats["mean_c"]) / stats["std_c"]
                c_norm = c_norm.view(1, 1, 128, 128)
                
                # Time step scalar (dt = 1/15)
                dt_val = 1.0 / stats["time"]
                t_scalar = torch.tensor([[dt_val]], device=device)
                
                # If your model expects the time-channel as a 3rd input
                if val_dataset.time_input:
                    t_channel = torch.ones_like(c_norm) * dt_val

                # 3. Perform Autoregressive Rollout
                preds_phys = [u_gt_phys[0]] # Store in physical units
                
                for _ in range(1, 16):
                    # Prepare Input: cat([u, c, dt]) or cat([u, c])
                    if val_dataset.time_input:
                        x_in = torch.cat([u_curr, c_norm, t_channel], dim=1)
                    else:
                        x_in = torch.cat([u_curr, c_norm], dim=1)
                        
                    # Predict next step
                    out = pl_module(x_in, t_scalar)
                    u_next_norm = out[:, 0:1] # Extract wavefield channel
                    
                    # Store denormalized result
                    u_phys = (u_next_norm.squeeze().cpu().numpy() * stats["std"]) + stats["mean"]
                    preds_phys.append(u_phys)
                    
                    # Update current for next iteration
                    u_curr = u_next_norm

                preds_phys = np.array(preds_phys)

                # 4. Plotting (Your side-by-side logic)
                u_min = min(u_gt_phys.min(), preds_phys.min())
                u_max = max(u_gt_phys.max(), preds_phys.max())
                
                fig, axes = plt.subplots(8, 4, figsize=(20, 32))
                for i in range(16):
                    # GT (Rows 0-3)
                    ax_gt = axes[i // 4, i % 4]
                    im_gt = ax_gt.imshow(u_gt_phys[i], vmin=u_min, vmax=u_max, cmap='jet')
                    ax_gt.set_title(f"GT u(t={i})")
                    ax_gt.axis('off')
                    
                    # Prediction (Rows 4-7)
                    ax_pred = axes[(i // 4) + 4, i % 4]
                    im_pred = ax_pred.imshow(preds_phys[i], vmin=u_min, vmax=u_max, cmap='jet')
                    ax_pred.set_title(f"CNO u(t={i})")
                    ax_pred.axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(self.save_dir, f"rollout_epoch_{trainer.current_epoch}.pdf")
                plt.savefig(save_path)
                plt.close()
                print(f"\n[Visuals] Saved 16-frame rollout comparison to {save_path}")

            except Exception as e:
                print(f"\n[Visuals Warning] Rollout plotting failed: {e}")

if len(sys.argv) <= 2:
    
    training_properties = {
        "learning_rate": 0.0003, 
        "weight_decay": 1e-6,
        "scheduler_step": 3,
        "scheduler_gamma": 0.7,
        "epochs": 40,
        "batch_size": 32,         
        "time_steps": 15,         # How many time steps to select?
        "dt": 1,                  # What is the time step? (1 means include entire traj, 2 means taking every other step, etc.
        "training_samples": 2048, # How many training samples?
        "time_input": 1,          # Should we include time in the input channels?
        "allowed": 'all',         # All2ALL (train) - all , or One2All (train) - one2all, AR training - one
        "cluster": True,          # Something internal (don't bother)
    }
    
    model_architecture_ = {
        "N_layers": 4,            # Number of (D) & (U) blocks 
        "channel_multiplier": 64, # Parameter d_e (how the number of channels changes)
        "N_res": 8,               # Number of (R) blocks in the middle networs.
        "N_res_neck" : 8,         # Number of (R) blocks in the BN
        
        "batch_norm": 1,          # Should we use simple BN -- 1: use it? If is_time == 1, we turn it off
        "is_time": 1,             # Should we conditional BN/LN/IN?
        "nl_dim": "23",           # If yes, which norm? '23'-IN, '023'-BN, '123'-LN
        
        "in_size": 128,           # Resolution of the computational grid
        "activation": 'cno_lrelu_torch',# cno_lrelu, cno_lrelu_torch or lrelu
        
        "is_att": False,          # Should we use attention in the bottleneck? You could add it!
        "patch_size" : 1,         # ViT parameters, if is_att == True
        "dim_multiplier" : 1,
        "depth" : 2,
        "heads" : 2,
        "dim_head_multiplier" : 0.5,
        "mlp_dim_multiplier" : 1.0,
        "emb_dropout" : 0.
        }
    
    # AVAILABLE EXPERIMENTS:
    # "ns_brownian", "ns_pwc", "ns_gauss", "ns_sin", "ns_vortex", "ns_shear
    # "ns_pwc_t:
    # "eul_kh", "eul_riemann", "eul_riemann_kh", "eul_riemann_cur", "eul_gauss"
    # "rich_mesh", "rayl_tayl" "kolmogorov"
    # "wave_seismic", "wave_gauss", "allen_cahn"
    # "airfoil", "poisson_gauss", "helmholtz"
    
    # FOR PRETRAINING CNO-FM: which_example = "eul_ns_mix1"
    
    # WHAT IS THE EXPERIMENT?
    which_example = "wave_gauss"
    new_which_example = "new_wave_gauss" # this is a hardcoded workaround for the new all_experiments.json adapted to the new dataset.
    
    folder = "/cluster/home/svisnjic/Thalassa/fwd_results/checkpoints" # "--- PROVIDE THE FOLDER TO SAVE THE MODEL ----" 
    
else:
    raise ValueError("To many args")
    
    

#---------------------------------------------------------    
cluster = True  # We always tun on cluster

model_architecture_["batch_norm"] = True if model_architecture_["batch_norm"] in [1, True] else False
training_properties["time_input"] = True if training_properties["time_input"] in [1, True] else False
model_architecture_["is_att"]     = True if model_architecture_["is_att"]     in [1, True] else False
training_properties["cluster"]    = True if training_properties["cluster"]    in [1, True] else False

if model_architecture_["nl_dim"] in ["023"]:
    model_architecture_["nl_dim"] = [0,2,3]
elif model_architecture_["nl_dim"] in ["123"]:
    model_architecture_["nl_dim"] = [1,2,3]
elif model_architecture_["nl_dim"] in ["23"]:
    model_architecture_["nl_dim"] = [2,3]

#---------------------------------------------------------
if not cluster: # Set the defaulf folder
    folder = "--- PROVIDE THE FOLDER TO SAVE THE MODEL (no cluster) ----"  

if not os.path.exists(folder):
    os.makedirs(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

#---------------------------------------------------------
# Load parameters related to the specific experiment -- "DataLoaders/all_experiments.json"

Dict_EXP = json.load( open( "DataLoaders/all_experiments.json" ))
if new_which_example in Dict_EXP:
    loader_dict =  Dict_EXP[new_which_example]
else:
    raise ValueError("Please specify different benchmark")

# loader_dict: INFORMATION ABOUT THE EXPERIMENT, TRAINING, etc -- VERY IMPORTANT!
loader_dict["which"] = which_example
loader_dict["time_input"] = training_properties["time_input"]
loader_dict["cluster"] = training_properties["cluster"]
loader_dict["num_samples"] = training_properties["training_samples"]
loader_dict["dt"] = training_properties["dt"]
loader_dict["time_steps"] = training_properties["time_steps"]

#---------------------------------------------------------
# Which transitions during the training are allowed?
_allowed = []
if "include_zero" in loader_dict and loader_dict["include_zero"]:
    start_t = 0
else:
    start_t = 1
if training_properties['allowed'] == 'all':
    for t in range(start_t,loader_dict["time_steps"]+1):
        _allowed.append(t)
elif training_properties['allowed']  == "one2all":
    _allowed = None
elif training_properties['allowed'] == 'one':
    _allowed = [1]
loader_dict["allowed_tran"] = _allowed

#---------------------------------------------------------
# Initialize CNO
model   = CNO_time(in_dim =  loader_dict["in_dim"],               
                    in_size  = 128,                  
                    N_layers = model_architecture_["N_layers"],                
                    N_res =  model_architecture_["N_res"], 
                    N_res_neck = model_architecture_["N_res_neck"],           
                    channel_multiplier = model_architecture_["channel_multiplier"],  
                    batch_norm = model_architecture_["batch_norm"],    
                    out_dim = loader_dict["out_dim"],               
                    activation = model_architecture_["activation"], 
                    time_steps = loader_dict["time_steps"],
                    is_time = model_architecture_["is_time"],
                    nl_dim= model_architecture_["nl_dim"],

                    lr = training_properties["learning_rate"],
                    batch_size = training_properties["batch_size"],
                    weight_decay = training_properties["weight_decay"],
                    scheduler_step = training_properties["scheduler_step"],
                    scheduler_gamma = training_properties["scheduler_gamma"],

                    loader_dictionary = loader_dict,

                    is_att = model_architecture_["is_att"],
                    patch_size = model_architecture_["patch_size"],
                    dim_multiplier = model_architecture_["dim_multiplier"],
                    depth = model_architecture_["depth"],
                    heads = model_architecture_["heads"],
                    dim_head_multiplier = model_architecture_["dim_head_multiplier"],
                    mlp_dim_multiplier = model_architecture_["mlp_dim_multiplier"],
                    emb_dropout = model_architecture_["emb_dropout"])

#---------------------------------------------------------

ver = "_0903_newgauss" # Just a random string to be added to the model name


visual_callback = RolloutVisualCallback(
    save_dir=folder + "/visuals" + str(ver), 
    frequency=1  # Save every 10 epochs
)

checkpoint_callback = ModelCheckpoint(dirpath = folder+"/model"+str(ver), monitor='mean_val_l')
early_part = 10
early_stop_callback = EarlyStopping(monitor="mean_val_l", patience=training_properties["epochs"]//early_part)

logger = TensorBoardLogger(save_dir=folder, version=ver, name="logs")
trainer = Trainer(devices = -1,
                max_epochs = training_properties["epochs"],
                callbacks = [checkpoint_callback,early_stop_callback, visual_callback],
                strategy="ddp_find_unused_parameters_true", #IMPORTANT!!!
                logger=logger)

# --- ADDED RESUMPTION LOGIC ---
checkpoint_path = os.path.join(folder, "model" + str(ver), "epoch=3-step=34816.ckpt")

if os.path.exists(checkpoint_path):
    print(f"==================================================")
    print(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
    print(f"==================================================")
    trainer.fit(model, ckpt_path=checkpoint_path)
else:
    print(f"==================================================")
    print(f"NO CHECKPOINT FOUND. STARTING FRESH.")
    print(f"==================================================")
    trainer.fit(model)

trainer.validate(model)


dest_dir = folder + "/model" + str(ver)
os.makedirs(dest_dir, exist_ok=True)

# Move properties to the specific model version folder
for movable_file in ['/training_properties.txt', '/net_architecture.txt']:
    src = folder + movable_file
    if os.path.exists(src):
        shutil.move(src, dest_dir + movable_file)
#---------------------------------------------------------