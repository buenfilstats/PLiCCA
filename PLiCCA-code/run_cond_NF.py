import utils_for_cluster_first_try as utils
import torch
import numpy as np
import torch.nn as nn
#import torch.nn.functional as F
#import torch.nn.utils.parametrize as parametrize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from torch.optim import Adam
#from adam_mini import Adam_mini

import copy
#import proximal_gradient.proximalGradient as pg
import pickle

import importlib
importlib.reload(utils)

from pathlib import Path
import normflows as nf

PATH = Path.cwd()

#EXPERIMENT_NAME = "blood_cells"
#EXPERIMENT_NAME = "brain_surfaces"
EXPERIMENT_NAME = "rings_and_discs"


if EXPERIMENT_NAME == "rings_and_discs":
    DATASET_LOAD_PATH = PATH / EXPERIMENT_NAME / "rings_and_discs_dataset_for_NF.pkl"
    
    with open(DATASET_LOAD_PATH, "rb") as f:
        dataset_dict = pickle.load(f)
    train_dataset_NF = utils.XYDataset(dataset_dict["X"], dataset_dict["Zhat"])

p = train_dataset_NF.X.shape[1]
d = train_dataset_NF.Y.shape[1]

num_layers = 1  # instead of 4
flows = []
split = d // 2  # = 2

for i in range(num_layers):
    param_map = nf.nets.MLP([split, 4, 4, 2 * split], init_zeros=True)
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(d, mode='swap'))

base = nf.distributions.DiagGaussian(d)

    
# Now we can run the VAE with the best lambda
# approach 1, running the VAE with formulas and proximal gradient descent
#Change settings if we want to:
batch_size = 50 # set to None if you want to use the full dataset as a single batch
num_epochs = 10000

lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_flag = True # set to False to disable printing of the loss at each epoch
proximal_flag = True
prox_reg = 1.5
beta_reg = .5
#convergence_tol = .1 # set to None to just run num_epoch iterations
convergence_tol = None
model = utils.cond_NF(d=d, p=p, beta_reg=beta_reg, base=base, flows=flows).to(device)

MODEL_LOAD_PATH = None # set to None if you don't want to load a model
#MODEL_LOAD_PATH =PATH / EXPERIMENT_NAME / "model_and_optimizer_cond_NF.pth" #uncomment to keep training a previous model



# no need to touch the below
DICT_LOAD_PATH =PATH / EXPERIMENT_NAME / "dict_running_cond_NF.pkl"

MODEL_SAVE_PATH= PATH / EXPERIMENT_NAME / "model_and_optimizer_cond_NF.pth"
DICT_SAVE_PATH = PATH / EXPERIMENT_NAME / "dict_running_cond_NF.pkl"

if MODEL_LOAD_PATH is None:

    output_current = utils.run_cond_NF(model,train_dataset_NF,batch_size,num_epochs,lr,device,print_flag,proximal_flag=proximal_flag,prox_reg=prox_reg,
                               shuffle_flag=True,graph_flag=True,convergence_tol = convergence_tol,LOAD_PATH=None)

    # there's no previous run, so set train_dict_running to None
    train_dict_running = None

else:
    output_current = utils.run_cond_NF(model,train_dataset_NF,batch_size,num_epochs,lr,device,print_flag,proximal_flag=proximal_flag,prox_reg=prox_reg,
                               shuffle_flag=True,graph_flag=True,convergence_tol = convergence_tol,LOAD_PATH=MODEL_LOAD_PATH)

    #if there were previous runs, then we want to load train_dict_running
    with open(DICT_LOAD_PATH, "rb") as f:
        output_previous = pickle.load(f)
    train_dict_running = output_previous["train_dict"]

# get the results from the current run
train_dict_current_run = output_current["train_dict"]
train_dict_running = utils.append_dict_lists(train_dict_running, train_dict_current_run)

output_current["train_dict"] = train_dict_running
# save train_dict_running to a file
with open(DICT_SAVE_PATH, "wb") as f:
    pickle.dump(output_current, f)


torch.save({
        'model_state_dict': output_current["model"].state_dict(),
        'optimizer_state_dict': output_current["optimizer"].state_dict(),
    }, MODEL_SAVE_PATH)