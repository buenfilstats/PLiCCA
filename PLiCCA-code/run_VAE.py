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

PATH = Path.cwd()

#EXPERIMENT_NAME = "blood_cells"
#EXPERIMENT_NAME = "brain_surfaces"
EXPERIMENT_NAME = "rings_and_discs"

if EXPERIMENT_NAME == "rings_and_discs":
    DATASET_LOAD_PATH = PATH / EXPERIMENT_NAME / "rings_and_discs_dataset.pkl"
    
    with open(DATASET_LOAD_PATH, "rb") as f:
        dataset_dict = pickle.load(f)
    train_dataset = utils.XYDataset(dataset_dict["X"], dataset_dict["Y"])


if EXPERIMENT_NAME == "rings_and_discs":
    d = 4
    q = train_dataset.Y.shape[1]
    p = train_dataset.X.shape[1]


    encoder_mean = nn.Sequential(
        nn.Linear(q, 40),
        nn.ReLU(),
        nn.Linear(40,20),
        nn.ReLU(),
        nn.Linear(20,d),
    )

    encoder_logvar = nn.Sequential(nn.Linear(q, d))

    decoder_mean = nn.Sequential(
        nn.Linear(d, 20),
        nn.ReLU(),
        nn.Linear(20, 40),
        nn.ReLU(),
        nn.Linear(40, q),
        nn.Sigmoid()
    )


batch_size = 50 # set to None if you want to use the full dataset as a single batch
num_epochs = 10000

lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print_flag = True # set to False to disable printing of the loss at each epoch
#proximal_flag = True
#prox_reg = 0.02
beta=1e-7
#convergence_tol = .1 # set to None to just run num_epoch iterations
convergence_tol = None
model = utils.VAE_no_X(d=d, q=q,beta=beta,encoder_mean=encoder_mean,decoder_mean=decoder_mean,encoder_logvar=encoder_logvar).to(device)

MODEL_LOAD_PATH = None # set to None if you don't want to load a model
#MODEL_LOAD_PATH =PATH / EXPERIMENT_NAME / "model_and_optimizer_VAE.pth" #uncomment to keep training a previous model



# no need to touch the below
DICT_LOAD_PATH =PATH / EXPERIMENT_NAME / "dict_running_VAE.pkl"

MODEL_SAVE_PATH= PATH / EXPERIMENT_NAME / "model_and_optimizer_VAE.pth"
DICT_SAVE_PATH = PATH / EXPERIMENT_NAME / "dict_running_VAE.pkl"
# MODEL_SAVE_PATH= PATH / EXPERIMENT_NAME / "model_and_optimizer_VAE_for_NF.pth"
# DICT_SAVE_PATH = PATH / EXPERIMENT_NAME / "dict_running_VAE_for_NF.pkl"

if MODEL_LOAD_PATH is None:

    output_current = utils.run_VAE(model,train_dataset,batch_size,num_epochs,lr,device,print_flag,beta=beta,
                               shuffle_flag=True,graph_flag=True,convergence_tol = convergence_tol,LOAD_PATH=None)

    # there's no previous run, so set train_dict_running to None
    train_dict_running = None
else:
    output_current = utils.run_VAE(model,train_dataset,batch_size,num_epochs,lr,device,print_flag,beta=beta,
                               shuffle_flag=True,graph_flag=True,convergence_tol=convergence_tol,LOAD_PATH=MODEL_LOAD_PATH)

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
    