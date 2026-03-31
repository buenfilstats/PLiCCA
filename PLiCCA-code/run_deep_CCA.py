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

from cca_zoo.deep import (
    DCCA,
    DCCA_NOI,
    DCCA_SDL,
    DCCAE,
    DVCCA,    
    architectures
)


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

if EXPERIMENT_NAME == "blood_cells":
    DATASET_LOAD_PATH = PATH / EXPERIMENT_NAME / "dataset.pkl"

    with open(DATASET_LOAD_PATH, "rb") as f:
        dataset_dict = pickle.load(f)
    train_dataset = utils.XYDataset(dataset_dict["Z"], dataset_dict["Y"])
elif EXPERIMENT_NAME == "brain_surfaces":
        
    DATASET_LOAD_PATH = PATH / EXPERIMENT_NAME / "brain_meshes_and_high_dim_dataset.pkl"
    
    with open(DATASET_LOAD_PATH, "rb") as f:
        dataset_dict = pickle.load(f)

    Y =  dataset_dict["Y_left"] #for now we'll just look at the left hemisphere
    X = dataset_dict["X"]
    train_dataset = utils.XYDataset(X, Y)
    # Y_right_all = dataset_dict_0["Y_right"]
    # train_dataset = utils.YDataset(Y)

elif EXPERIMENT_NAME == "rings_and_discs":
    DATASET_LOAD_PATH = PATH / EXPERIMENT_NAME / "rings_and_discs_dataset.pkl"
    
    with open(DATASET_LOAD_PATH, "rb") as f:
        dataset_dict = pickle.load(f)
    train_dataset = utils.XYDataset(dataset_dict["X"], dataset_dict["Y"])

if EXPERIMENT_NAME == "rings_and_discs":
    d = 4
    q = train_dataset.Y.shape[1]
    p = train_dataset.X.shape[1]



batch_size = 50 # set to None if you want to use the full dataset as a single batch
num_epochs = 10000

lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_flag = True # set to False to disable printing of the loss at each epoch
# proximal_flag = True
# prox_reg = 0.5
# beta=1e-6
# beta_recon = 1e08
# beta_reg = 1e-08

encoder_Y = nn.Sequential(
    nn.Linear(q, 40),
    nn.ReLU(),
    nn.Linear(40,20),
    nn.ReLU(),
    nn.Linear(20,d),
)


decoder_Y = nn.Sequential(
    nn.Linear(d, 20),
    nn.ReLU(),
    nn.Linear(20, 40),
    nn.ReLU(),
    nn.Linear(40, q),
    nn.Sigmoid()
)
encoder_X = nn.Sequential(
        nn.Linear(p, d),
)


decoder_X = nn.Sequential(
        nn.Linear(d, p),
)

METHOD_NAME = "DCCA"  # options are "DCCA", "DCCA_NOI", "DCCA_SDL", "DCCAE", "DVCCA"


if METHOD_NAME == "DCCA":
    model = DCCA(latent_dimensions=d, encoders=[encoder_Y, encoder_X])
elif METHOD_NAME == "DCCA_SDL":
    model = DCCA_SDL(latent_dimensions=d, encoders=[encoder_Y, encoder_X])
elif METHOD_NAME == "DCCA_NOI":
    model = DCCA_NOI(latent_dimensions=d, encoders=[encoder_Y, encoder_X],rho =0.1)
elif METHOD_NAME == "DCCAE":
    model = DCCAE(latent_dimensions=d, encoders=[encoder_Y, encoder_X], decoders=[decoder_Y, decoder_X],lam=0.99) #lambda higher means better reconstruction
elif METHOD_NAME == "DVCCA":
    #Specify encoder for Y here, not above
    encoder_Y = architectures.Encoder( 
        latent_dimensions=d,
        feature_size=q,
        variational=True,
        layer_sizes=(40,20,d),
    )
    model = DVCCA(latent_dimensions=d, encoders=[encoder_Y], decoders=[decoder_Y, decoder_X])

MODEL_LOAD_PATH = None # set to None if you don't want to load a model
#MODEL_LOAD_PATH =PATH / EXPERIMENT_NAME / "model_and_optimizer_deep_CCA.pth" #uncomment to keep training a previous model



# no need to touch the below
DICT_LOAD_PATH =PATH / EXPERIMENT_NAME / "dict_running_deep_CCA.pkl"

MODEL_SAVE_PATH= PATH / EXPERIMENT_NAME / "model_and_optimizer_deep_CCA.pth"
DICT_SAVE_PATH = PATH / EXPERIMENT_NAME / "dict_running_deep_CCA.pkl"

if MODEL_LOAD_PATH is None:

    output_current = utils.run_deep_CCA(model,train_dataset,batch_size,num_epochs,lr,device,print_flag,
                               shuffle_flag=True,graph_flag=True,LOAD_PATH=None)

    # there's no previous run, so set train_dict_running to None
    train_dict_running = None


else:
    output_current = utils.run_deep_CCA(model,train_dataset,batch_size,num_epochs,lr,device,print_flag,
                               shuffle_flag=True,graph_flag=True,LOAD_PATH=MODEL_LOAD_PATH)

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
    

#IMPORT VALIDATION DATASET

DATASET_LOAD_PATH = Path.cwd() / EXPERIMENT_NAME / "rings_and_discs_validation_dataset.pkl"

with open(DATASET_LOAD_PATH, "rb") as f:
    dataset_dict_val = pickle.load(f)

Y = train_dataset.Y.detach().cpu().numpy()
X = train_dataset.X.detach().cpu().numpy()
Y_val = dataset_dict_val["Y_val"]
X_val = dataset_dict_val["X_val"]

val_dataset = utils.XYDataset(X_val,Y_val)

# EVALUATE THE MODEL
# with DVCCA there is only one encoder, for Y. So we compute the correlation of U with X, rather than between U and V.

if METHOD_NAME == "DVCCA":
    encoder_Y = model.encoders[0].layers

    U = encoder_Y(torch.tensor(Y).float()).detach().cpu().numpy()
    U_val = encoder_Y(torch.tensor(Y_val).float()).detach().cpu().numpy()
    
    mycca_out = utils.mycca(X, U, 2)
    if mycca_out is None:
        print("mycca_out is None, no correlation detected on training set.")
        val_correlation = 0
    else:
        print("Correlation captured on training set, according to mycca",mycca_out['S'])
        V_val = X_val @ mycca_out['That']
        U_val_rotated = U_val @ mycca_out['Hhat']
        val_correlation =  [np.corrcoef(U_val_rotated[:, i], V_val[:, i])[0, 1] for i in range(U_val_rotated.shape[1])]
        print("Correlation captured on validation set, according to mycca:",val_correlation)
else:
    encoder_Y = model.encoders[0]
    encoder_X = model.encoders[1]

    U = encoder_Y(torch.tensor(Y).float()).detach().cpu().numpy()
    V = encoder_X(torch.tensor(X).float()).detach().cpu().numpy()
    V_val = encoder_X(torch.tensor(X_val).float()).detach().cpu().numpy()
    U_val = encoder_Y(torch.tensor(Y_val).float()).detach().cpu().numpy()

    cca_out = utils.CCA(V, U, 2)
    if cca_out is None:
        print("cca_out is None, no correlation detected on training set.")
        val_correlation = 0
    else:
        print("Correlation captured on training set:",cca_out['S'])
        V_val_rotated = V_val @ cca_out['T']
        U_val_rotated = U_val @ cca_out['H']
        val_correlation = [np.corrcoef(U_val_rotated[:, i], V_val_rotated[:, i])[0, 1] for i in range(U_val_rotated.shape[1])]
        print("Correlation captured on validation set:",val_correlation)
    
# With these methods, we can also compute the reconstruction error for Y
if METHOD_NAME == "DCCAE" or METHOD_NAME == "DVCCA":
    decoder_Y = model.decoders[0]
    Y_reconstructed = decoder_Y(torch.tensor(U).float()).detach().cpu().numpy()
    Y_val_reconstructed = decoder_Y(torch.tensor(U_val).float()).detach().cpu().numpy()
    train_reconstruction_error = np.mean((Y - Y_reconstructed)**2)
    val_reconstruction_error = np.mean((Y_val - Y_val_reconstructed)**2)
    print("Reconstruction error for Y on training set:", train_reconstruction_error)
    print("Reconstruction error for Y on validation set:", val_reconstruction_error)
else:
    val_reconstruction_error = 5000.0 #dummy value


# #Plot the loss
# train_dict = output_current["train_dict"]
# losses = train_dict["losses"]

# plt.figure(figsize=(10, 6))
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss vs Epoch')
# plt.grid(True)
# plt.show()