#from sklearn.linear_model import MultiTaskLassoCV
#from sklearn.linear_model import MultiTaskElasticNetCV

import torch
import numpy as np
import torch.nn as nn
#import torch.nn.functional as F
#import torch.nn.utils.parametrize as parametrize
import matplotlib.pyplot as plt
from matplotlib import animation

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from torch.optim import Adam
from adam_mini import Adam_mini

import copy
#import proximal_gradient.proximalGradient as pg
import time
#from mpl_toolkits.mplot3d import Axes3D
import random

#from scipy.stats import ortho_group


from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from sklearn.linear_model import MultiTaskLassoCV
#from sklearn.linear_model import MultiTaskElasticNetCV

from collections import OrderedDict
import normflows as nf

from cca_zoo.deep import (
    DCCA,
    DCCA_NOI,
    DCCA_SDL,
    DCCAE,
    DVCCA,    
    architectures
)

# these three functions are used in order to warmstart the conditional VAE using the weights learned from an unsupervised VAE.
def _extract_state_dict(ckpt):
    """
    Accepts:
      - {'model_state_dict': <state_dict>, 'optimizer_state_dict': ...}
      - {'state_dict': <state_dict>}
      - a raw state_dict (OrderedDict[str, Tensor])
    Returns a state_dict (OrderedDict[str, Tensor]).
    """
    if isinstance(ckpt, (dict, OrderedDict)):
        for key in ('model_state_dict', 'state_dict'):
            if key in ckpt and isinstance(ckpt[key], (dict, OrderedDict)):
                return ckpt[key]
        # Heuristic: looks like a raw state_dict if values are tensors
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError("Could not find a state_dict in the provided checkpoint.")

def _align_module_prefix(src_sd, target_keys):
    """
    If source keys start with 'module.' and target doesn't (or vice versa),
    add/remove the prefix so names can match.
    """
    src_has_module = any(k.startswith('module.') for k in src_sd.keys())
    tgt_has_module = any(k.startswith('module.') for k in target_keys)

    # Nothing to change
    if src_has_module == tgt_has_module:
        return src_sd

    aligned = OrderedDict()
    if src_has_module and not tgt_has_module:
        for k, v in src_sd.items():
            aligned[k[len('module.'):] if k.startswith('module.') else k] = v
    elif not src_has_module and tgt_has_module:
        for k, v in src_sd.items():
            aligned['module.' + k] = v
    return aligned

def load_matching_weights(model, load_path, map_location='cpu', verbose=True):
    """
    Load matching parameters by (name, shape) from a checkpoint into `model`.
    Leaves everything else in `model` as-initialized.

    Args:
        model: nn.Module (the new model)
        load_path: path to the checkpoint file you saved with torch.save({...})
        map_location: e.g. 'cpu' or a torch.device
        verbose: print a short report

    Returns:
        report: dict with details of what was loaded/skipped.
    """
    ckpt = torch.load(load_path, map_location=map_location)
    old_sd = _extract_state_dict(ckpt)

    new_sd = model.state_dict()
    old_sd = _align_module_prefix(old_sd, new_sd.keys())

    matched = {}
    skipped_shape = []
    skipped_missing = []

    for k, v in old_sd.items():
        if k in new_sd:
            if new_sd[k].shape == v.shape:
                matched[k] = v
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(new_sd[k].shape)))
        else:
            skipped_missing.append(k)

    # Load only the matched subset; everything else stays as-initialized
    missing_keys, unexpected_keys = model.load_state_dict(matched, strict=False)

    report = {
        "loaded": sorted(matched.keys()),
        "num_loaded": len(matched),
        "skipped_shape": skipped_shape,  # [(name, old_shape, new_shape), ...]
        "skipped_missing_in_new": skipped_missing,  # names present in ckpt but not in new
        "left_uninitialized_in_model": missing_keys,  # names in new not provided (expected)
        "unexpected_in_load_state_dict": unexpected_keys,  # usually empty with strict=False
    }

    if verbose:
        print(f"[load_matching_weights] Loaded {len(matched)} tensors.")
        if skipped_shape:
            print(f"  Skipped (shape mismatch): {len(skipped_shape)}")
        if skipped_missing:
            print(f"  Skipped (not in new model): {len(skipped_missing)}")
        if missing_keys:
            print(f"  New model params left at init (no match provided): {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys ignored by load_state_dict: {unexpected_keys}")

    return report

def CV_split(full_train_dataset, K, k):
        n = len(full_train_dataset)
        rng = np.random.default_rng(0)  # deterministic split across jobs
        indices = rng.permutation(n)

        fold_sizes = np.full(K, n // K, dtype=int)
        fold_sizes[: n % K] += 1
        print(fold_sizes)
        fold_indices = []
        current = 0
        for fold_size in fold_sizes:
            fold_idx = indices[current:current + fold_size]
            fold_indices.append(fold_idx)
            current += fold_size

        val_dataset = full_train_dataset.subset(fold_indices[k])
        train_dataset = full_train_dataset.subset(np.concatenate([fold_indices[i] for i in range(K) if i != k]))
        return train_dataset, val_dataset

## translate job_id to hyperparameter indices and CV fold to use
def l_to_ijk(job_id: int, a: int, b: int, K: int) -> tuple[int, int, int]:
    if a <= 0 or b <= 0 or K <= 0:
        raise ValueError("a, b, and K must be positive.")
    total = a * b * K
    if not (0 <= job_id < total):
        raise ValueError(f"job_id must be in [0, {total-1}] for a*b*K={total}.")

    i = job_id // (b * K)
    rem = job_id % (b * K)
    j = rem // K
    k = rem % K
    return i, j, k


# Define the data from Y.
# Y is a matrix with N rows (number of datapoints) and q columns (number of features).
class YDataset(Dataset):
    def __init__(self, Y):
        self.Y = torch.tensor(Y).float().detach()
        self.len = self.Y.shape[0]

    def __getitem__(self, index):
        return self.Y[index,:]

    def __len__(self):
        return self.len
    # subset over the rows of Y
    def subset(self, indices):
        return YDataset(self.Y[indices,:])

class VAE_no_X(nn.Module):
    def __init__(self, d, q, beta,encoder_mean,decoder_mean,encoder_logvar):
        super(VAE_no_X, self).__init__()

        self.d = d
        self.q = q
        self.beta = beta
        self.encoder_mean = encoder_mean
        self.decoder_mean = decoder_mean
        self.encoder_logvar = encoder_logvar
        
        self.decoder_logvar = nn.parameter.Parameter(torch.tensor(1.0))

    def encode(self, y, eps: float = 1e-8):
        mu = self.encoder_mean(y)
        covariance = torch.diag_embed(torch.exp(self.encoder_logvar(y)) + eps)
        return torch.distributions.MultivariateNormal(mu, covariance_matrix = covariance)
        
    def reparameterize(self, dist):
        return dist.rsample()
    
    def forward(self, y):
        dist = self.encode(y)
        z = self.reparameterize(dist)
        y_recon = self.decoder_mean(z)

        # compute loss terms
        # this MSELoss actually computes the frobenious error and divides it by the number of entries, so p*N or q*N. (it doesn't just
        # divide by N)
        loss_recon_obj_Y = nn.MSELoss(reduction='mean')
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )

        # this is the term corresponding to KL(q(z|x,y)||p(z))
        loss_kl = self.beta * torch.distributions.kl.kl_divergence(dist, std_normal).mean() / self.d
        
        # this is the term corresponding to E_q[log p(y|z)]
        # (self.q * torch.log(2*torch.tensor(np.pi)))*0.5 is a constant term
        loss_recon_y = (0.5 * self.q * torch.log(2*torch.tensor(np.pi)) + 0.5 * self.q * self.decoder_logvar + 0.5 * self.q * loss_recon_obj_Y(y_recon, y)/torch.exp(self.decoder_logvar))/self.q
        #dividing by p and q in loss_recon_y and loss_recon_x is not strictly the MLE anymore, but I found that there is a better tradeoff between X and Y this way, since the dimension of Y tends to be so large.
        #print(loss_recon_obj_X(x_recon, x))
        
        # total loss
        loss =  loss_recon_y + loss_kl
        
        return {
            "z": self.encoder_mean(y),
            "y_recon": y_recon,
            "loss_kl": loss_kl,
            "loss_y_recon": loss_recon_y,
            "loss": loss
        }
    

def train_no_X(model, optimizer, epochs, device,train_loader,print_flag=False,graph_flag=False,convergence_tol = None):
    
    model.train()
    #zhats = []  #can set to None if we don't want to store these
    zhats = None
    losses = []
    loss_kls = []
    loss_y_recons = []
    encoder_weights = None #not storing these for now
    encoder_logvars = None #not storing these for now[]
    Y_decoder_logvars = []
    for epoch in range(epochs):
        overall_loss = 0
        overall_loss_kl = 0
        overall_loss_y_recon = 0
        for batch,(y) in enumerate(train_loader):
            y = y.to(device)

            output = model(y)  # Forward pass
            loss = output["loss"]

            overall_loss_kl += output["loss_kl"].item()
            overall_loss_y_recon += output["loss_y_recon"].item()
            overall_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0) # Clip the gradients to prevent exploding gradients
        if print_flag:
            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch+1))

        
        # Calculate the gradient norm for the Y_decoder_mean parameters
        # and store it for plotting
        #total_norm = 0
        #for param in model.decoder_mean.parameters():
        #    if param.grad is not None:
        #        param_norm = param.grad.data.norm(2)
        #        total_norm += param_norm.item() ** 2
        #total_norm = total_norm ** 0.5
        
        # store the loss at each epoch
        losses.append(overall_loss / (batch + 1))
        loss_kls.append(overall_loss_kl / (batch + 1))
        loss_y_recons.append(overall_loss_y_recon / (batch + 1))

        # store the log of the decoder variance at each iteration
        Y_decoder_logvars.append(model.decoder_logvar.detach().cpu().numpy().copy())

        # Store parameter values of the encoder weights at each epoch
        #encoder_weights.append(model.encoder_mean[0].weight.detach().cpu().numpy())

        # Get the diagonal of the weight matrix of encoder_logvar (assumes it's of the form nn.Sequential(Linear(q, d)))
        #encoder_logvars.append(np.diag(model.encoder_logvar[0].weight.detach().cpu().numpy().copy()))

        # Store the Zhat values for each epoch
        #zhats.append(model.encoder_mean(train_loader.dataset.Y.to(device)).detach().cpu().numpy().copy())
    
        if convergence_tol is not None and epoch > 0 and epoch >= 251:
            # Check for convergence over the last 250 epochs
            if abs(Y_decoder_logvars[-1] - Y_decoder_logvars[-250]) < convergence_tol:
                print(f"Converged at epoch {epoch + 1}")
                break

    # here we're returning things that are not just contained in the model; they are things that we can plot later to see how training went.
    return {
        "zhats": zhats,
        "losses": losses,
        "loss_kls": loss_kls,
        "loss_y_recons": loss_y_recons,
        "encoder_weights": encoder_weights,
        "Y_decoder_logvars": Y_decoder_logvars,
        "encoder_logvars": encoder_logvars
    }

# passing in a dataset with only Y, we just train a VAE on Y.
# if an XYDataset is passed in, we do the same thing,
# and afterwards run mycca between X and the learned Zhat (encoder mean of Y given Z)
def run_VAE(model,dataset,batch_size,num_epochs,lr,device,print_flag,beta=1.0,
                          shuffle_flag = True,graph_flag = False,convergence_tol = None, LOAD_PATH = None):
    Y = dataset.Y

    # If batch_size is None, we use the full dataset as a single batch.
    if batch_size is None:
        batch_size = Y.shape[0]
    
    optimizer = Adam_mini(
             named_parameters = model.named_parameters(),
             lr = lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,verbose=False)
    #optimizer.wv_names = {} 

    if LOAD_PATH is not None:
        print("Loading model from", LOAD_PATH)
        checkpoint = torch.load(LOAD_PATH, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #optimizer = Adam(model.parameters(), lr=lr)
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    #Standard gradient descent
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dataset_no_X = YDataset(Y)
    train_loader = DataLoader(dataset=dataset_no_X, batch_size=batch_size, shuffle=shuffle_flag)
    start_time = time.time()
    train_dict = train_no_X(model, optimizer, epochs=num_epochs, device=device,train_loader=train_loader,print_flag=print_flag,graph_flag=graph_flag,convergence_tol = convergence_tol)
    end_time = time.time()
    num_epochs_ran = len(train_dict["losses"])
    print(f"Average training time per epoch: {(end_time - start_time)/num_epochs_ran:.2f} seconds")
    model.eval()
    with torch.no_grad():
        # after training the model, we can evaluate it on the training data
        output = model(Y.float().to(device))
    Zhat = output["z"].cpu().numpy() # Zhat is a sample from the posterior distribution of z, given Y (not just the encoder mean)
    Yhat = output["y_recon"].cpu().numpy() # Yhat is the decoder mean of Y, evaluated at Zhat (not sampling from Y|Z)
    
    has_X = hasattr(dataset, 'X')
    if has_X:
        X = dataset.X.numpy()
        ccaoutput = mycca(X, Zhat, Zhat.shape[1], lassotype="group", SigmaX=None, SigmaY=None)
    else:
        ccaoutput = None
    # When sparse asymmetric CCA sets B to be 0, it is saying there is no correlation between X and Zhat.
    # In this case, we set T_est to be None.
    # If sparse CCA returned 0s or we didn't pass in an X, then we set T_est and H_est to be None.
    if ccaoutput is None:
        T_est = None
        H_est = None
        lambdas_est = None
        B_est = None
    else:
        T_est = ccaoutput["That"]
        H_est = ccaoutput["Hhat"]
        lambdas_est = ccaoutput['S']
        B_est = ccaoutput['B']
    return {"encoder_mean":model.encoder_mean, "Zhat":Zhat,
            "Yhat":Yhat,"decoder_mean":model.decoder_mean,
            "model":model,"optimizer":optimizer,"train_dict":train_dict,
            "T_est":T_est,"H_est":H_est,"lambdas_est":lambdas_est,"B_est":B_est}

# Define the data from X and Y.
# X and Y are matrices with N rows (number of datapoints) and p and q columns respectively (number of features).
class XYDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y).float()
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index,:], self.Y[index,:]

    def __len__(self):
        return self.len
    # subset over the rows of X and Y
    def subset(self, indices):
        return XYDataset(self.X[indices,:], self.Y[indices,:])
    
# import torch.nn.utils.prune as prune
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class cond_VAE(nn.Module):
    def __init__(self, d, q, p, beta, beta_reg, beta_recon,
                 encoder_mean, decoder_mean, encoder_logvar):
        super(cond_VAE, self).__init__()

        self.d = d
        self.q = q
        self.p = p
        self.beta = beta
        self.beta_reg = beta_reg
        self.beta_recon = beta_recon

        self.encoder_mean = encoder_mean      # maps y -> mu
        self.decoder_mean = decoder_mean      # maps z -> y_hat
        self.encoder_logvar = encoder_logvar  # maps y -> logvar (diag)

        # scalar log-variance for p(y|z)
        self.decoder_logvar = nn.Parameter(torch.tensor(1.0))

        # regression layer: maps x -> mu_x (for KL mean term)
        # B is the transpose of this weight matrix in your notation
        self.regression_layer = nn.Linear(
            in_features=self.p,
            out_features=self.d,
            bias=False
        )

        # 0.5 * log(2π), stored as a buffer so it moves with the model
        self.register_buffer("log_two_pi", torch.tensor(math.log(2 * math.pi)))

    # Encode just returns (mu, logvar) now
    def encode(self, y):
        mu = self.encoder_mean(y)         # (N, d)
        logvar = self.encoder_logvar(y)   # (N, d)
        return mu, logvar

    # Standard reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        # Encode once
        mu, logvar = self.encode(y)

        # Sample z and reconstruct y
        z = self.reparameterize(mu, logvar)
        y_recon = self.decoder_mean(z)

        # ----------------------------
        # KL variance term (vs N(0, I))
        # ----------------------------
        # term_i = 0.5 * ( -log|Σ| - d + tr(Σ) ) / d, Σ = diag(exp(logvar))
        term = 0.5 * (
            -logvar.sum(-1) - self.d + torch.exp(logvar).sum(-1)
        ) / self.d
        loss_kl_var = term.mean()  # scalar

        # ----------------------------
        # KL mean term (vs N(0, I))
        # ----------------------------
        # You had:
        #   loss_kl_mean = 0.5 * self.d * MSE(mu, Bx) / self.d
        # which simplifies to 0.5 * MSE(mu, Bx)
        reg_mean = self.regression_layer(x)  # (N, d)
        loss_kl_mean = 0.5 * F.mse_loss(mu, reg_mean, reduction="mean")

        # ----------------------------
        # Reconstruction term  E_q[ -log p(y|z) ]
        # ----------------------------
        # Original:
        # (0.5*q*log(2π) + 0.5*q*decoder_logvar + 0.5*q*MSE/exp(decoder_logvar)) / q
        # which simplifies to:
        # 0.5*log(2π) + 0.5*decoder_logvar + 0.5*MSE/exp(decoder_logvar)
        mse_y = F.mse_loss(y_recon, y, reduction="mean")
        decoder_var = torch.exp(self.decoder_logvar)

        loss_recon_y = (
            0.5 * self.log_two_pi +              # 0.5 * log(2π)
            0.5 * self.decoder_logvar +          # 0.5 * log σ^2
            0.5 * mse_y / decoder_var            # 0.5 * ||y - ŷ||^2 / σ^2
        )

        # ----------------------------
        # Total loss
        # ----------------------------
        loss = (
            self.beta_recon * loss_recon_y +
            self.beta_reg * loss_kl_mean +
            self.beta * loss_kl_var
        )

        return {
            "z": mu,  # reuse encoder mean, no extra forward pass
            "y_recon": y_recon,
            "loss_kl_mean": self.beta_reg * loss_kl_mean,
            "loss_kl_var":  self.beta * loss_kl_var,
            "loss_y_recon": self.beta_recon * loss_recon_y,
            "loss": loss,
        }

    

def l21(parameter, bias=None, reg=0.01, lr=0.1):
    """L21 Regularization"""
    
    if bias is not None:
        w_and_b = torch.cat((parameter, bias.unfold(0,1,1)),1)
    else:
        w_and_b = parameter
    L21 = reg # lambda: regularization strength
    Norm = (lr*L21/w_and_b.norm(2, dim=1)) # Key insight here: apply rowwise (by using dim 1)
    if Norm.is_cuda:
        ones = torch.ones(w_and_b.size(0), device=torch.device("cuda"))
    else:
        ones = torch.ones(w_and_b.size(0), device=torch.device("cpu"))
    l21T = 1.0 - torch.min(ones, Norm)
    update = (parameter*(l21T.unsqueeze(1)))
    #parameter.data = update
    # Update bias
    if bias is not None:
        update_b = (bias*l21T)
        #bias.data = update_b
        return update, update_b
    return update


def compute_validation_loss_cond_VAE(model, val_loader, device):
    model.eval()
    val_regr_loss = 0.0
    val_recon_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x, y)
            val_regr_loss += output["loss_kl_mean"].item()
            val_recon_loss += output["loss_y_recon"].item()
            count += 1

    model.train()
    return val_regr_loss / count, val_recon_loss / count

def train_cond_VAE(model, optimizer, epochs, device,train_loader,print_flag=False,proximal_flag = True,prox_reg=0.25,graph_flag=False,convergence_tol = None,val_flag = False):
    
    if val_flag:
        graph_flag = True
        # generate validation data
        val_data = generate_data(seed=0)
        val_dataset = XYDataset(val_data['X'], val_data['Y'])
        val_loader = DataLoader(dataset=val_dataset)
        val_loss_regrs = []
        val_loss_recons = []
        val_epochs = []
        Bs = []
    else:
        Bs = None
        val_loss_recons = None
        val_loss_regrs = None
    model.train()
    #zhats = []  #can set to None if we don't want to store these
    zhats = None
    losses = []
    loss_kl_means = []
    #loss_kl_vars = []
    loss_kl_vars = None
    loss_y_recons = []
    encoder_weights = None #not storing these for now
    encoder_logvars = None #not storing these for now[]
    #Y_decoder_logvars = []
    Y_decoder_logvars = None
    #Bs = []
    #Bs = None
    
    for epoch in range(epochs):
        overall_loss = 0
        overall_loss_kl_mean = 0
        #overall_loss_kl_var = 0
        overall_loss_y_recon = 0
        for batch,(x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            output = model(x, y)  # Forward pass
            loss = output["loss"]

            overall_loss_kl_mean += output["loss_kl_mean"].item()
            #overall_loss_kl_var += output["loss_kl_var"].item()
            overall_loss_y_recon += output["loss_y_recon"].item()
            overall_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if proximal_flag:
                # apply proximal gradient descent step to the regression layer
                # I had to mess with things a bit since I needed to transpose the weight matrix.
                # to add a bias and also apply the soft-thresholding to it, see the l21 function defined above.
                #print(model.regression_layer.weight.T)
                model.regression_layer.weight.data = l21(model.regression_layer.weight.T, bias=None, reg=prox_reg, lr=optimizer.param_groups[0]['lr']).T
                #print(model.regression_layer.weight.T)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0) # Clip the gradients to prevent exploding gradients
        if print_flag:
            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch+1))
    
        # Calculate the gradient norm for the Y_decoder_mean parameters
        # and store it for plotting
        #total_norm = 0
        #for param in model.decoder_mean.parameters():
        #    if param.grad is not None:
        #        param_norm = param.grad.data.norm(2)
        #        total_norm += param_norm.item() ** 2
        #total_norm = total_norm ** 0.5
        
        # store the loss at each epoch
        losses.append(overall_loss / (batch + 1))
        loss_kl_means.append(overall_loss_kl_mean / (batch + 1))
        #loss_kl_vars.append(overall_loss_kl_var / (batch + 1))
        loss_y_recons.append(overall_loss_y_recon / (batch + 1))


        # store the log of the decoder variance at each iteration
        #Y_decoder_logvars.append(model.decoder_logvar.detach().cpu().numpy().copy())

        # Store parameter values of the encoder weights at each epoch
        #encoder_weights.append(model.encoder_mean[0].weight.detach().cpu().numpy())

        # Get the diagonal of the weight matrix of encoder_logvar (assumes it's of the form nn.Sequential(Linear(q, d)))
        #encoder_logvars.append(np.diag(model.encoder_logvar[0].weight.detach().cpu().numpy().copy()))
        #Store the regression matrix B at each epoch
        # Store the Zhat values for each epoch
        #zhats.append(model.encoder_mean(train_loader.dataset.Y.to(device)).detach().cpu().numpy().copy())
        # x-axis = epochs 1..current
        # plot_every = 250
        # if val_flag and ((epoch + 1) % plot_every == 0):

        #     # Only now pay the cost of validation
        #     loss_regr_val, loss_recon_val = compute_validation_loss_cond_VAE(model, val_loader, device)
        #     val_loss_regrs.append(loss_regr_val)
        #     val_loss_recons.append(loss_recon_val)
        #     val_epochs.append(epoch + 1)   # x-axis positions for val curves
        #     with torch.no_grad():
        #         B_now = model.regression_layer.weight.detach().cpu().numpy().copy().T
        #     Bs.append(B_now)
            
        # if convergence_tol is not None and epoch > 0 and epoch >= 251:
        #     # Check for convergence over the last 250 epochs
        #     if abs(Y_decoder_logvars[-1] - Y_decoder_logvars[-250]) < convergence_tol:
        #         print(f"Converged at epoch {epoch + 1}")
        #         break
    if val_flag:
        validation_dict = {"Bs": Bs,"val_loss_recons": val_loss_recons,"val_loss_regrs": val_loss_regrs,"val_epochs": val_epochs}
    else:
        validation_dict = None
    # here we're returning things that are not just contained in the model; they are things that we can plot later to see how training went.
    return {
        "zhats": zhats,
        "losses": losses,
        "loss_kl_means": loss_kl_means,
        "loss_kl_vars": loss_kl_vars,
        "loss_y_recons": loss_y_recons,
        "encoder_weights": encoder_weights,
        "Y_decoder_logvars": Y_decoder_logvars,
        "encoder_logvars": encoder_logvars,
        "validation_dict": validation_dict
    }

def run_cond_VAE(model,dataset,batch_size,num_epochs,lr,device,print_flag,proximal_flag,prox_reg,
                          shuffle_flag = True,graph_flag = False,convergence_tol = None, LOAD_PATH = None,
                          WARM_START_UNSUP_VAE = False,val_flag = False):
    Y = dataset.Y
    X = dataset.X
    # If batch_size is None, we use the full dataset as a single batch.
    if batch_size is None:
        batch_size = Y.shape[0]
    
    optimizer = Adam_mini(
            named_parameters = model.named_parameters(),
            lr = lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,verbose=False)
    #optimizer.wv_names = {} 

    if LOAD_PATH is not None:
        if WARM_START_UNSUP_VAE:
            print("Warm starting from", LOAD_PATH)
            report = load_matching_weights(model, LOAD_PATH, map_location='cpu', verbose=True)
            print(report)
        else:
            checkpoint = torch.load(LOAD_PATH, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'],strict = False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #optimizer = Adam(model.parameters(), lr=lr)
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    #Standard gradient descent
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dataset = XYDataset(X,Y)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle_flag)
    start_time = time.time()
    train_dict = train_cond_VAE(model, optimizer, epochs=num_epochs, device=device,train_loader=train_loader,print_flag=print_flag,
                                proximal_flag=proximal_flag,prox_reg=prox_reg,graph_flag=graph_flag,convergence_tol = convergence_tol,val_flag=val_flag)
    end_time = time.time()
    num_epochs_ran = len(train_dict["losses"])
    print(f"Average training time per epoch: {(end_time - start_time)/num_epochs_ran:.3f} seconds")
    model.eval()
    with torch.no_grad():
        # after training the model, we can evaluate it on the training data
        output = model(X.float().to(device), Y.float().to(device))
    Zhat = output["z"].cpu().numpy() # Zhat is the encoder mean applied to Y (not a sample from q(z|y))
    Yhat = output["y_recon"].cpu().numpy() # Yhat is the decoder mean of Y, evaluated at Z (not sampling from Y|Z, but we
    # did sample from q(z|y) to get Z)
    B_est = model.regression_layer.weight.detach().cpu().numpy().T # B_est is p by d

    Sigma_Z_est = (Zhat- np.mean(Zhat, axis=0)).T @ (Zhat- np.mean(Zhat, axis=0)) / Zhat.shape[0]
    
    # Remove zero-variance latent dimensions (diagonal entries equal to 0)
    diag_vals = np.diag(Sigma_Z_est)
    zero_var_idx = np.where(diag_vals < 1e-9)[0]

    if zero_var_idx.size > 0:
        mask = np.ones(len(diag_vals), dtype=bool)
        mask[zero_var_idx] = False
        if mask.sum() == 0:
            print("All latent dimensions have zero variance; skipping CCA-related computations.")
        else:
            Sigma_Z_est = Sigma_Z_est[mask][:, mask]
            B_est = B_est[:, mask]
            Zhat = Zhat[:, mask]  # Keep Zhat consistent with reduced covariance
            print(f"Pruned {zero_var_idx.size} zero-variance latent dims. Remaining dims: {mask.sum()}")
    #d_rec = Sigma_Z_est.shape[0]
    X_zero_mean = X.cpu().numpy() - np.mean(X.cpu().numpy(), axis=0)
    Sigma_X_est =  X_zero_mean.T @ X_zero_mean / X_zero_mean.shape[0]
    Sigma_Z_est_neg_sqrt = neg_sqrt(Sigma_Z_est)
    Sigma_X_est_sqrt = sqrt(Sigma_X_est) # + np.eye(Sigma_X_est.shape[0]) # can optionally add 
                                         # a ridge term before the sqrt to ensure invertibility
    if Sigma_X_est_sqrt is None:
        print("Sigma_X_est is not invertible, cannot compute its square root")
        T_est = None
        H_est = None
        lambdas_est = None
    elif Sigma_Z_est_neg_sqrt is None:
        print("Sigma_Z_est is not invertible, cannot compute its negative square root")
        T_est = None
        H_est = None
        lambdas_est = None
    else: # good to compute the SVD
        try:
            U, lambdas_est, H_tilde_T  = np.linalg.svd(Sigma_X_est_sqrt @ B_est @ Sigma_Z_est_neg_sqrt)
        except np.linalg.LinAlgError:
            print("Error in computing square root: SVD did not converge.")
            T_est = None
            H_est = None
            lambdas_est = None

        if lambdas_est is not None:
            # if the kth lambdas is too small, the VAE is saying there is no correlation beyond the kth canonical pair.
            if lambdas_est.min() < 1e-9:
                idx_no_more_corr = np.where(lambdas_est < 1e-9)[0]
                # all of the canonical correlations are too small, so we set T_est and H_est to be None
                if idx_no_more_corr.size == lambdas_est.size:
                    print("All canonical correlations are 0 or negative")
                    T_est = None
                    H_est = None
                    lambdas_est = None
                # some of the canonical correlations are too small, but some are ok.
                # In this case, we truncate T_est and lambdas_est, but not H_est.
                else:
                    print(f"Truncating to first {idx_no_more_corr[0]} canonical correlations")
                    H_est = Sigma_Z_est_neg_sqrt @ H_tilde_T.T
                    lambdas_est = lambdas_est[:idx_no_more_corr[0]]
                    T_est = B_est @ H_est[:, :idx_no_more_corr[0]] @ np.diag(lambdas_est ** (-1))
            # all of the canonical correlations are ok, so we keep everything and don't need to truncate anything.
            else:
                H_est = Sigma_Z_est_neg_sqrt @ H_tilde_T.T
                T_est = B_est @ H_est @ np.diag(lambdas_est ** (-1))

    return {"encoder_mean":model.encoder_mean, "Zhat":Zhat,
                "Yhat":Yhat,"decoder_mean":model.decoder_mean,"model":model,"optimizer":optimizer,"train_dict":train_dict,
                "H_est":H_est,"T_est":T_est,"B_est":B_est,"Sigma_Z_est":Sigma_Z_est,"Sigma_X_est":Sigma_X_est,"Zhat":Zhat,
                "lambdas_est":lambdas_est,"inactive_latent_dims": zero_var_idx if zero_var_idx.size > 0 else None}



class cond_NF(nf.NormalizingFlow):
    def __init__(self, d, p, beta_reg, base, flows):
        super().__init__(base, flows)

        self.d = d
        self.p = p
        self.beta_reg = beta_reg

        self.regression_layer = nn.Linear(in_features=self.p, out_features=self.d, bias=False) # B is the tranpose of the weight matrix here
        # d_unsup = 1
        #  # Permanently zero the last d_unsup rows in forward/backward
        # if d_unsup > 0:
        #     mask = torch.ones(self.d, self.p)
        #     mask[-d_unsup:, :] = 0
        #     prune.custom_from_mask(self.regression_layer, name="weight", mask=mask)
        #     # Optional sanity check:
        #     with torch.no_grad():
        #         assert torch.all(self.regression_layer.weight[-d_unsup:, :] == 0)


    def forward(self, x, y):
        z, log_det = self.inverse_and_log_det(y)

        loss_log_det = - torch.mean(log_det)

        loss_recon_obj_mean = nn.MSELoss(reduction='mean')
        loss_reg = 0.5 * loss_recon_obj_mean(z, self.regression_layer(x))

        # total loss
        loss =  self.beta_reg * loss_reg + loss_log_det

        return {
            "z": z,
            "loss_reg": self.beta_reg * loss_reg,
            "loss_log_det": loss_log_det,
            "loss": loss
        }


def train_cond_NF(model, optimizer, epochs, device,train_loader,print_flag=False,proximal_flag = True,prox_reg=0.25,graph_flag=False,convergence_tol = None):
    
    model.train()
    #zhats = []  #can set to None if we don't want to store these
    zhats = None
    losses = []
    losses_reg = []
    losses_log_det = []

    if graph_flag:
        Bs = []
    else:
        Bs = None
    for epoch in range(epochs):
        overall_loss = 0
        overall_loss_reg = 0
        overall_loss_log_det = 0
        for batch,(x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            output = model(x, y)  # Forward pass
            loss = output["loss"]

            overall_loss_reg += output["loss_reg"].item()
            overall_loss_log_det += output["loss_log_det"].item()
            overall_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if proximal_flag:
                # apply proximal gradient descent step to the regression layer
                # I had to mess with things a bit since I needed to transpose the weight matrix.
                # to add a bias and also apply the soft-thresholding to it, see the l21 function defined above.
                #print(model.regression_layer.weight.T)
                model.regression_layer.weight.data = l21(model.regression_layer.weight.T, bias=None, reg=prox_reg, lr=optimizer.param_groups[0]['lr']).T
                #print(model.regression_layer.weight.T)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0) # Clip the gradients to prevent exploding gradients
        if print_flag:
            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch+1))


        losses.append(overall_loss / (batch + 1))
        losses_reg.append(overall_loss_reg / (batch + 1))
        losses_log_det.append(overall_loss_log_det / (batch + 1))
        if graph_flag:
            # Calculate the gradient norm for the Y_decoder_mean parameters
            # and store it for plotting
            #total_norm = 0
            #for param in model.decoder_mean.parameters():
            #    if param.grad is not None:
            #        param_norm = param.grad.data.norm(2)
            #        total_norm += param_norm.item() ** 2
            #total_norm = total_norm ** 0.5
            
            # store the loss at each epoch
            

            # Store the regression matrix B at each epoch
            Bs.append(model.regression_layer.weight.detach().cpu().numpy().copy().T)
            
            # Store the Zhat values for each epoch
            #zhats.append(model.encoder_mean(train_loader.dataset.Y.to(device)).detach().cpu().numpy().copy())

    # here we're returning things that are not just contained in the model; they are things that we can plot later to see how training went.
    return {
        "zhats": zhats,
        "losses": losses,
        "losses_reg": losses_reg,
        "losses_log_det": losses_log_det,
        "Bs": Bs,
    }

def run_cond_NF(model,dataset,batch_size,num_epochs,lr,device,print_flag,proximal_flag,prox_reg,
                          shuffle_flag = True,graph_flag = False,convergence_tol = None, LOAD_PATH = None,WARM_START_UNSUP_VAE = False):
    Y = dataset.Y
    X = dataset.X
    # If batch_size is None, we use the full dataset as a single batch.
    if batch_size is None:
        batch_size = Y.shape[0]
    
    optimizer = Adam_mini(
            named_parameters = model.named_parameters(),
            lr = lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,verbose=False)
    #optimizer.wv_names = {} 

    if LOAD_PATH is not None:
        
        checkpoint = torch.load(LOAD_PATH, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'],strict = False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #optimizer = Adam(model.parameters(), lr=lr)
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    #Standard gradient descent
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dataset = XYDataset(X,Y)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle_flag)
    start_time = time.time()
    train_dict = train_cond_NF(model, optimizer, epochs=num_epochs, device=device,train_loader=train_loader,print_flag=print_flag,
                                proximal_flag=proximal_flag,prox_reg=prox_reg,graph_flag=graph_flag,convergence_tol = convergence_tol)
    end_time = time.time()
    num_epochs_ran = len(train_dict["losses"])
    print(f"Average training time per epoch: {(end_time - start_time)/num_epochs_ran:.3f} seconds")
    model.eval()
    with torch.no_grad():
        # after training the model, we can evaluate it on the training data
        output = model(X.float().to(device), Y.float().to(device))
    Zhat = output["z"].cpu().numpy() # Zhat is the encoder mean applied to Y (not a sample from q(z|y))
    # did sample from q(z|y) to get Z)
    B_est = model.regression_layer.weight.detach().cpu().numpy().T # B_est is p by d

    Sigma_Z_est = (Zhat- np.mean(Zhat, axis=0)).T @ (Zhat- np.mean(Zhat, axis=0)) / Zhat.shape[0]
    X_zero_mean = X.cpu().numpy() - np.mean(X.cpu().numpy(), axis=0)
    Sigma_X_est =  X_zero_mean.T @ X_zero_mean / X_zero_mean.shape[0]
    Sigma_Z_est_neg_sqrt = neg_sqrt(Sigma_Z_est)
    Sigma_X_est_sqrt = sqrt(Sigma_X_est) # + np.eye(Sigma_X_est.shape[0]) # can optionally add 
                                         # a ridge term before the sqrt to ensure invertibility
    if Sigma_X_est_sqrt is None:
        print("Sigma_X_est is not invertible, cannot compute its square root")
        T_est = None
        H_est = None
        lambdas_est = None
    elif Sigma_Z_est_neg_sqrt is None:
        print("Sigma_Z_est is not invertible, cannot compute its negative square root")
        T_est = None
        H_est = None
        lambdas_est = None
    else: # good to compute the SVD
        try:
            U, lambdas_est, H_tilde_T  = np.linalg.svd(Sigma_X_est_sqrt @ B_est @ Sigma_Z_est_neg_sqrt)
        except np.linalg.LinAlgError:
            print("Error in computing square root: SVD did not converge.")
            T_est = None
            H_est = None
            lambdas_est = None

        if lambdas_est is not None:
            # if the kth lambdas is too small, the VAE is saying there is no correlation beyond the kth canonical pair.
            if lambdas_est.min() < 1e-9:
                idx_no_more_corr = np.where(lambdas_est < 1e-9)[0]
                # all of the canonical correlations are too small, so we set T_est and H_est to be None
                if idx_no_more_corr.size == lambdas_est.size:
                    print("All canonical correlations are 0 or negative")
                    T_est = None
                    H_est = None
                    lambdas_est = None
                # some of the canonical correlations are too small, but some are ok.
                # In this case, we truncate T_est and lambdas_est, but not H_est.
                else:
                    print(f"Truncating to first {idx_no_more_corr[0]} canonical correlations")
                    H_est = Sigma_Z_est_neg_sqrt @ H_tilde_T.T
                    lambdas_est = lambdas_est[:idx_no_more_corr[0]]
                    T_est = B_est @ H_est[:, :idx_no_more_corr[0]] @ np.diag(lambdas_est ** (-1))
            # all of the canonical correlations are ok, so we keep everything and don't need to truncate anything.
            else:
                H_est = Sigma_Z_est_neg_sqrt @ H_tilde_T.T
                T_est = B_est @ H_est @ np.diag(lambdas_est ** (-1))

    return {"Zhat":Zhat,"model":model,"optimizer":optimizer,"train_dict":train_dict,
                "H_est":H_est,"T_est":T_est,"B_est":B_est,"Sigma_Z_est":Sigma_Z_est,
                "Sigma_X_est":Sigma_X_est,"lambdas_est":lambdas_est,}


def append_dict_lists(dict1, dict2):
    """
    Appends lists from dict2 to lists in dict1 for matching keys.
    Returns a new dictionary.
    """
    if dict1 is None:
        return dict2
    result = {}
    for key in dict1:
        if key in dict2:
            if dict1[key] is None or dict2[key] is None:
                result[key] = None
            else:
                result[key] = dict1[key] + dict2[key]
        else:
            result[key] = dict1[key]
    # Add keys that are only in dict2
    for key in dict2:
        if key not in result:
            result[key] = dict2[key]
    return result

def neg_sqrt(A):
    try:
        U_svd, S_svd, Vt_svd = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        print("Error in computing negative square root: SVD did not converge.")
        return None
    if min(S_svd) < 0:
            print("Error in computing negative square root: Matrix is not positive definite.")
            return None
    else:
        neg_sqrt_eigvals = 1/np.sqrt(S_svd)
        return U_svd @ np.diag(neg_sqrt_eigvals) @ U_svd.T

def sqrt(A):
    try:
        U_svd, S_svd, Vt_svd = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        print("Error in computing square root: SVD did not converge.")
        return None
    if min(S_svd) < 0:
            print("Error in computing square root: Matrix is not positive definite.")
            return None
    else:
        neg_sqrt_eigvals = np.sqrt(S_svd)
        return U_svd @ np.diag(neg_sqrt_eigvals) @ U_svd.T

def graph_training_output(train_dict,model_name = None,num_skip_first_epoch=0):
    if model_name == "unsup_VAE":
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs = axs.flatten()

        epochs = len(train_dict["losses"]) - num_skip_first_epoch
        losses = train_dict["losses"][num_skip_first_epoch:]
        loss_kls = train_dict["loss_kls"][num_skip_first_epoch:]
        loss_y_recons = train_dict["loss_y_recons"][num_skip_first_epoch:]
        #encoder_weights = train_dict["encoder_weights"][num_skip_first_epoch:]
        Y_decoder_logvars = train_dict["Y_decoder_logvars"][num_skip_first_epoch:]
        #encoder_logvars = train_dict["encoder_logvars"][num_skip_first_epoch:]
        #encoder_vars = np.exp(encoder_logvars)
        # Plot Log of Y Decoder Variance
        axs[0].plot(range(1, epochs + 1), Y_decoder_logvars, marker='o')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Log of Y Decoder Variance')
        axs[0].set_title('Log of Y Decoder Variance vs Epoch')
        axs[0].grid(True)


        # Plot Loss vs Epoch
        axs[1].plot(range(1, epochs + 1), losses, marker='o',label ='Total Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Loss vs Epoch')
        axs[1].grid(True)

        axs[1].plot(range(1, epochs + 1), loss_kls, marker='o', label='KL Loss')
        axs[1].plot(range(1, epochs + 1), loss_y_recons, marker='o', label='Y Reconstruction Loss')
        axs[1].legend()
        
        # # Plot Encoder Log Variances
        # for j in range(encoder_vars.shape[1]):
        #     axs[2].plot(range(1, epochs + 1), encoder_vars[:, j], marker='o', label=f'Encoder Variance {j+1}')

        # axs[2].set_xlabel('Epoch')
        # axs[2].set_ylabel('Encoder Variance')
        # axs[2].set_title('Encoder Variance vs Epoch')
        # axs[2].legend()
        # axs[2].grid(True)
        plt.tight_layout()
        plt.show()

    elif model_name == "cond_VAE":
        # 3 subplots:
        # 1) recon loss (train + val)
        # 2) regression loss (train + val)
        # 3) row 2-norms of B (from validation_dictionary["Bs"])
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        ax_recon, ax_reg, ax_B = axs

        # --- Training quantities (after skipping first epochs) ---
        losses = train_dict["losses"][num_skip_first_epoch:]
        loss_kl_means = train_dict["loss_kl_means"][num_skip_first_epoch:]
        loss_y_recons = train_dict["loss_y_recons"][num_skip_first_epoch:]

        epochs_train = np.arange(1, len(loss_y_recons) + 1)

        # --- Validation quantities (if available) ---
        val_dict = train_dict["validation_dict"]
        if val_dict is not None:
            val_epochs_raw = np.array(val_dict["val_epochs"])        # original epoch numbers (1-based)
            val_loss_recons_raw = np.array(val_dict["val_loss_recons"])
            val_loss_regrs_raw = np.array(val_dict["val_loss_regrs"])
            Bs_raw = val_dict["Bs"]                                  # list of B matrices at val epochs

            # Keep only epochs > num_skip_first_epoch
            mask = val_epochs_raw > num_skip_first_epoch
            val_epochs = val_epochs_raw[mask] - num_skip_first_epoch  # reindex to match epochs_train starting at 1
            val_loss_recons = val_loss_recons_raw[mask]
            val_loss_regrs = val_loss_regrs_raw[mask]
            # Apply same mask to Bs (still calling it Bs)
            Bs = [B for B, keep in zip(Bs_raw, mask) if keep]
        else:
            val_epochs = None
            val_loss_recons = None
            val_loss_regrs = None
            Bs = None

        # ---------------------------------------------------------
        # 1) Recon loss (training + validation)
        # ---------------------------------------------------------
        ax_recon.plot(epochs_train, loss_y_recons, marker='o', label='Train recon')

        if val_epochs is not None and len(val_loss_recons) > 0:
            ax_recon.plot(val_epochs, val_loss_recons, marker='s', label='Val recon')

        ax_recon.set_xlabel('Epoch')
        ax_recon.set_ylabel('Reconstruction loss')
        ax_recon.set_title('Reconstruction loss vs Epoch')
        ax_recon.legend()
        ax_recon.grid(True)

        # ---------------------------------------------------------
        # 2) Regression loss (training + validation)
        # ---------------------------------------------------------
        ax_reg.plot(epochs_train, loss_kl_means, marker='o', label='Train regression')

        if val_epochs is not None and len(val_loss_regrs) > 0:
            ax_reg.plot(val_epochs, val_loss_regrs, marker='s', label='Val regression')

        ax_reg.set_xlabel('Epoch')
        ax_reg.set_ylabel('Regression loss')
        ax_reg.set_title('Regression loss vs Epoch')
        ax_reg.legend()
        ax_reg.grid(True)

        # ---------------------------------------------------------
        # 3) Row 2-norms of B (from validation_dictionary["Bs"])
        # ---------------------------------------------------------
        if Bs is not None and len(Bs) > 0:
            Bs_array = np.stack(Bs, axis=0)              # shape: (num_val_points, n_rows, n_cols)
            row_norms = np.linalg.norm(Bs_array, axis=2) # (num_val_points, n_rows)

            for i in range(row_norms.shape[1]):
                ax_B.plot(val_epochs, row_norms[:, i], marker='o', label=f'Row {i+1} norm')

            ax_B.set_xlabel('Epoch')
            ax_B.set_ylabel('Row 2-norm of B')
            ax_B.set_title('Row 2-norms of B vs Epoch')
            ax_B.legend(ncol=2, fontsize=8)
            ax_B.grid(True)

        plt.tight_layout()
        plt.show()

    elif model_name == "cond_NF":
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs = axs.flatten()

        epochs = len(train_dict["losses"]) - num_skip_first_epoch
        losses = train_dict["losses"][num_skip_first_epoch:]
        losses_reg = train_dict["losses_reg"][num_skip_first_epoch:]
        losses_log_det = train_dict["losses_log_det"][num_skip_first_epoch:]


        # Plot Loss vs Epoch
        axs[1].plot(range(1, epochs + 1), losses, marker='o',label ='Total Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Loss vs Epoch')
        axs[1].grid(True)

        axs[1].plot(range(1, epochs + 1), losses_reg, marker='o', label='Regression loss')
        axs[1].plot(range(1, epochs + 1), losses_log_det, marker='o', label='Log determinant loss')
        axs[1].legend()
        
        # # Plot Encoder Log Variances
        # for j in range(encoder_vars.shape[1]):
        #     axs[2].plot(range(1, epochs + 1), encoder_vars[:, j], marker='o', label=f'Encoder Variance {j+1}')

        # axs[2].set_xlabel('Epoch')
        # axs[2].set_ylabel('Encoder Variance')
        # axs[2].set_title('Encoder Variance vs Epoch')
        # axs[2].legend()
        # axs[2].grid(True)
        plt.tight_layout()
        plt.show()

def out_of_sample_correlation(val_dataset,encoder_mean,H,T,inactive_latent_dims=None,num_corr=None):
    # if there was no correlation found, we return 0 correlation
    if H is None or T is None:
        return 0.0
    
    X_test = val_dataset.X
    Y_test = val_dataset.Y
    
    Z_test = encoder_mean(Y_test).detach().numpy()

    # if there are inactive latent dimensions, then we need to remove them from Z
    if inactive_latent_dims is not None:

        mask = np.ones(Z_test.shape[1], dtype=bool)
        mask[inactive_latent_dims] = False
        if mask.sum() == 0:
            print("All latent dimensions have zero variance; returning 0 for out-of-sample correlations.")
            return [0.0]
        else:
            Z_test = Z_test[:, mask]  # remove inactive latent dimensions

    U = X_test.detach().numpy()@T
    V = Z_test@H
    
    if num_corr is not None:
        U = U[:, 0:num_corr]
        V = V[:, 0:num_corr]
    # the sum of the out-of-sample canonical correlations
    return [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(U.shape[1])]

# computes out of sample correlation between a new Y and X
def out_of_sample_correlation_NF(val_dataset,encoder_mean,H,T,model,num_corr=None):
# if there was no correlation found, we return 0 correlation
    if H is None or T is None:
        return 0.0
    
    X_test = val_dataset.X
    Y_test = val_dataset.Y
    
    W_test = encoder_mean(Y_test)
    Z_test = model.inverse(W_test).detach().numpy()
    U = X_test.detach().numpy()@T
    V = Z_test@H

    if num_corr is not None:
        U = U[:, 0:num_corr]
        V = V[:, 0:num_corr]
    
    # the sum of the out-of-sample canonical correlations
    return [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(U.shape[1])]

# computes out of sample correlation between a new Z and X
def out_of_sample_correlation_NF_Z(val_dataset,H,T,num_corr=None):
# if there was no correlation found, we return 0 correlation
    if H is None or T is None:
        return 0.0
    
    X_test = val_dataset.X
    Z_test = val_dataset.Y
    
    U = X_test.detach().numpy()@T
    V = Z_test.detach().numpy()@H
    
    
    if num_corr is not None:
        U = U[:, 0:num_corr]
        V = V[:, 0:num_corr]
    # the sum of the out-of-sample canonical correlations
    return [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(U.shape[1])]

def out_of_sample_reconstruction_error(val_dataset,encoder_mean,decoder_mean):
    X_test = val_dataset.X
    Y_test = val_dataset.Y
    
    Z_test = encoder_mean(Y_test)
    Y_recon = decoder_mean(Z_test)

    return np.mean((Y_test.detach().numpy() - Y_recon.detach().numpy())**2)

def plot_canonical_variables(Uhat, Vhat, colors_1, colors_2):
        
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    # Plot U_1 vs V_1 colored by r_1
    scatter = axs[0].scatter(Uhat[:, 0], Vhat[:, 0], c=colors_1, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ax=axs[0], label='$r_3$ (constrast)')
    corr_1 = np.corrcoef(Uhat[:, 0], Vhat[:, 0])[0, 1]
    axs[0].text(0.05, 0.95, f'corr={corr_1:.2f}', transform=axs[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    axs[0].set_xlabel('First canonical variable $U_1$')
    axs[0].set_ylabel('First canonical variable $V_1$')
    axs[0].set_title('Canonical variables $U_1$ vs $V_1$')
    axs[0].set_xlim(-3, 3)
    axs[0].set_ylim(-3, 3)
    axs[0].grid(True)

    # Plot U_2 vs V_2 colored by r_2
    scatter = axs[1].scatter(Uhat[:, 1], Vhat[:, 1], c=colors_2, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ax=axs[1], label='$r_4$ (ellipticity)')
    corr_2 = np.corrcoef(Uhat[:, 1], Vhat[:, 1])[0, 1]
    axs[1].text(0.05, 0.95, f'corr={corr_2:.2f}', transform=axs[1].transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    axs[1].set_xlabel('Second canonical variable $U_2$')
    axs[1].set_ylabel('Second canonical variable $V_2$')
    axs[1].set_title('Canonical variables $U_2$ vs $V_2$')
    axs[1].set_xlim(-3, 3)
    axs[1].set_ylim(-3, 3)
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    # Plot 1: Projections of X onto the first two columns of T_est, colored by the first entry of Z
    scatter = axs[0, 0].scatter(Uhat[:, 0], Uhat[:, 1], c=colors_1, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ax=axs[0, 0], label='r_1 (Radius of the hole)')
    corr_v = np.corrcoef(Uhat[:, 0], Uhat[:, 1])[0, 1]
    axs[0, 0].text(0.05, 0.95, f'corr={corr_v:.2f}', transform=axs[0, 0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    axs[0, 0].set_title('Canonical variables U_1 and U_2 (colored by r_1)')
    axs[0, 0].set_xlim(-3, 3)
    axs[0, 0].set_ylim(-3, 3)
    axs[0, 0].grid(True)

    # Plot 2: Projections of X onto the first two columns of T_est, colored by the second entry of Z
    scatter = axs[0, 1].scatter(Uhat[:, 0], Uhat[:, 1], c=colors_2, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ax=axs[0, 1], label='r_2 (Width of the ring)')
    corr_v2 = np.corrcoef(Uhat[:, 0], Uhat[:, 1])[0, 1]
    axs[0, 1].text(0.05, 0.95, f'corr={corr_v2:.2f}', transform=axs[0, 1].transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    axs[0, 1].set_title('Canonical variables U_1 and U_2 (colored by r_2)')
    axs[0, 1].set_xlim(-3, 3)
    axs[0, 1].set_ylim(-3, 3)
    axs[0, 1].grid(True)

    # Plot 3: Projections of Zhat onto the first two columns of H_est, colored by the first entry of Z
    scatter = axs[1, 0].scatter(Vhat[:, 0], Vhat[:, 1], c=colors_1, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ax=axs[1, 0], label='r_1 (Radius of the hole)')
    corr_u = np.corrcoef(Vhat[:, 0], Vhat[:, 1])[0, 1]
    axs[1, 0].text(0.05, 0.95, f'corr={corr_u:.2f}', transform=axs[1, 0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    axs[1, 0].set_title('Canonical variables V_1 and V_2 (colored by r_1)')
    axs[1, 0].set_xlim(-3, 3)
    axs[1, 0].set_ylim(-3, 3)
    axs[1, 0].grid(True)

    # Plot 4: Projections of Zhat onto the first two columns of H_est, colored by the second entry of Z
    scatter = axs[1, 1].scatter(Vhat[:, 0], Vhat[:, 1], c=colors_2, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ax=axs[1, 1], label='r_2 (Width of the ring)')
    corr_u2 = np.corrcoef(Vhat[:, 0], Vhat[:, 1])[0, 1]
    axs[1, 1].text(0.05, 0.95, f'corr={corr_u2:.2f}', transform=axs[1, 1].transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    axs[1, 1].set_title('Canonical variables V_1 and V_2 (colored by r_2)')
    axs[1, 1].set_xlim(-3, 3)
    axs[1, 1].set_ylim(-3, 3)
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

# inputs are latent variables Zhat, the colors for the points, as well as an optional H_est.
# currently the colors are r_1 and r_2.
def plot_latent_variables(Zhat, colors_1, colors_2, color_1_label=None, color_2_label=None, H_est = None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    #compute mean of Zhat for plotting
    Zhat_mean = np.mean(Zhat, axis=0)

    if colors_1 is None:
        colors_1 = np.zeros(Zhat.shape[0])
    if colors_2 is None:
        colors_2 = np.zeros(Zhat.shape[0])
    # Set default color labels if not provided
    if color_1_label is None:
        color_1_label = 'color_1_label not specified'
    if color_2_label is None:
        color_2_label = 'color_2_label not specified'

    # Plot 1: Latent space Zhat in terms of r_1
    scatter = axs[0].scatter(Zhat[:, 0], Zhat[:, 1], c=colors_1, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ax=axs[0], label=color_1_label)
    axs[0].set_xlabel('Z[0]')
    axs[0].set_ylabel('Z[1]')
    axs[0].set_title(f'Latent Space Zhat Visualization ({color_1_label})')
    axs[0].grid(True)

    if H_est is not None:
        eta_1_normalized = H_est[:, 0] / np.linalg.norm(H_est[:, 0])
        eta_2_normalized = H_est[:, 1] / np.linalg.norm(H_est[:, 1])
    # Plot the columns of H_est for the first subplot
    if H_est is not None:
        axs[0].quiver(Zhat_mean[0], Zhat_mean[1], eta_1_normalized[0], eta_1_normalized[1], angles='xy', scale_units='xy', scale=1, color='r', label='H_est[:, 0]')
        axs[0].quiver(Zhat_mean[0], Zhat_mean[1], eta_2_normalized[0], eta_2_normalized[1], angles='xy', scale_units='xy', scale=1, color='b', label='H_est[:, 1]')
        axs[0].legend()

    # Plot 2: Latent space Zhat in terms of r_2
    scatter = axs[1].scatter(Zhat[:, 0], Zhat[:, 1], c=colors_2, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ax=axs[1], label=color_2_label)
    axs[1].set_xlabel('Z[0]')
    axs[1].set_ylabel('Z[1]')
    axs[1].set_title(f'Latent Space Zhat Visualization ({color_2_label})')
    axs[1].grid(True)

    if H_est is not None:
        eta_1_normalized = H_est[:, 0] / np.linalg.norm(H_est[:, 0])
        eta_2_normalized = H_est[:, 1] / np.linalg.norm(H_est[:, 1])
        # Plot the columns of H_est for the second subplot
        axs[1].quiver(Zhat_mean[0], Zhat_mean[1], eta_1_normalized[0], eta_1_normalized[1], angles='xy', scale_units='xy', scale=1, color='r', label='H_est[:, 0]')
        axs[1].quiver(Zhat_mean[0], Zhat_mean[1], eta_2_normalized[0], eta_2_normalized[1], angles='xy', scale_units='xy', scale=1, color='b', label='H_est[:, 1]')
        axs[1].legend()

    plt.tight_layout()
    plt.show()

# inputs are latent variables Zhat, the colors for the points, as well as an optional H_est.
# currently the colors are r_1 and r_2.
def plot_latent_variables_d_greater_than_2(Zhat, colors):
    d = Zhat.shape[1]
    #compute mean of Zhat for plotting
    Zhat_mean = np.mean(Zhat, axis=0)
    
    for k in range(d):
        color_label = f'Color (Dimension {k})'
        fig, axs = plt.subplots(d-1, d-1, figsize=(8, 6))
        for i in range(d):
            for j in range(i+1, d):
                scatter = axs[i, j-1].scatter(Zhat[:, i], Zhat[:, j], c=colors[:, k], cmap='viridis', alpha=0.7)
                fig.colorbar(scatter, ax=axs[i, j-1], label=color_label)
                axs[i, j-1].set_xlabel(f'Z[{i}]')
                axs[i, j-1].set_ylabel(f'Z[{j}]')
                axs[i, j-1].set_title(f'Dimensions {i} vs {j}')
                axs[i, j-1].grid(True)

        plt.tight_layout()
        plt.show()


def animate_zhats(zhats, colors=None, color_label=None, step=1):
    """
    Creates a looping animation of the Zhats over the epochs using FuncAnimation.

    Parameters:
        zhats (list): A list of N x d numpy arrays representing Zhats at each epoch.
        colors (numpy.ndarray): A numpy array of colors for each point in Zhat.
        color_label (str): Label for the colorbar.
        step (int): Number of epochs to skip between frames.
    """
    N = zhats[0].shape[0]  # Number of points in Zhat
    fig, ax = plt.subplots()
    # Find the min and max values for x and y in zhats
    x_min = min(zhat[:, 0].min() for zhat in zhats)
    x_max = max(zhat[:, 0].max() for zhat in zhats)
    y_min = min(zhat[:, 1].min() for zhat in zhats)
    y_max = max(zhat[:, 1].max() for zhat in zhats)

    # Set the x and y limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.set_title("Latent Space Zhat Animation")
    ax.set_xlabel("Z[0]")
    ax.set_ylabel("Z[1]")

    scatter = ax.scatter(np.zeros(N), np.zeros(N), c=colors, cmap='viridis', alpha=0.7)
    text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    if colors is not None and color_label is not None:
        cbar = fig.colorbar(scatter, ax=ax, label=color_label)

    def update(epoch):
        zhat = zhats[epoch]
        scatter.set_offsets(zhat)
        text.set_text(f'Epoch: {epoch}')
        return scatter, text

    interval = 200  # milliseconds between frames
    blit = True  # blitting doesn't work well with color updates

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(zhats), step), interval=interval, blit=blit)

    return ani


# Let's see how well the VAE reconstructs the images with only Y, no X
def plot_reconstruction_no_X(Y,Yhat,num_samples=None):
    """
    Plot original and reconstructed images.
    
    Parameters:
        Y (np.ndarray): Original Y data (images).
        Yhat (np.ndarray): Reconstructed Y data (images).
        num_samples (int): Number of samples to plot.
    """
    if num_samples is None or num_samples > len(Y):
        num_samples = len(Y)

    fig, axs = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    num_pixels = int(np.floor(np.sqrt(Y.shape[1])))
    indices = np.random.choice(len(Y), size=num_samples, replace=False)
    i = 0
    for j in indices:

        # Plot original image
        axs[i, 0].imshow(Y[j].reshape(num_pixels, num_pixels), cmap='gray', origin='lower')
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')
        
        # Plot reconstructed image
        axs[i, 1].imshow(Yhat[j].reshape(num_pixels, num_pixels), cmap='gray', origin='lower')
        axs[i, 1].set_title('Reconstructed Image')
        axs[i, 1].axis('off')
        i += 1
    plt.tight_layout()
    plt.show()


def plot_regression(Zhat, B, X):
    """
    Plots the latent variables Zhat and the regression lines defined by B and X.

    Parameters:
        Zhat (numpy.ndarray): An N x d array of latent variables.
        B (numpy.ndarray): A p x d regression coefficient matrix.
        X (numpy.ndarray): An N x p array of input features.
    """
    N, d = Zhat.shape
    p = X.shape[1]
    
    if B.shape != (p, d):
        raise ValueError(f"Shape of B should be ({p}, {d}), but got {B.shape}")

    # Compute the predicted Zhat from X and B
    Zhat_pred = X @ B  # Shape: (N, d)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot actual Zhat
    ax.scatter(Zhat[:, 0], Zhat[:, 1], c='blue', alpha=0.5, label='g(Y)')

    # Plot predicted Zhat from regression
    ax.scatter(Zhat_pred[:, 0], Zhat_pred[:, 1], c='orange', alpha=0.5, label='B^T@X')

    ax.set_title('g(Y) vs B^T@X')
    ax.set_xlabel('Z[0]')
    ax.set_ylabel('Z[1]')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()

    plt.tight_layout()
    plt.show()


# outputs the canonical vectors between (X,Y) given the covariance matrices
def CCA(X,Y,r,SigmaX = None,SigmaY=None,SigmaXY=None,reg_param_X= 0,reg_param_Y= 0):
    N, p = X.shape
    d = Y.shape[1]

    # Subtract means from the data
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    # Estimate SigmaX and SigmaY if not provided
    if SigmaX is None:
        SigmaX = (1 / (N - 1)) * X.T @ X

    if SigmaY is None:
        SigmaY = (1 / (N - 1)) * Y.T @ Y
    if SigmaXY is None:
        SigmaXY = (1 / (N - 1)) * X.T @ Y
    neg_sqrt_SigmaX = neg_sqrt(SigmaX + reg_param_X*np.eye(p))
    neg_sqrt_SigmaY = neg_sqrt(SigmaY + reg_param_Y*np.eye(d))
    if neg_sqrt_SigmaX is None:
        print("SigmaX was not positive definite.")
        return None
    if neg_sqrt_SigmaY is None:
        print("SigmaY was not positive definite.")
        return None
    else:
        U, S, Vt = np.linalg.svd(neg_sqrt_SigmaX@SigmaXY@neg_sqrt_SigmaY)

        T = (neg_sqrt_SigmaX@U)[:, 0:r]  # p x r
        H = (neg_sqrt_SigmaY@Vt.T)[:, 0:r]  # d x r
        S = S[0:r]  # r x 1
        return {"T":T,"H":H,"S":S}


def mycca(X, Y, r, lassotype="group", SigmaX=None, SigmaY=None,verbose_flag = False):
    N, p = X.shape
    d = Y.shape[1]

    # Subtract means from the data
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    # Estimate SigmaX and SigmaY if not provided
    if SigmaX is None:
        SigmaXhat = (1 / (N - 1)) * X.T @ X
    else:
        SigmaXhat = SigmaX

    if SigmaY is None:
        SigmaYhat = (1 / (N - 1)) * Y.T @ Y
    else:
        SigmaYhat = SigmaY

    # Transform data for lasso
    #but check if SigmaYhat is invertible first
    vals, vecs = np.linalg.eigh(SigmaYhat)
    
    if (min(vals) < 10**(-10)):
        print("SigmaYhat was not invertible.")
        return None

    SigmaYhat_minusonehalf = vecs @ np.diag(vals**(-0.5)) @ vecs.T
    
    
    Ytrans = Y @ SigmaYhat_minusonehalf



    if lassotype == "univariate":
        B = np.zeros((d, p))
        for j in range(d):
            lasso = MultiTaskLassoCV(cv=5, fit_intercept=False)
            lasso.fit(X, Ytrans[:, j])
            B[j, :] = lasso.coef_

    elif lassotype == "group":
        lasso = MultiTaskLassoCV(cv=5, fit_intercept=False,verbose= verbose_flag)
        #lasso = MultiTaskElasticNetCV(cv=5, fit_intercept=False,verbose= False)

        lasso.fit(X, Ytrans)
        B = lasso.coef_.T

    else:
        print("lassotype not specified correctly")
        return None

    # Compute eigenvectors of B^T SigmaX B
    A = B.T @ SigmaXhat @ B

    #find eigenvector decomposition, then sort from largest to smallest by eigenvalue
    vals, vecs = np.linalg.eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)

    idx = vals.argsort()[::-1]    
    vals = vals[idx]
    vecs = vecs[:,idx]

    keep = min(r, np.sum(vals > 10**(-8)))
    #only want the ones that aren't too small
    if keep == 0:
        print("Determined that there was no correlation between X and Y; unable to estimate the canonical vectors")
        return None

    vals = vals[0:keep]
    Htildehat = vecs[:,0:keep]
    
    S = np.sqrt(vals)
    That = B @ Htildehat @ np.diag(1 / S)

    Hhat = SigmaYhat_minusonehalf @ Htildehat

    return {"That": That, "Hhat": Hhat, "S": S, "B": B}

def compare_vector_estimates_with_truth(r, thetas, That,normalize_flag = True):
    
    p = thetas.shape[0]
    for i in range(r):
        print(i)
        # theta hat
        theta = np.array(That[:, i])
        if normalize_flag:
            norm_inv = np.linalg.norm(theta)**(-1)
            theta = theta * norm_inv

        # theta true
        thetatrue = np.array(thetas[:, i])
        if normalize_flag:
            norm_inv = np.linalg.norm(thetatrue)**(-1)
            thetatrue = thetatrue * norm_inv
            
        # flip sign if necessary
        if np.sum(thetatrue * theta) < 0:
            theta = -theta

        plt.figure()
        plt.plot(range(1, p + 1), theta, 'bo', label='estimated theta')
        plt.plot(range(1, p + 1), thetatrue, 'go', label='truth')
        plt.ylim(-1, 1)
        plt.title(f'theta {i+1} versus truth')
        plt.legend(loc='lower right')
        plt.show()

def average_f_one_score(A,B,r):

    if A is None or B is None:
        return 0
    
    assert A.shape == B.shape, "Matrices must have the same shape"

    f1_scores = []
    for col in range(r):
        # Binarize the columns of A and B
        A_col = (A[:, col] != 0).astype(int)
        B_col = (B[:, col] != 0).astype(int)
    
        f1 = f1_score(A_col, B_col)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


# we can also consider dividing this by r. The subspace score is equivalent to the sum of the squares of the cosines
# of the angles between the subspaces spanned by the first r columns of A and B.
def subspace_alignment_score(A,B,r):
    
    if A is None or B is None:
        # this is the lowest possible score, meaning the subspaces had no intersection at all
        return 0
    
    assert A.shape == B.shape, "Matrices must have the same shape"
    A_r = A[:, 0:r]  # Take the first r columns of A
    B_r = B[:, 0:r]  # Take the first r columns of B
    P_A = A_r@ np.linalg.inv(A_r.T@A_r)@A_r.T  # Projection matrix onto the column space of A
    P_B = B_r@ np.linalg.inv(B_r.T@B_r)@B_r.T # Projection matrix onto the column space of B
    
    return np.trace(P_A @ P_B)





def compute_correlations(X,Y,r):
    dim = min(X.shape[1], Y.shape[1], r)
    if dim ==1:
        return [np.corrcoef(X[:,0], Y[:,0])[0,1]]
    else:
        cca_output = CCA(np.cov(X, rowvar=False), np.cov(Y, rowvar=False), np.cov(X, Y, rowvar=False)[:X.shape[1], X.shape[1]:])
    if cca_output is None:
        return None
    else:
        return cca_output["S"][:dim]
    

def train_deep_CCA(model, optimizer, epochs, device,train_loader,print_flag=False,graph_flag=False):
    
    model.train()
    #zhats = []  #can set to None if we don't want to store these
    zhats = None
    losses = []
    encoder_weights = None #not storing these for now
    encoder_logvars = None #not storing these for now[]
    for epoch in range(epochs):
        overall_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Putting Y first because of how cca-zoo coded DVCCA where there is only one encoder
            loss = model.training_step({"views": [y, x]}, batch_idx)  # returns a Tensor loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            overall_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0) # Clip the gradients to prevent exploding gradients
        if print_flag:
            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx+1))

        losses.append(overall_loss / (batch_idx + 1))
            # Calculate the gradient norm for the Y_decoder_mean parameters
            # and store it for plotting
            #total_norm = 0
            #for param in model.decoder_mean.parameters():
            #    if param.grad is not None:
            #        param_norm = param.grad.data.norm(2)
            #        total_norm += param_norm.item() ** 2
            #total_norm = total_norm ** 0.5
            
            # store the loss at each epoch
            

            # Store parameter values of the encoder weights at each epoch
            #encoder_weights.append(model.encoder_mean[0].weight.detach().cpu().numpy())

            # Get the diagonal of the weight matrix of encoder_logvar (assumes it's of the form nn.Sequential(Linear(q, d)))
            #encoder_logvars.append(np.diag(model.encoder_logvar[0].weight.detach().cpu().numpy().copy()))
            
            # Store the Zhat values for each epoch
            #zhats.append(model.encoder_mean(train_loader.dataset.Y.to(device)).detach().cpu().numpy().copy())

    # here we're returning things that are not just contained in the model; they are things that we can plot later to see how training went.
    return {
        "zhats": zhats,
        "losses": losses,}
    #     "encoder_weights": encoder_weights,
    #     "encoder_logvars": encoder_logvars,
    # }


def run_deep_CCA(model,dataset,batch_size,num_epochs,lr,device,print_flag,
                          shuffle_flag = True,graph_flag = False, LOAD_PATH = None):
    Y = dataset.Y
    X = dataset.X
    # If batch_size is None, we use the full dataset as a single batch.
    if batch_size is None:
        batch_size = Y.shape[0]
    
    optimizer = Adam_mini(
            named_parameters = model.named_parameters(),
            lr = lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,verbose=False)
    #optimizer.wv_names = {} 

    if LOAD_PATH is not None:
        checkpoint = torch.load(LOAD_PATH, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'],strict = False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #optimizer = Adam(model.parameters(), lr=lr)
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    #Standard gradient descent
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dataset = XYDataset(X,Y)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle_flag)
    start_time = time.time()
    train_dict = train_deep_CCA(model, optimizer, epochs=num_epochs, device=device,train_loader=train_loader,print_flag=print_flag,
                                graph_flag=graph_flag)
    end_time = time.time()
    num_epochs_ran = len(train_dict["losses"])
    print(f"Average training time per epoch: {(end_time - start_time)/num_epochs_ran:.3f} seconds")
    model.eval()
    # with torch.no_grad():
    #     # after training the model, we can evaluate it on the training data
    #     U,V = model([Y.float().to(device), X.float().to(device)])
    # #model.fit([X.float().to(device), Y.float().to(device)])
    
    
    return {"model":model,"optimizer":optimizer,"train_dict":train_dict,}
                
                #"encoder_X":model.encoders[1],"encoder_Y":model.encoders[0],"U":U,"V":V}
                #"Sigma_Z_est":Sigma_Z_est,"Zhat":Zhat,
                #"lambdas_est":lambdas_est,"inactive_latent_dims": zero_var_idx if zero_var_idx.size > 0 else None}

def generate_data(seed,N= 300, p = 20, num_pixels = 20,lams = [0.9, 0.7]):

    from rings_and_discs import utils_for_rings_and_discs as gen_and_plot_utils
    #torch.manual_seed(16)
    np.random.seed(seed)
    q = 4  # Dimension of the latent space Z. d here is 4 because we have four parameters which determine the ring and disc images.
    d = 2 # number of canonical vectors
    #p = 30  # Length of vector X
    image_noise_var = 0.003  # Variance of Gaussian noise for images
    #num_pixels = 20  # Image resolution

    # radius of the hole in the ring
    ring_r_1_lower_bd = 0.1 # choosing 0 so we also get discs in the dataset
    ring_r_1_upper_bd = 0.5

    # thickness of the ring
    disc_r_2_lower_bd = 0.1
    disc_r_2_upper_bd = 0.5

    # contrast parameter
    constrast_param_lower_bd = 0.3
    constrast_param_upper_bd = 1

    # aspect ratio parameter
    aspect_ratio_lower_bd = .7
    aspect_ratio_upper_bd = 1.3
    #lams = [0.9, 0.7]  # Canonical correlations

    # canonical vectors
    thetas = np.zeros((p,d))
    # thetas[0,0] = 1
    #thetas[1,1] = 1
    thetas[2,0] = 1
    thetas[3,1] = 1

    etas = np.zeros((q,d))
    #etas[0,0] = 1
    #etas[1,1] = 1
    etas[2,0] = 1
    etas[3,1] = 1


    output_dict = gen_and_plot_utils.generate_images_and_X_known_cvs_dataset(N, p, q, lams, thetas, etas,num_pixels,image_noise_var,
                            ring_r_1_lower_bd, ring_r_1_upper_bd,
                            disc_r_2_lower_bd, disc_r_2_upper_bd,
                            constrast_param_lower_bd, constrast_param_upper_bd,
                            aspect_ratio_lower_bd, aspect_ratio_upper_bd)

    Z = output_dict["Z"]
    #images = output_dict["images"]
    Z_standard = output_dict["Z_standard"]
    X = output_dict["X"]
    Y = output_dict["Y"]
    #W = output_dict["W"]
    #gen_and_plot_utils.plot_image_dataset(images, labels = [1]*N)


    # # Compute correlation between Z[:, 0] and X[:, 0]
    # correlation_Z1_X1 = np.corrcoef(Z[:, 0], X[:, 0])[0, 1]
    # print(f"Correlation between Z[:, 0] and X[:, 0]: {correlation_Z1_X1:.4f}")

    # # Compute correlation between Z[:, 1] and X[:, 1]
    # correlation_Z2_X2 = np.corrcoef(Z[:, 1], X[:, 1])[0, 1]
    # print(f"Correlation between Z[:, 1] and X[:, 1]: {correlation_Z2_X2:.4f}")

    N_val = 2500  # Total number of images in validation dataset
    output_dict_val = gen_and_plot_utils.generate_images_and_X_known_cvs_dataset(N_val, p, q, lams, thetas, etas,num_pixels,image_noise_var,
                            ring_r_1_lower_bd, ring_r_1_upper_bd,
                            disc_r_2_lower_bd, disc_r_2_upper_bd,
                            constrast_param_lower_bd, constrast_param_upper_bd,
                            aspect_ratio_lower_bd, aspect_ratio_upper_bd)

    Z_val = output_dict_val["Z"]
    #images_val = output_dict_val["images"]
    Z_standard_val = output_dict_val["Z_standard"]
    Y_val = output_dict_val["Y"]
    X_val = output_dict_val["X"]

    return {"Y": Y,
            "X": X,
            "Z": Z,
            "Z_standard": Z_standard,
            "Y_val": Y_val,
            "X_val": X_val,
            "Z_val": Z_val,
            "Z_standard_val": Z_standard_val
            }