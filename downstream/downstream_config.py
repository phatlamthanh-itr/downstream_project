import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from downstream.downstream_loss import CombinedBCEDiceLoss, DiceLoss

class DownStream_ExpConfig():
    def __init__(self, 
                 # model parameters
                 subseq_size : int, 
                 label_size : int,
                 feature_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 output_size: int,
                 # model training parameters
                 epochs=50, lr=0.01, batch_size=64, save_epochfreq=15, lr_step_size=5, gamma_decay=0.1,
                 save_pred_epoch = 2,                                   # Save model predictions after save_pred_epoch epochs
                 # experiment params
                 loss = CombinedBCEDiceLoss(alpha=0.5),
                 seed=1234,
                 run_dir = 'downstream/out/checkpoints',                # Directory to save checkpoints
                 log_dir = 'downstream/out/logs',
                 save_pred_dir = 'downstream/out/predictions',        # Directory to save prediction npy for annotation
                 save_beat_dir = 'val_prediction_atr'
                 ):
        
        self.subseq_size = subseq_size
        self.label_size = label_size
        self.feature_dim = feature_dim
        self.num_layers = num_layers    # Number of LSTM layers
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.epochs = epochs
        self.lr = lr 
        self.batch_size = batch_size
        self.save_epochfreq = save_epochfreq
        self.save_pred_epoch = save_pred_epoch
        self.seed = seed
        self.device = None
        self.input_dims = None
        self.loss = CombinedBCEDiceLoss(alpha=0.5)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(save_pred_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_beat_dir, exist_ok=True)
        self.run_dir = run_dir
        self.log_dir = log_dir
        self.save_pred_dir = save_pred_dir
        self.save_beat_dir = save_beat_dir
        self.lr_step_size = lr_step_size
        self.lr_gamma = gamma_decay

    def set_device(self, device):
        self.device = device
    def set_inputdims(self, dims):
        self.input_dims = dims
    def set_rundir(self, run_dir):
        self.run_dir = run_dir



downstream_config = DownStream_ExpConfig(
        subseq_size = 2500,
        label_size = 2500,
        feature_dim= 320,
        num_layers=1,
        hidden_dim = 32,
        output_size = 80,
        epochs = 20,
        lr = 0.01,
        batch_size = 32,
        save_epochfreq = 10,
        lr_step_size=5,
        gamma_decay=0.1,
        seed = 1234,
    )