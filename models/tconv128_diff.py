import torch
import torch.nn as nn
import pytorch_lightning as pl
from task.diff_separation import DiffSeparation


class TConv128Diff(DiffSeparation):
    def __init__(self, input_channels, output_channels, task_args):
        super().__init__(**task_args)
        self.conv1u = nn.Conv1d(input_channels, 16, 11)
        self.conv2u = nn.Conv1d(16, 32, 9)
        self.conv3u = nn.Conv1d(32, 64, 7)
        self.conv4u = nn.Conv1d(64, 128, 5)
        
        self.conv4d = nn.ConvTranspose1d(128, 64, 5)
        self.conv3d = nn.ConvTranspose1d(64, 32, 7)
        self.conv2d = nn.ConvTranspose1d(32, 16, 9)
        self.conv1d = nn.ConvTranspose1d(16, output_channels, 11)
        

    def forward(self, x_t, waveform, t):
        # x (batch , 1, 2, len) --> (batch, 10, len)   
        x = torch.concat(
            (waveform, x_t),
            dim=1)
        
        x = self.conv1u(x)
        x = self.conv2u(x)
        x = self.conv3u(x)
        x = self.conv4u(x)
        
        x = self.conv4d(x)
        x = self.conv3d(x)
        x = self.conv2d(x)
        x = self.conv1d(x)
        
        return x # (batch, 8, len)
