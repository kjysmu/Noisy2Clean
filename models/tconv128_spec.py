import torch
import torch.nn as nn
import pytorch_lightning as pl
from task.separation import SeparationSpec

class TConv128Spec(SeparationSpec):
    # TODO: Variable Parent class 
    # https://stackoverflow.com/questions/56746709/can-i-choose-the-parent-class-of-a-class-from-a-fixed-set-of-parent-classes-cond    
    def __init__(self, input_channels, output_channels, task_args):
        super().__init__(**task_args)
        
        self.conv1u = nn.Conv1d(input_channels, 16, 11, padding=0)
        self.conv2u = nn.Conv1d(16, 32, 9, padding=0)
        self.conv3u = nn.Conv1d(32, 64, 7, padding=0)
        self.conv4u = nn.Conv1d(64, 128, 5, padding=0)
        
        self.conv4d = nn.ConvTranspose1d(128, 64, 5, padding=0)
        self.conv3d = nn.ConvTranspose1d(64, 32, 7, padding=0)
        self.conv2d = nn.ConvTranspose1d(32, 16, 9, padding=0)
        self.conv1d = nn.ConvTranspose1d(16, output_channels, 11, padding=0)

    def forward(self, x):
        
        x = self.conv1u(x)
        x = self.conv2u(x)
        x = self.conv3u(x)
        x = self.conv4u(x)
        
        x = self.conv4d(x)
        x = self.conv3d(x)
        x = self.conv2d(x)
        x = self.conv1d(x)
        
        return x # (batch, 8, len)