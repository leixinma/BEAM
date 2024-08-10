import sys
import os
import pathlib
from pathlib import Path
import numpy as np
import time
import datetime
import ast
import csv
from torch import nn 
import torch
from PIL import Image




class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

                                     
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size= 3,stride = 2, padding=1,output_padding= 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size= 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size= 3, padding=1),
            nn.ReLU(),
            
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

