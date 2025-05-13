from modules.conv2d import Conv2D 
from modules.relu import ReLU 
from modules.maxpool2d import MaxPool2D
from modules.flatten import Flatten
from modules.dense import Dense
from modules.softmax import Softmax
from modules.batchnorm import BatchNorm2D
from modules.dropout import Dropout
from models.basemodel import BaseModel

class AlexNet_CIFAR100(BaseModel): 
     def __init__(self, conv_algo=0):
         print("Building AlexNet for CIFAR-100")
         layers = [] 
         layers.append(Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo))
         layers.append(BatchNorm2D(64))
         layers.append(ReLU())
         layers.append(MaxPool2D(kernel_size=2, stride=2))  # 32x32 → 16x16
         
         layers.append(Conv2D(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo))
         layers.append(BatchNorm2D(192))
         layers.append(ReLU())
         layers.append(MaxPool2D(kernel_size=2, stride=2))  # 16x16 → 8x8
         
         layers.append(Conv2D(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo))
         layers.append(BatchNorm2D(384))
         layers.append(ReLU())

         layers.append(Conv2D(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo))
         layers.append(BatchNorm2D(256))
         layers.append(ReLU())

         layers.append(Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo))
         layers.append(BatchNorm2D(256))
         layers.append(ReLU())
         layers.append(MaxPool2D(kernel_size=2, stride=2))  # 8x8 → 4x4

         layers.append(Flatten())
         layers.append(Dense(256 * 4 * 4, 1024))
         layers.append(ReLU())
         layers.append(Dropout(0.5))  # Dropout layer with 50% probability
         
         layers.append(Dense(1024, 512))
         layers.append(ReLU())
         layers.append(Dropout(0.5))  # Dropout layer with 50% probability
         layers.append(Dense(512, 100))  # 100 classes for CIFAR-100
         layers.append(Softmax())

         super().__init__(layers)

