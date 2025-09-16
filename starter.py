import torch 
from transformers import Swinv2Config, Swinv2Model, AutoImageProcessor
import os
import pandas as pd 
from PIL import Image
from Swing_regression import *
from CustomImageDataset import *
from torch.utils.data import DataLoader
from torch.optim import  Adam
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchview import draw_graph
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
import scipy.io 
from parser import get_parser

dataset_name="LIVE"
dataset_path=f"/archive/HPCLab_exchange/MORTE_AL_DAVINCI/databaserelease2"
# dataset_path=f"/archive/HPCLab_exchange/MORTE_AL_DAVINCI/KADID10/{dataset_name}"
image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")


labels=[] 
images_path=""
img_noise=[]

if dataset_name== "kadid10k":
    #open the dataset
    files=os.listdir(dataset_path)
    
   # label_file=""
    
    for file in files:

        if file.endswith(".csv"):
            
            labels_dataset=pd.read_csv(f"{dataset_path}/{file}")   
            img_noise=labels_dataset["dist_img"].values.tolist()
            labels=labels_dataset["dmos"].values.tolist()
        
        else:
            images_path=f"{dataset_path}/{file}"

    

elif dataset_name=="LIVE":

    images_path=dataset_path
    files=os.listdir(dataset_path)

    mat=scipy.io.loadmat(f"{dataset_path}/refnames_all.mat")
    
    for file in files:
        
        if file.endswith(".mat") or file.endswith(".txt"):
            continue
        
        else:
            path=os.path.join(dataset_path,file)
            images=os.listdir(path)
            
            if "info.txt" in images:

                with open(path+"/info.txt","r") as ref:
                    for line in ref.readlines():
                        
                        if line.split():
                            labels.append(float(line.split()[2]))
                            img_noise.append(f"{file}/{line.split()[1]}") 

            else:
                continue


else:
    print("no other dataset at the moment")   

#create a dataloader 

dataset=CustomImageDataset(images_path, img_noise,labels)
 

#Create dataloader

dataloader=DataLoader(dataset, batch_size=16, shuffle=True)

# for X, y in dataloader:
#     print(X.shape)
#     print(y )
#     input()

#create the network
model=Swin_regression()

if torch.cuda.is_available():
 
    model=model.to("cuda")

#import optimizer 
optimizer= Adam(model.parameters(), lr=0.01, weight_decay=0.01)
scheduler =MultiStepLR(optimizer, milestones=[1.0 , 0.7, 0.6] , gamma=0.25, last_epoch=-1)
# define loss function

loss_fn=torch.nn.MSELoss()

#DEFINE THE TRAINING ROUTINE
epochs=30

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tb_writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

last_loss=0.0
epoch_losses=[] 

# kfold cross validation
model.train()

for epoch in range(epochs):

    running_loss=0.0
    epoch_loss=0.
    print("*"*15, f"start epoch: {epoch}", "*"*15)

    
    for i, (X, y) in enumerate(dataloader):
        
    
        optimizer.zero_grad()
        
        X=image_processor(X, return_tensors="pt").pixel_values

        X=X.to("cuda")
        
        output=model(X)

        loss=loss_fn(output.squeeze(1),y.to("cuda", dtype=torch.float32))
        loss.backward()
        
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        epoch_loss +=loss.item()

        if i % 30 == 0 and i!=0:
            last_loss = running_loss / 30 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    scheduler.step()
    epoch_losses.append(epoch_loss/len(dataloader))

plt.plot(epoch_losses)
plt.show()

