import torch 
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter
import copy 
from torch.optim import AdamW
from Swing_regression import Swin_regression
from torch.utils.data import DataLoader
from ray import tune
from transformers import  AutoImageProcessor
from load_dataset import load_dataset
import os
from ray import tune

def train(model, epochs,  train_loader, dataset_name, batch_size, lr=0.0001,  verbose=False):
    
 
    device="cpu"
    if torch.cuda.is_available():
        device="cuda" 

    model=model.to(device)

    loss_fn=torch.nn.MSELoss()    
    optimizer= AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256", use_fast=True)
    
    tb_writer = SummaryWriter(f"runs/SWIN_train_{dataset_name}_{batch_size}")
    iteration_number=0
    
    model.train()

    for epoch in range(epochs):

        running_loss=0.0
        epoch_loss=0.

        print("*"*15, f"start epoch: {epoch}", "*"*15)
         
        for i, (X, y) in enumerate(train_loader):
            
        
            optimizer.zero_grad()
            
            X=image_processor(X, return_tensors="pt").pixel_values

            X=X.to(device)
            
            output=model(X)

            loss=loss_fn(output.squeeze(1),y.to(device, dtype=torch.float32))
            loss.backward()
            
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            epoch_loss +=loss.item()

            if i % 16== 0 and i!=0:
                last_loss = running_loss / 16 # loss per batch
                if verbose:
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = iteration_number
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        
        iteration_number+=1
        
    #save tensor table 
    tb_writer.flush()
    tb_writer.close()

    return model

def evaluate(model, loader, image_processor):
    model.eval()
    preds, targets_all = [], []
    device="cpu"

    if torch.cuda.is_available():
        device="cuda"

    loss_fn=torch.nn.MSELoss()
    loss=0.0

    model=model.to(device)
    with torch.no_grad():
        for inputs, targets in loader:
            
            inputs=image_processor(inputs, return_tensors="pt").pixel_values

            inputs=inputs.to(device)
            outputs = model(inputs).squeeze(1)
            targets = torch.tensor([targets], dtype=outputs.dtype, device=outputs.device)
            loss+=loss_fn(outputs, targets)
            preds.append(outputs)
            targets_all.append(targets)

            
    loss/=len(loader)

    preds = torch.cat(preds).cpu().numpy()
    targets_all = torch.cat(targets_all).cpu().numpy()
    
    # Pearson and Spearman correlations
    pearson_corr, _ = pearsonr(preds, targets_all)
    spearman_corr, _ = spearmanr(preds, targets_all)

    return pearson_corr, spearman_corr, loss

def training_configuration(config):

    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    model=Swin_regression().to(device)

    optimizer=AdamW(model.parameters(), lr=config["lr"],weight_decay=0.01)


    train_dataset, val_dataset, test_dataset = load_dataset(config["dataset_name"], config["dataset_path"] )

    train_loader=DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True )
    
    loss_fn=torch.nn.MSELoss()

    if config["verbose"]:
        os.makedirs("/davinci-1/home/micherusso/PycharmProjects/IMAGE-QA/logs/", exist_ok=True)
        with open(f"/davinci-1/home/micherusso/PycharmProjects/IMAGE-QA/logs/SWIN_train_{config["dataset_name"]}_{config["batch_size"]}_{config["lr"]}.txt","w" ) as logs_file:
                logs_file.write(f"Configuration parameters Dataset_Name: {config["dataset_name"]},  Bath_Size: {config["batch_size"]} Learning Rate: {config["lr"]}")


    epochs=5

   
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256", use_fast=True)
    
    os.makedirs(f"runs/SWIN_train_{config["dataset_name"]}_{config["batch_size"]}_{config["lr"]}",exist_ok=True )
    tb_writer = SummaryWriter(f"/davinci-1/home/micherusso/PycharmProjects/IMAGE-QA/runs/SWIN_train_{config["dataset_name"]}_{config["batch_size"]}_{config["lr"]}")
    
    model.train()
    iteration_number=0
    for epoch in range(epochs):

        running_loss=0.0
        epoch_loss=0.
        
        if config["verbose"]:
            with open(f"/davinci-1/home/micherusso/PycharmProjects/IMAGE-QA/logs/SWIN_train_{config['dataset_name']}_{config['batch_size']}_{config['lr']}.txt", "a") as logs_file:
                
                msg = f"{'*'*15} start epoch: {epoch} {'*'*15}\n"
                logs_file.write(msg)   
                
       
        for i, (X, y) in enumerate(train_loader):
            
        
            optimizer.zero_grad()
            
            X=image_processor(X, return_tensors="pt").pixel_values

            X=X.to(device)
            
            output=model(X)

            loss=loss_fn(output.squeeze(1),y.to(device, dtype=torch.float32))
            loss.backward()
            
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            epoch_loss +=loss.item()
            
            if i % 16== 0 and i!=0:
                last_loss = running_loss / 16 # loss per batch
                tb_x = iteration_number

                if config["verbose"]:
                    with open(f"/davinci-1/home/micherusso/PycharmProjects/IMAGE-QA/logs/SWIN_train_{config['dataset_name']}_{config['batch_size']}_{config['lr']}.txt", "a") as logs_file:
                        logs_file.write('batch {} loss: {}'.format(i + 1, last_loss))

                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
            iteration_number+=1
   #evaluate the model 
    tb_writer.flush()
    tb_writer.close()

    model.eval() 
        
    with torch.no_grad():
        pearson_corr, spearman_corr, val_loss =evaluate(model, val_dataset, image_processor)

        tune.report( 

            metrics={
            "loss": val_loss.item(),
            "spearman_corr": spearman_corr,
            "pearson_corr": pearson_corr

            }
        )


    return 

