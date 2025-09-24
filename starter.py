import torch 
from transformers import  AutoImageProcessor
from Swing_regression import *
from CustomImageDataset import *
import pickle as pkl
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiStepLR

import os
from parser import get_parser
from sklearn.model_selection import KFold
from train import train, evaluate, training_configuration
import numpy as np  

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from load_dataset import load_dataset


def main(args):

    dataset_name=args.dataset_name
    dataset_path=f"{args.data_path}{dataset_name}"


    training_config={
        "batch_size": args.batch_size, 
        "lr":args.lr             
    }

    if args.hyperparam_ft:
        #create a Configuration file
        config={
            "batch_size": tune.choice([4, 8, 16, 32, 64] ) , 
            "lr": tune.loguniform(1e-4, 1e-6), 
            "dataset_name" : dataset_name, 
            "dataset_path": dataset_path, 
            "verbose": True, 
        }


        scheduler = ASHAScheduler(
                metric="loss",
                mode="min",
                max_t=10,
                grace_period=1,
                reduction_factor=2
            )

        reporter = CLIReporter(
                parameter_columns=["lr", "batch_size"],
                metric_columns=["loss", "spearman_corr", "pearson_corr"]
        )

        result = tune.run(
                training_configuration,
                config=config,
                scheduler=scheduler,
                progress_reporter=reporter, 
                num_samples=20, 
                resources_per_trial={"cpu": 2, "gpu": 1}, 
                verbose=0
        )


        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final loss: {}".format(best_trial.last_result["loss"]))


        training_config=best_trial.config

    print("*"*20, "START TRAINING","*"*20 )

    #create the network
    model=Swin_regression()
    if args.load_model:
        model.load_state_dict(torch.load("model.pth"))
    #Create dataloader
    train_set, val_set, test_set=load_dataset(dataset_name, dataset_path)
    train_set= add_datasets(train_set, val_set)
    
    D_train=DataLoader(train_set, batch_size=training_config["batch_size"], shuffle=True)
    D_test=DataLoader(test_set , shuffle=True)
 
    trained_model=train(model, args.epochs, D_train, dataset_name, batch_size= training_config["batch_size"], lr= training_config["lr"], verbose=True )

    # scheduler =MultiStepLR(optimizer, milestones=[200, 100, 60, 20, 1.0 , 0.7, 0.6] , gamma=0.1, last_epoch=-1)

    print("*"*20, "START TEST",  "*"*20)
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256", use_fast=True)
    pearson_corr, spearman_corr, loss=evaluate(trained_model,D_test,image_processor)

    print(f"Pearson correlation: {pearson_corr},Spearman correlation {spearman_corr}")

    if args.save_model:
        torch.save(model.state_dict(),"model_weight.pth")

    if False:
        # define loss function
        loss_fn=torch.nn.MSELoss()

        #DEFINE THE TRAINING ROUTINE
        epochs=5

        # kfold cross validation
        model.train()

        ###make this a function
        batch_sizes=[4, 8, 16, 32, 64]
        splits=KFold(n_splits=5,  shuffle=True, random_state=42)
        metrics=[] 

        #Perform Kfold Cross Validation
        
        for bs in batch_sizes:
            pearson_corr_avg=0.0
            spearman_corr_avg=0.0
            
            print("*"*20, f"batch parameter {bs} ", "*"*20)
            for i , (train_index, val_index) in enumerate(splits.split(train_set)):

                D_train=DataLoader(Subset(train_set, train_index),batch_size=bs, shuffle=True)
                D_val=DataLoader(Subset(train_set,val_index ), shuffle=True)

                model_trained=train(model, epochs, loss_fn, D_train,image_processor,  dataset_name,verbose=True)

                #eval the model 
                model_trained.eval()
                with torch.no_grad():
                    pearson_corr, spearman_corr =evaluate(model_trained, D_val, image_processor)

                pearson_corr_avg+=pearson_corr
                spearman_corr_avg+=spearman_corr

            pearson_corr_avg/=5
            spearman_corr_avg/=5

            metrics.append((pearson_corr_avg, spearman_corr_avg))

        pearsons = [p for p, _ in metrics]
        idx = np.argmax(pearsons)

        best_pearson, best_spearman = metrics[idx]
        print(f"Index: {idx}, Pearson: {best_pearson}, Spearman: {best_spearman}")

        #save the model
        os.mkdir("trained_networks/")

        torch.save(model.state_dict, "trained_networks/ViT.pt")

        #test the model
        model.eval()

        with torch.no_grad():
            evaluate(model, test_loader, image_processor)

if __name__=="__main__":

    args = get_parser().parse_args()
    print(args)
    main(args)