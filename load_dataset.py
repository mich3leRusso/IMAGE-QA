import os 
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from CustomImageDataset import *

def load_dataset(dataset_name, dataset_path):
    
    labels=[] 
    images_path=""
    img_noise=[]

    if dataset_name== "KADID10K":
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
        folder_order=["jp2k", "jpeg", "wn", "gblur","fastfading" ]
        images_path=dataset_path
        files=os.listdir(dataset_path)

        mat_dmos=scipy.io.loadmat(f"{dataset_path}/dmos.mat")
        
        dmos=mat_dmos["dmos"][0] 
        orgs=mat_dmos["orgs"][0]  
        

        folder_index=0
        change=0

        for row_index in range(1,len(dmos)+1):
            
            
                file=folder_order[folder_index] 
                path=os.path.join(dataset_path,file)
                
                images=os.listdir(path)
                number_images=len(images)-2

                labels.append(dmos[row_index-1]) 
                image_index=row_index-change
                image_name=f"{file}/img{image_index}.bmp"
                img_noise.append(image_name)

                if row_index+1-change>number_images:
                    folder_index+=1
                    change=row_index
            
        
        v_min = min(labels)
        v_max = max(labels)
        
        # Normalize to [0, 5]
        if v_min == v_max:
            # Handle constant vector case
            labels = [2.5 for _ in labels]  # midpoint of [0,5]
        else:
            labels = [5 * (x - v_min) / (v_max - v_min) for x in labels]


                


    elif dataset_name=="TID2013":
        images_path=dataset_path

        with open(f"{dataset_path}/mos_with_names.txt", "r") as mos_names:
            for line in mos_names.readlines():
                labels.append(float(line.split()[0]))
                img_noise.append(f"distorted_images/{line.split()[1]}")  
        
        ref_image=os.listdir(f"{dataset_path}/reference_images")
        
        #labels for the reference images
        new_vector = [0.0] * len(ref_image)
        labels=labels+new_vector
        
        #extract reference images 
        for i in range(len(ref_image)):
            ref_image[i]=f"reference_images/{ref_image[i]}" 

        img_noise=img_noise+ref_image
        
        v_min = min(labels)
        v_max = max(labels)
        
        # Normalize to [0, 5]
        if v_min == v_max:
            # Handle constant vector case
            labels = [2.5 for _ in labels]  # midpoint of [0,5]
        else:
            labels = [5 * (x - v_min) / (v_max - v_min) for x in labels]

    else:
            print("no other dataset at the moment")   

    #split_dataset
    split_test=0.2
    split_val=0.1

   # First split: Train vs. Temp (Validation + Test)
    img_noise_train, img_noise_temp, labels_train, labels_temp = train_test_split(
        img_noise, labels, test_size=(split_test + split_val), 
        shuffle=True
    )

    # Compute proportions for validation relative to the remaining data
    valid_ratio = split_val / (split_val + split_test)

    # Second split: Validation vs. Test
    img_noise_valid, img_noise_test, labels_valid, labels_test = train_test_split(
        img_noise_temp, labels_temp, test_size=(1 - valid_ratio), 
        shuffle=True
    )

    # Create dataset objects
    train_set = CustomImageDataset(images_path, img_noise_train, labels_train)
    valid_set = CustomImageDataset(images_path, img_noise_valid, labels_valid)
    test_set = CustomImageDataset(images_path, img_noise_test, labels_test)

    return train_set, valid_set, test_set