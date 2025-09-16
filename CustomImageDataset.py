import os 
from torchvision.io import decode_image
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms
class CustomImageDataset(Dataset):
    def __init__(self,image_path,image_name ,labels):
        super().__init__()

        self.root_folder=image_path
        self.image_names=image_name
        self.labels=labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        image_name=self.image_names[index] 
        image_path=os.path.join(self.root_folder, image_name)
       
        if image_path.endswith("jpeg") or image_path.endswith("png"):
            image=decode_image(image_path) 

        else:
            
            transform = transforms.Compose([transforms.PILToTensor()])
            image= Image.open(image_path)
            image=image.resize(size=(256,256))
            image=transform(image)
        
        
        label=self.labels[index] 
    
        return image, label

