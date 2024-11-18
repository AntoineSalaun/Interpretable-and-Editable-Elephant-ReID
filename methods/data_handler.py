from pathlib import Path
from PIL import Image
import pandas as pd
import random

from IPython.display import display
import matplotlib.pyplot as plt
import torch.utils.data as data


class EleHandler(data.Dataset):
    def __init__(self, dataset_dir = Path('/archive/vision/beery/animal_reid/datasets/elephants_zooniverse'), dictonary_path = Path('/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary.csv') , 
                  num_elephants = None, transform = None):
        
        self.dataset_dir = dataset_dir
        self.num_elephants = num_elephants
        self.image_dir = Path(dataset_dir) / 'images'
        self.transform = transform

        # the transform is now useless as we preprocessed the images
        import torchvision.transforms as transforms

        if transform == 'MegaDescriptor-elephant':
            self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        self.dictonary = pd.read_csv(dictonary_path)        

    def __len__(self):
        return len(self.dictonary)
    

    def __getitem__(self, idx):

        image_path = self.image_dir / self.dictonary.iloc[idx]['preprocessed_image_path']
        image = Image.open(image_path).convert('RGB')
        
        import torchvision.transforms as transforms
        image = transforms.ToTensor()(image)

        # Not needed as we have preprocessed the images
        #if self.transform is not None:
        #    image = self.transform(image)

        subject_id = self.dictonary.iloc[idx]['subject_id']
        ele_id = self.dictonary.iloc[idx]['ele_id']
        identified = [True if self.dictonary.iloc[idx]['#season'] == 'EFA_IDI' else False]
        #anonymized_capture_id = self.dictonary[idx]['anonymized_capture_id']
        
        subject_SEEK = self.dictonary.iloc[idx]['subject-SEEK']
        ele_SEEK = self.dictonary.iloc[idx]['ele-SEEK']

        return image, subject_id, ele_id, identified, subject_SEEK, ele_SEEK
    
    def print_image(self, idx, print_with_transform = True):
        image_path = self.image_dir / self.dictonary.iloc[idx]['image']
        print(image_path)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None and print_with_transform is True:
            image = self.transform(image)
        
        # Display the image using matplotlib
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        
        subject_id = self.dictonary.iloc[idx]['subject_id']
        ele_id = self.dictonary.iloc[idx]['ele_id']
        identified = self.dictonary.iloc[idx]['#season'] == 'EFA_IDI'
        subject_SEEK = self.dictonary.iloc[idx]['subject-SEEK']
        ele_SEEK = self.dictonary.iloc[idx]['ele-SEEK']

        label = {
            'subject_id': subject_id,
            'ele_id': ele_id,
            'identified': identified,
            'subject_SEEK': subject_SEEK,
            'ele_SEEK': ele_SEEK
        }
        print(label)
        return image, label
    

if __name__ == "__main__":
    dataset_dir = Path('/archive/vision/beery/animal_reid/datasets/elephants_zooniverse')
    EleHandle = EleHandler(dataset_dir)
    # Select a random index
    random_idx = random.randint(0, len(EleHandle) - 1)
    EleHandle.print_image(random_idx)
    EleHandle.print_image(0)