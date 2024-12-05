from pathlib import Path
from PIL import Image
import pandas as pd
import random
import torch

from IPython.display import display
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms

from seek_code import SEEK


class EleHandler(data.Dataset):
    def __init__(self, dataset_dir=Path('/archive/vision/beery/animal_reid/datasets/elephants_zooniverse'), 
                 dictonary_path=Path('/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary.csv'), 
                 num_elephants=None,
                 transform=None,
                 EFA_IDI_only=False):
        
        self.dataset_dir = dataset_dir
        self.num_elephants = num_elephants
        self.image_dir = Path(dataset_dir) / 'images'
        self.transform = transform

        # Preprocessing transformation
        if transform == 'MegaDescriptor-elephant':
            import torchvision.transforms as transforms
            self.transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
        # Load the dictionary and create ele_id mapping
        self.dictonary = pd.read_csv(dictonary_path)
        self.ele_id_to_label = {ele_id: i for i, ele_id in enumerate(sorted(self.dictonary['ele_id'].unique()))}
        self.label_to_ele_id = {i: ele_id for ele_id, i in self.ele_id_to_label.items()}

        if EFA_IDI_only:
            self.dictonary = self.dictonary[self.dictonary['#season'] == 'EFA_IDI']

    def __len__(self):
        return len(self.dictonary)
    
    def __getitem__(self, idx):
        preprocessed_image_path = self.image_dir / self.dictonary.iloc[idx]['preprocessed_image_path']
        preprocessed_image = Image.open(preprocessed_image_path).convert('RGB')
        if self.transform:
            preprocessed_image = self.transform(preprocessed_image)
        else:
            preprocessed_image = transforms.ToTensor()(preprocessed_image)

        # Load left and right ear images
        left_ear = self._load_ear(self.dictonary.iloc[idx]['left_ear_path'])
        right_ear = self._load_ear(self.dictonary.iloc[idx]['right_ear_path'])

        # Convert ele_id to numeric label
        ele_id = self.dictonary.iloc[idx]['ele_id']
        ele_id_label = self.ele_id_to_label[ele_id]

        # Other metadata
        subject_id = self.dictonary.iloc[idx]['subject_id']
        identified = self.dictonary.iloc[idx]['#season'] == 'EFA_IDI'
        subject_SEEK = self.dictonary.iloc[idx]['subject-SEEK']
        ele_SEEK = self.dictonary.iloc[idx]['ele-SEEK']

        # Encode SEEK
        subject_SEEK_1hot = SEEK(subject_SEEK).one_hot_encode()
        ele_SEEK_1hot = SEEK(ele_SEEK).one_hot_encode()

        return preprocessed_image, subject_id, ele_id_label, identified, subject_SEEK_1hot, ele_SEEK_1hot, left_ear, right_ear, subject_SEEK, ele_SEEK
    
    def _load_ear(self, ear_path):
        """Helper function to load ear images, returning zeros if path is NaN."""
        if pd.notna(ear_path):
            ear_image_path = self.image_dir / ear_path
            ear_image = Image.open(ear_image_path).convert('RGB')
            return transforms.ToTensor()(ear_image)
        return torch.zeros(3, 224, 224)

    def get_original_image(self, idx):
        image_path = self.image_dir / self.dictonary.iloc[idx]['image']
        image = Image.open(image_path).convert('RGB')
        return image
    
    def print_image(self, idx, print_with_transform=True):
        image_path = self.image_dir / self.dictonary.iloc[idx]['image']
        print(image_path)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None and print_with_transform:
            image = self.transform(image)
        
        # Display the image using matplotlib
        plt.imshow(image.permute(1, 2, 0))  # Unpermute for display (C, H, W -> H, W, C)
        plt.axis('off')
        plt.show()
        
        # Print metadata
        label = {
            'subject_id': self.dictonary.iloc[idx]['subject_id'],
            'ele_id': self.dictonary.iloc[idx]['ele_id'],
            'identified': self.dictonary.iloc[idx]['#season'] == 'EFA_IDI',
            'subject_SEEK': self.dictonary.iloc[idx]['subject-SEEK'],
            'ele_SEEK': self.dictonary.iloc[idx]['ele-SEEK']
        }
        print(label)
        return image, label

    def get_IDI_indices(self):
        return self.dictonary[self.dictonary['#season'] == 'EFA_IDI'].index.tolist()
    
    def get_IDU_indices(self):
        return self.dictonary[self.dictonary['#season'] == 'EFA_IDU'].index.tolist()
    
    def split_along_encounters(self, split_sizes=[0.7, 0.15, 0.15], hour_delta=1, identification=None):
        from sklearn.model_selection import train_test_split

        # Ensure split sizes sum to 1
        if sum(split_sizes) != 1:
            raise ValueError("Split sizes should sum to 1")
        '''
        # Standardize the #season column
        self.dictonary['#season'] = self.dictonary['#season'].str.strip().str.upper()

        # Filter dictionary based on identification
        if identification == 'EFA_IDI':
            print('filtering onlz the EFA_IDI')
            self.dictonary = self.dictonary[self.dictonary['#season'] == 'EFA_IDI']
        elif identification == 'EFA_IDU':
            print('filtering onlz the EFA_IDU')
            self.dictonary = self.dictonary[self.dictonary['#season'] == 'EFA_IDU']
        else:
            print('Sampling from both EFA_IDI and EFA_IDU')
            self.dictonary = self.dictonary

        # Verify filtering worked correctly
        print("Filtered dictionary #season unique values:", self.dictonary['#season'].unique())
        if identification == 'EFA_IDI' and 'EFA_IDU' in self.dictonary['#season'].unique():
            raise ValueError("Filtered dictionary for EFA_IDI still contains EFA_IDU.")
        elif identification == 'EFA_IDU' and 'EFA_IDI' in self.dictonary['#season'].unique():
            raise ValueError("Filtered dictionary for EFA_IDU still contains EFA_IDI.")
        '''
        # Ensure 'picture_time' is sorted
        self.dictonary.sort_values(by='picture_time', inplace=True)

        # Create a new column 'group' to group images taken within an hour
        self.dictonary['group'] = (pd.to_datetime(self.dictonary['picture_time']).diff() > pd.Timedelta(hours=hour_delta)).cumsum()

        # Split the groups into train, validation, and test sets
        unique_groups = self.dictonary['group'].unique()
        
        train_groups, temp_groups = train_test_split(unique_groups, test_size=1-split_sizes[0], random_state=42)
        val_groups, test_groups = train_test_split(temp_groups, test_size=split_sizes[2]/(1-split_sizes[0]), random_state=42)

        # Get the indices for each group
        train_indices = self.dictonary[self.dictonary['group'].isin(train_groups)].index.tolist()
        val_indices = self.dictonary[self.dictonary['group'].isin(val_groups)].index.tolist()
        test_indices = self.dictonary[self.dictonary['group'].isin(test_groups)].index.tolist()

        print(f"Train indices: {len(train_indices)}, Validation indices: {len(val_indices)}, Test indices: {len(test_indices)}")

        return train_indices, val_indices, test_indices
    


if __name__ == "__main__":
    EleHandle = EleHandler()
    # Select a random index
    #random_idx = random.randint(0, len(EleHandle) - 1)
    #EleHandle.print_image(random_idx)
    #EleHandle.print_image(0)

    EleHandle.split_along_encounters(identification='EFE_IDI')