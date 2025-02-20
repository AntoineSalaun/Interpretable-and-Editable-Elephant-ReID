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
                 subset=None):
        
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


        if subset == 'IDI_6':
            # Filter the dictionary to include only EFA_IDI season and elephants that appear at least 6 times
            self.dictonary = self.dictonary[self.dictonary['#season'] == 'EFA_IDI']
            valid_ele_ids = self.dictonary['ele_id'].value_counts()[lambda x: x >= 6].index
            self.dictonary = self.dictonary[self.dictonary['ele_id'].isin(valid_ele_ids)].reset_index(drop=True)

            # Remap `ele_id` for the current subset
            self.ele_id_to_label = {ele_id: i for i, ele_id in enumerate(sorted(self.dictonary['ele_id'].unique()))}
            self.label_to_ele_id = {i: ele_id for ele_id, i in self.ele_id_to_label.items()}

        if subset == 'IDI':
            self.dictonary = self.dictonary[self.dictonary['#season'] == 'EFA_IDI']
            self.dictonary = self.dictonary.reset_index(drop=True)

            # Remap `ele_id` for the current subset
            self.ele_id_to_label = {ele_id: i for i, ele_id in enumerate(sorted(self.dictonary['ele_id'].unique()))}
            self.label_to_ele_id = {i: ele_id for ele_id, i in self.ele_id_to_label.items()}

                

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

        return preprocessed_image, subject_id, ele_id_label, identified, subject_SEEK_1hot, ele_SEEK_1hot, left_ear, right_ear, subject_SEEK, ele_SEEK, idx
    
    def _load_ear(self, ear_path):
        """Helper function to load ear images, returning zeros if path is NaN."""
        if pd.notna(ear_path):
            ear_image_path = self.image_dir / ear_path
            ear_image = Image.open(ear_image_path).convert('RGB')
            return transforms.ToTensor()(ear_image)
        return torch.zeros(3, 224, 224)

    def get_original_image(self, idx=None, subject_id=None):
        if idx is not None:
            image_path = self.image_dir / self.dictonary.iloc[idx]['image']
        elif subject_id is not None:
            image_path = self.image_dir / self.dictonary[self.dictonary['subject_id'] == subject_id].iloc[0]['image']
        else:
            raise ValueError("Either idx or subject_id must be provided")
        
        #print(f"Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB').copy()
        return image

    def print_image(self, idx, print_with_transform=True):
        # Create a figure with subplots for main image and ears
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Main image
        image_path = self.image_dir / self.dictonary.iloc[idx]['image']
        print(f"Main image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None and print_with_transform:
            image = self.transform(image)
            ax1.imshow(image.permute(1, 2, 0))
        else:
            ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Main Image')
        
        # Left ear
        left_ear_path = self.dictonary.iloc[idx]['left_ear_path']
        if pd.notna(left_ear_path):
            left_ear_image_path = self.image_dir / left_ear_path
            print(f"Left ear: {left_ear_image_path}")
            left_ear = Image.open(left_ear_image_path).convert('RGB')
            ax2.imshow(left_ear)
        ax2.axis('off')
        ax2.set_title('Left Ear')
        
        # Right ear
        right_ear_path = self.dictonary.iloc[idx]['right_ear_path']
        if pd.notna(right_ear_path):
            right_ear_image_path = self.image_dir / right_ear_path
            print(f"Right ear: {right_ear_image_path}")
            right_ear = Image.open(right_ear_image_path).convert('RGB')
            ax3.imshow(right_ear)
        ax3.axis('off')
        ax3.set_title('Right Ear')
        
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

        # Work on a copy of the dictionary to avoid modifying the original
        dictonary_copy = self.dictonary.copy()

        # Ensure 'picture_time' is sorted
        dictonary_copy.sort_values(by='picture_time', inplace=True)

        # Create a new column 'group' to group images taken within an hour
        dictonary_copy['group'] = (pd.to_datetime(dictonary_copy['picture_time']).diff() > pd.Timedelta(hours=hour_delta)).cumsum()

        # Split the groups into train, validation, and test sets
        unique_groups = dictonary_copy['group'].unique()
        print(f"Unique groups: {len(unique_groups)}")

        train_groups, temp_groups = train_test_split(unique_groups, test_size=1-split_sizes[0], random_state=42, shuffle=True)
        val_groups, test_groups = train_test_split(temp_groups, test_size=split_sizes[2]/(1-split_sizes[0]), random_state=42, shuffle=True)

        # Get the indices for each group
        train_indices = dictonary_copy[dictonary_copy['group'].isin(train_groups)].index.tolist()
        val_indices = dictonary_copy[dictonary_copy['group'].isin(val_groups)].index.tolist()
        test_indices = dictonary_copy[dictonary_copy['group'].isin(test_groups)].index.tolist()

        print(f"Train indices: {len(train_indices)}, Validation indices: {len(val_indices)}, Test indices: {len(test_indices)}")

        return train_indices, val_indices, test_indices

    def split_perpendicular_to_elephants_and_encounters(self, split_sizes=[0.5, 0.5], hour_delta=0.2):
        # Ensure split sizes sum to 1
        if sum(split_sizes) != 1: raise ValueError("Split sizes should sum to 1")

        # Work on a copy of the dictionary to avoid modifying the original
        dictonary_copy = self.dictonary.copy()

        # Ensure 'picture_time' is sorted
        dictonary_copy.sort_values(by='picture_time', inplace=True)

        # Create a new column 'group' to group images taken within an hour
        dictonary_copy['group'] = (pd.to_datetime(dictonary_copy['picture_time']).diff() > pd.Timedelta(hours=hour_delta)).cumsum()

        # Ensure each set contains all elephants
        train_indices, test_indices = [], []

        for ele_id in dictonary_copy['ele_id'].unique(): # Loop over unique elephant IDs
            
            # Extracting the indices of the current elephant
            ele_id_indices = dictonary_copy[dictonary_copy['ele_id'] == ele_id].index.tolist()

            # Create a small specific to this elephant with the required columns
            ele_id_df = dictonary_copy.loc[ele_id_indices, ['ele_id', 'group']].reset_index()

            # Count the occurrences of each group
            group_counts = ele_id_df['group'].value_counts()

            # Sort the DataFrame by group count in descending order
            ele_id_df_sorted = ele_id_df.set_index('group').loc[group_counts.index].reset_index()
            
            train_count = int(len(ele_id_indices) * split_sizes[0])
            test_count = len(ele_id_indices) - train_count 

            train_contribution_df = pd.DataFrame(columns=['index', 'ele_id', 'group'])
            test_contribution_df = pd.DataFrame(columns=['index', 'ele_id', 'group'])

            # for each image of this elephant
            for idx in ele_id_df_sorted.index:

                # extract its group
                group = ele_id_df_sorted.loc[idx, 'group']
                
                
                # if the number of images in the training set is less than the required number of images, we start by filling the train (otherwis we start with the test) 
                if len(train_contribution_df) / train_count <= len(test_contribution_df) / test_count:
                    # if this group shows up less in train than test AND we did not fill the train set yet, we add it to the train set, otherwise we add it to the test set
                    if len(train_contribution_df[train_contribution_df['group']==group]) <= len(test_contribution_df[test_contribution_df['group']==group]) and len(train_contribution_df) < train_count: 
                        train_contribution_df = pd.concat([train_contribution_df, ele_id_df_sorted.loc[[idx]]])
                    else:
                        test_contribution_df = pd.concat([test_contribution_df, ele_id_df_sorted.loc[[idx]]])
                else:
                    # if this group shows up less in test than train AND we did not fill the test set yet, we add it to the test set, otherwise we add it to the train set
                    if len(test_contribution_df[test_contribution_df['group']==group]) < len(train_contribution_df[train_contribution_df['group']==group]) and len(test_contribution_df) < test_count:
                        test_contribution_df = pd.concat([test_contribution_df, ele_id_df_sorted.loc[[idx]]])
                    else:
                        train_contribution_df = pd.concat([train_contribution_df, ele_id_df_sorted.loc[[idx]]])

            train_indices += train_contribution_df['index'].tolist()
            test_indices += test_contribution_df['index'].tolist()


        return train_indices, test_indices

if __name__ == "__main__":
    EleHandle = EleHandler()
    # Select a random index
    #random_idx = random.randint(0, len(EleHandle) - 1)
    #EleHandle.print_image(random_idx)
    #EleHandle.print_image(0)

    EleHandle.split_along_encounters(identification='EFE_IDI')