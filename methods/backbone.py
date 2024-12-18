import torch
from pathlib import Path
from datetime import datetime
import timm

import torch.nn as nn



class Backbone(nn.Module):
    def __init__(self, with_ears = True, pretraining = None, experiment_code = None):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"

        super().__init__()

        # Load the backbone model
        self.layer = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)
        if pretraining == "savannah_elephants":
            state_dict = torch.load(Path(__file__).parent.parent / "weights/savanna_elephants_md_v2_epoch_60.pt", map_location=self.device, weights_only=False)
        elif pretraining == "forest_elephants":
            state_dict = torch.load(Path(__file__).parent.parent / "weights/forest_elephants-reid_weights.pt", map_location=self.device, weights_only=False)
        elif pretraining == None:
            print("No pretraining, using MegaDescriptor")
            state_dict = torch.load(Path(__file__).parent.parent / "weights/forest_elephants-reid_weights.pt", map_location=self.device, weights_only=False)

        if pretraining in ["savannah_elephants", "forest_elephants"]:   
            self.layer.load_state_dict(state_dict["model"] if "optimizer" in state_dict else state_dict)

        self.layer.to(self.device)
        self.freeze()

        self.with_ears = with_ears

                # Create experiment directory
        exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = Path(__file__).parent.parent / 'experiments' / f'exp_{exp_code}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(self.experiment_dir / 'note.txt', 'w') as f: f.write('---Backbone---/n')


    def freeze(self):
        self.layer.eval()
        # Freeze the backbone parameters
        for param in self.layer.parameters():
            param.requires_grad = False

    def forward(self, images, left_ears = None, right_ears = None):
        
        with torch.no_grad():
            embeddings = self.layer(images)
                
            if self.with_ears:
                left_embeddings = self.layer(left_ears.to(self.device))
                right_embeddings = self.layer(right_ears.to(self.device))
                embeddings = torch.cat((embeddings, left_embeddings, right_embeddings), dim=1)

        return embeddings
    
