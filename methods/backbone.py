import torch
import torch.nn as nn
import timm
from pathlib import Path


class Backbone(nn.Module):
    def __init__(self, frozen=True, pretraining="savannah_elephants", with_ears=True):
        """
        Initializes the Backbone model.
        
        Args:
        - frozen (bool): Whether to freeze the backbone parameters.
        - pretraining (str): Pretrained model type ('savannah_elephants' or 'forest_elephants').
        - with_ears (bool): Whether to include ear embeddings in the forward pass.
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.with_ears = with_ears
        self.model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)

        # Load pretraining weights
        if pretraining == "savannah_elephants":
            state_dict = torch.load(
                Path(__file__).parent.parent / "weights/savanna_elephants_md_v2_epoch_60.pt",
                map_location=self.device
            )
        elif pretraining == "forest_elephants":
            state_dict = torch.load(
                Path(__file__).parent.parent / "weights/forest_elephants-reid_weights.pt",
                map_location=self.device
            )
        self.model.load_state_dict(state_dict["model"] if "optimizer" in state_dict else state_dict)
        self.model.to(self.device)

        # Optionally freeze backbone parameters
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images, left_ears=None, right_ears=None):
        """
        Forward pass for the backbone.
        
        Args:
        - images (torch.Tensor): Input images.
        - left_ears (torch.Tensor, optional): Left ear images.
        - right_ears (torch.Tensor, optional): Right ear images.

        Returns:
        - embeddings (torch.Tensor): Concatenated embeddings if `with_ears` is True, else main image embeddings.
        """
        embeddings = self.model(images)

        if self.with_ears:
            left_embeddings = self.model(left_ears)
            right_embeddings = self.model(right_ears)
            embeddings = torch.cat((embeddings, left_embeddings, right_embeddings), dim=1)

        return embeddings