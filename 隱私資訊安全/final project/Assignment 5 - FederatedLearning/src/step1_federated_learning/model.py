# src/step1_federated_learning/model.py

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models  # Load pretrained CNN architectures
from collections import OrderedDict
import torch

# --- Federated Learning Model (based on ResNet18) ---
class FederatedModel(nn.Module):
    """
    A model designed for Federated Learning, based on ResNet18 from torchvision.
    The pretrained weights help accelerate training, and the final layer is adapted
    to match the number of classes in the dataset.
    """

    def __init__(self, num_classes: int):
        super(FederatedModel, self).__init__()

        # Load ResNet18 with pretrained weights (ImageNet)
        self.base_model = models.resnet18(pretrained=True)

        # --- Optional: Freeze base layers ---
        # Uncomment the block below to freeze all convolutional layers
        # This reduces training time and risk of overfitting with small datasets
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        # Replace the final fully connected layer with a new one for our task
        num_ftrs = self.base_model.fc.in_features  # Usually 512
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Defines how input data flows through the model.
        """
        return self.base_model(x)

    def set_parameters(self, parameters):
        """
        Utility method to load weights from a list of NumPy arrays.
        This is used by Flower during federated learning to update the model
        after server-side aggregation.
        """
        param_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.load_state_dict(state_dict, strict=True)
