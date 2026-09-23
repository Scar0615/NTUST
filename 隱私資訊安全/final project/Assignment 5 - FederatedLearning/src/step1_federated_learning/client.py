# src/step1_federated_learning/client.py

# Import necessary libraries
import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import flwr as fl  # Flower: federated learning framework

# Set a global seed for reproducibility across all libraries
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from .model import FederatedModel  # Import the CNN model definition

# Define a custom Flower client for federated learning
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: int, device: torch.device, trainloader: torch.utils.data.DataLoader,
                 valloader: torch.utils.data.DataLoader, num_classes: int):
        """
        Initialize the client with local data and model configuration.
        """
        super().__init__()
        self.cid = cid  # client ID
        self.device = device
        self.model = FederatedModel(num_classes=num_classes).to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_examples = {
            "train": len(self.trainloader.dataset),
            "val": len(self.valloader.dataset)
        }
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        """
        Return model weights as a list of NumPy arrays.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """
        Load model weights from a list of NumPy arrays.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train the local model using client data and return the updated weights.
        """
        self.set_parameters(parameters)
        epochs = config.get("epochs", 1)

        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # Save the first batch's gradients and input for client 0 (for gradient leakage attack)
                if self.cid == 0 and epoch == 0 and batch_idx == 0:
                    base_dir = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "step2_gradient_leakage_attack", "extracted_gradients")
                    )
                    os.makedirs(base_dir, exist_ok=True)
                    grads = [param.grad.clone().cpu() for param in self.model.parameters() if param.grad is not None]
                    torch.save(grads, os.path.join(base_dir, "gradients_client0.pt"))
                    torch.save(data.cpu(), os.path.join(base_dir, "original_input.pt"))
                    torch.save(target.cpu(), os.path.join(base_dir, "original_label.pt"))
                    print(f"[Client {self.cid}] ✅ Clean gradient and input snapshot saved.")

                self.optimizer.step()

        return self.get_parameters(config={}), self.num_examples["train"], {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model on the local validation set.
        """
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.valloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= self.num_examples["val"]
        accuracy = correct / self.num_examples["val"]
        return loss, self.num_examples["val"], {"accuracy": accuracy}
