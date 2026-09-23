# src/step4_defend_against_gl_by_he/client_he.py

# Simulates Federated Learning client with Homomorphic Encryption (HE)
# This does not implement real HE, but simulates secure communication behavior
# and stores gradients for analysis.

import os
import torch
import torch.nn as nn
from collections import OrderedDict
import flwr as fl

from src.step1_federated_learning.model import FederatedModel


class FlowerClientHE(fl.client.NumPyClient):
    def __init__(self, cid, device, trainloader, valloader, num_classes):
        # Initialize client ID, device, model, data loaders, optimizer, and loss function
        self.cid = cid
        self.device = device
        self.model = FederatedModel(num_classes=num_classes).to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.CrossEntropyLoss()
        self.num_examples = {
            "train": len(self.trainloader.dataset),
            "val": len(self.valloader.dataset)
        }

    def get_parameters(self, config):
        # Return model parameters as NumPy arrays (required by Flower)
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Load parameters into the model from NumPy arrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Perform local training using current parameters
        self.set_parameters(parameters)
        self.model.train()
        epochs = config.get("epochs", 1)

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # ✅ Save gradients and data snapshot for client 0 (only once)
                if self.cid == 0 and epoch == 0 and batch_idx == 0:
                    base_dir = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "extracted_gradients")
                    )
                    os.makedirs(base_dir, exist_ok=True)

                    grads = [param.grad.cpu().clone() for param in self.model.parameters() if param.grad is not None]
                    torch.save(grads, os.path.join(base_dir, "he_gradients_client0.pt"))
                    torch.save(data.cpu(), os.path.join(base_dir, "he_original_input.pt"))
                    torch.save(target.cpu(), os.path.join(base_dir, "he_original_label.pt"))
                    print(f"[Client {self.cid}] ✅ HE gradient and input snapshot saved.")

                self.optimizer.step()

        return self.get_parameters(config={}), self.num_examples["train"], {}

    def evaluate(self, parameters, config):
        # Evaluate model on validation set
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
