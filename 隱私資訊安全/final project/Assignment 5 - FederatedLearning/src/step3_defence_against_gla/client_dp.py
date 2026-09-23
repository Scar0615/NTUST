# src/step3_defence_against_gla/client_dp.py

# A DP (Differential Privacy) defense by adding noise to client gradients during training

import os
import torch
import torch.nn as nn
from collections import OrderedDict
import flwr as fl

from src.step1_federated_learning.model import FederatedModel

class FlowerClientDP(fl.client.NumPyClient):
    """
    A federated learning client that defends against gradient leakage by
    applying Gaussian noise to the gradients (Differential Privacy).
    """

    def __init__(self, cid, device, trainloader, valloader, num_classes, noise_std=1.0):
        self.cid = cid
        self.device = device
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_examples = {
            "train": len(trainloader.dataset),
            "val": len(valloader.dataset)
        }
        self.model = FederatedModel(num_classes=num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.CrossEntropyLoss()
        self.noise_std = noise_std  # Standard deviation for Gaussian noise

    def get_parameters(self, config):
        """Return model weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model weights from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model locally and apply DP noise to gradients."""
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

                # ✅ Add Gaussian noise to gradients for DP
                for param in self.model.parameters():
                    if param.grad is not None:
                        noise = torch.normal(
                            0, self.noise_std, size=param.grad.shape, device=param.grad.device
                        )
                        param.grad += noise

                # ✅ Save noisy gradients and inputs (only once, client 0)
                if self.cid == 0 and epoch == 0 and batch_idx == 0:
                    dp_dir = os.path.abspath(os.path.join(
                        os.path.dirname(__file__), "extracted_gradients"
                    ))
                    os.makedirs(dp_dir, exist_ok=True)
                    grads = [param.grad.cpu().clone() for param in self.model.parameters() if param.grad is not None]
                    torch.save(grads, os.path.join(dp_dir, "dp_gradients_client0.pt"))
                    torch.save(data.cpu(), os.path.join(dp_dir, "dp_original_input.pt"))
                    torch.save(target.cpu(), os.path.join(dp_dir, "dp_original_label.pt"))
                    print(f"[Client {self.cid}] DP gradient snapshot saved.")

                self.optimizer.step()

        return self.get_parameters(config={}), self.num_examples["train"], {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the client's validation data."""
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
