# src/step4_defend_against_gl_by_he/main_he.py

# Runs Federated Learning simulation with simulated Homomorphic Encryption (HE) defense
# This step demonstrates a secure aggregation setting to protect gradients from leakage

import os
import torch
import flwr as fl
import random
import numpy as np

# Import client, model, strategy, and data loading utilities
from src.step4_defend_against_gl_by_he.client_he import FlowerClientHE
from src.step1_federated_learning.model import FederatedModel
from src.step1_federated_learning.server import get_federated_strategy
from src.step1_federated_learning.data_utils import (
    DATA_ROOT, load_all_data, partition_data_for_clients
)

# --- Configuration ---
SEED = 42  # Ensures reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NUM_CLIENTS = 4              # Number of simulated clients
NUM_ROUNDS = 30              # Number of communication rounds
LOCAL_EPOCHS_PER_ROUND = 2   # Number of local training epochs per round

print("--- 🚧 Simulated Homomorphic Encryption Federated Training ---")

# --- Load dataset and partition among clients ---
full_dataset, _, _ = load_all_data(DATA_ROOT)
num_classes = len(full_dataset.classes)
client_data_loaders = partition_data_for_clients(full_dataset, NUM_CLIENTS)

# --- Flower client function for HE simulation ---
def client_fn(cid: str):
    """
    Create a client with access to a portion of the dataset.
    Each client trains independently and simulates HE protection.
    """
    client_id = int(cid)
    device = torch.device(f"cuda:{client_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    trainloader, valloader, _ = client_data_loaders[client_id]
    return FlowerClientHE(client_id, device, trainloader, valloader, num_classes)

# --- Federated Averaging Strategy setup ---
strategy = get_federated_strategy(num_total_clients=NUM_CLIENTS)

# --- Start federated learning simulation ---
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    client_resources={"num_gpus": 1.0} if torch.cuda.is_available() else {"num_cpus": 1.0},
    strategy=strategy,
)

print("--- ✅ Federated training with HE complete ---")
