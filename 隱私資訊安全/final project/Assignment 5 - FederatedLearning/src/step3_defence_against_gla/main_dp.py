# src/step3_defence_against_gla/main_dp.py

# This script runs a Federated Learning (FL) simulation using Differential Privacy (DP)
# by injecting Gaussian noise into client gradients during training.

import os
import torch
import flwr as fl

# Import the custom DP-enabled client
from src.step3_defence_against_gla.client_dp import FlowerClientDP

# Use the same FedAvg server strategy as in Step 1
from src.step1_federated_learning.server import get_federated_strategy

# Import the shared model and data handling utilities
from src.step1_federated_learning.model import FederatedModel
from src.step1_federated_learning.data_utils import (
    DATA_ROOT, load_all_data, partition_data_for_clients, get_global_eval_dataloader
)

# --- Configuration ---
NUM_CLIENTS = 4                   # Number of federated clients
NUM_ROUNDS = 30                  # Total number of FL rounds
LOCAL_EPOCHS_PER_ROUND = 2      # Local epochs per round (not passed explicitly here)
NOISE_STD = 1.0                 # Std deviation for Gaussian noise (used for DP)

# --- Dataset Loading and Partitioning ---
# Loads the full dataset and partitions it among clients
full_dataset, _, _ = load_all_data(DATA_ROOT)
total_classes = len(full_dataset.classes)
client_data_loaders_list = partition_data_for_clients(full_dataset, NUM_CLIENTS)

# --- Flower Client Function ---
def client_fn(cid: str) -> FlowerClientDP:
    """
    Initializes a FlowerClientDP for a given client ID (cid).
    Applies Gaussian noise with standard deviation = NOISE_STD during gradient computation.
    """
    client_id = int(cid)
    device = torch.device(
        f"cuda:{client_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    )
    trainloader, valloader, _ = client_data_loaders_list[client_id]
    return FlowerClientDP(client_id, device, trainloader, valloader, total_classes, noise_std=NOISE_STD)

# --- Federated Learning Simulation Entry Point ---
if __name__ == "__main__":
    print(f"--- Starting DP Federated Learning with noise std = {NOISE_STD} ---")

    # Initialize model and strategy
    initial_model = FederatedModel(num_classes=total_classes)
    strategy = get_federated_strategy(num_total_clients=NUM_CLIENTS)

    # Start the FL simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_gpus": 1.0} if torch.cuda.is_available() else {"num_cpus": 1.0},
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("--- DP Federated Learning complete ---")
