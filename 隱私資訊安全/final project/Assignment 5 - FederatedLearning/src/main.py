# src/main.py

# --- Main script to simulate Federated Learning using Flower (Step 1) ---

import os
import time
import random
import numpy as np
import torch
import flwr as fl

# ✅ Reproducibility setup: Fix random seeds for consistent results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import custom modules
from step1_federated_learning.client import FlowerClient
from step1_federated_learning.server import get_federated_strategy
from step1_federated_learning.model import FederatedModel
from step1_federated_learning.data_utils import (
    DATA_ROOT, IMAGE_SIZE, load_all_data, partition_data_for_clients,
    get_global_eval_dataloader
)

# --- Federated Learning Configuration ---
NUM_CLIENTS = 4  # Number of simulated clients
NUM_ROUNDS = 30  # Number of federated training rounds
LOCAL_EPOCHS_PER_ROUND = 2  # Local epochs per client per round
SERVER_ADDRESS = "0.0.0.0:8080"  # (Optional) for real network setup

# Load dataset to determine global class count
FULL_DATASET_GLOBAL, _, _ = load_all_data(DATA_ROOT)
TOTAL_GLOBAL_CLASSES = len(FULL_DATASET_GLOBAL.classes)

# Partition dataset across clients once
client_data_loaders_list = partition_data_for_clients(FULL_DATASET_GLOBAL, NUM_CLIENTS)


def client_fn(cid: str) -> FlowerClient:
    """
    Factory function to initialize a Flower client.

    Args:
        cid (str): Client ID string passed by the Flower framework.

    Returns:
        FlowerClient: A fully initialized client with its own data and model.
    """
    client_id = int(cid)
    device = torch.device(f"cuda:{client_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    print(f"Client {client_id}: Initializing on {device}...")

    # Get data loaders for this client
    trainloader, valloader, _ = client_data_loaders_list[client_id]

    return FlowerClient(client_id, device, trainloader, valloader, TOTAL_GLOBAL_CLASSES)


if __name__ == "__main__":
    print(f"--- Starting Federated Learning Simulation with {NUM_CLIENTS} clients ---")
    print(f"Dataset root: {os.path.abspath(DATA_ROOT)}")

    # Create a global model and retrieve its initial parameters
    initial_model = FederatedModel(num_classes=TOTAL_GLOBAL_CLASSES)
    initial_parameters = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]

    # Define the federated learning strategy (FedAvg + metric tracking)
    strategy = get_federated_strategy(num_total_clients=NUM_CLIENTS)

    # Create a global evaluation dataset from unused data
    global_eval_dataloader, _ = get_global_eval_dataloader(
        FULL_DATASET_GLOBAL, client_data_loaders_list, TOTAL_GLOBAL_CLASSES
    )

    # 🌐 Start the federated training simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_gpus": 1.0} if torch.cuda.is_available() else {"num_cpus": 1.0},
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\n--- Federated learning simulation complete! ---")

    # 💾 Save the trained global model to disk for evaluation or attacks
    print("Attempting to save the final global model...")
    if hasattr(strategy, 'final_parameters') and strategy.final_parameters is not None:
        try:
            final_numpy_parameters = fl.common.parameters_to_ndarrays(strategy.final_parameters)
            final_global_model = FederatedModel(num_classes=TOTAL_GLOBAL_CLASSES)
            final_global_model.set_parameters(final_numpy_parameters)
            save_path = "step1_federated_learning/federated_model_final.pth"
            torch.save(final_global_model.state_dict(), save_path)
            print(f"Final global model saved successfully to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print("Could not retrieve final global model parameters from the strategy.")
