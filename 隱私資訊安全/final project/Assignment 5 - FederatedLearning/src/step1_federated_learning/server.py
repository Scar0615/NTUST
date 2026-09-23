# src/step1_federated_learning/server.py

import flwr as fl
import numpy as np
from typing import Dict, List, Optional, Tuple
from flwr.common import Scalar
from flwr.server.strategy import FedAvg

# ✅ Custom federated strategy that tracks final global model parameters
class FedAvgWithFinalParams(FedAvg):
    """
    A custom extension of Flower's FedAvg strategy that saves the final
    aggregated model parameters after the last federated training round.
    Useful when we want to use the trained model for evaluation or attacks later.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None  # Store final aggregated weights here

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[fl.common.Parameters, Dict[str, Scalar]]]:
        """
        Override the default aggregation behavior to capture and store
        the final model parameters at the end of training.
        """
        # Use the original FedAvg logic to aggregate client updates
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)

        # Save the aggregated parameters for access after training
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters

        return aggregated_parameters, metrics


def get_federated_strategy(num_total_clients: int) -> fl.server.strategy.Strategy:
    """
    Returns an instance of the custom FedAvg strategy for federated training.

    Args:
        num_total_clients (int): Total number of clients in simulation.

    Returns:
        fl.server.strategy.Strategy: Configured federated averaging strategy.
    """
    return FedAvgWithFinalParams(
        fraction_fit=1.0,  # All clients participate in training each round
        fraction_evaluate=1.0,  # All clients evaluated each round
        min_fit_clients=num_total_clients,  # Minimum training clients per round
        min_evaluate_clients=num_total_clients,  # Minimum evaluation clients per round
        min_available_clients=num_total_clients,  # All must be available to start
        evaluate_metrics_aggregation_fn=lambda results: {
            "accuracy": np.mean([r[1]["accuracy"] for r in results]) if results else 0.0
        },  # Compute average accuracy
    )
