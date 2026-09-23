# src/step1_federated_learning/data_utils.py

import os
from collections import defaultdict
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# --- Dataset Configuration ---
# Path to image dataset (organized in subfolders by class)
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "original"))

IMAGE_SIZE = 224  # Resize all images to 224x224
BATCH_SIZE = 32   # Batch size for all data loaders

def load_all_data(data_root):
    """
    Load the full dataset with preprocessing (resize, normalization).
    Returns both dataset and label mappings.
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_root, transform=transform)

    num_classes = len(full_dataset.classes)
    print(f"Total number of famous people (classes) found: {num_classes}")
    print(f"Detected classes: {full_dataset.classes}")

    return full_dataset, full_dataset.class_to_idx, {v: k for k, v in full_dataset.class_to_idx.items()}


def partition_data_for_clients(full_dataset, num_clients, split_ratio=0.8):
    """
    Split the dataset into disjoint subsets, one per client, based on class labels.
    Each client gets unique classes to simulate non-IID distribution.
    """

    # Group sample indices by class label
    class_indices = defaultdict(list)
    for i, (_, label) in enumerate(full_dataset):
        class_indices[label].append(i)

    all_class_labels = sorted(list(class_indices.keys()))
    num_total_classes = len(all_class_labels)

    # Ensure enough classes for all clients
    if num_total_classes < num_clients:
        raise ValueError(
            f"Not enough classes ({num_total_classes}) to distribute among {num_clients} clients."
        )

    # Assign equal number of classes per client
    classes_per_client = num_total_classes // num_clients
    client_class_assignments = []
    for i in range(num_clients):
        start_idx = i * classes_per_client
        end_idx = start_idx + classes_per_client
        if i == num_clients - 1:
            assigned_classes = all_class_labels[start_idx:]
        else:
            assigned_classes = all_class_labels[start_idx:end_idx]
        client_class_assignments.append(assigned_classes)

    client_data_loaders = []

    for client_id, assigned_classes in enumerate(client_class_assignments):
        print(f"Client {client_id} assigned classes: {assigned_classes}")

        # Gather all image indices belonging to the assigned classes
        client_indices = []
        for class_label in assigned_classes:
            client_indices.extend(class_indices[class_label])

        client_subset = Subset(full_dataset, client_indices)

        # Split into train/test for this client
        train_size = int(split_ratio * len(client_subset))
        test_size = len(client_subset) - train_size

        # Fallback in case client has too little data
        if train_size == 0 or test_size == 0:
            print(f"Warning: Client {client_id} has too few samples. Adjusting split.")
            if len(client_subset) >= 2:
                train_size = len(client_subset) // 2
                test_size = len(client_subset) - train_size
            else:
                train_size = len(client_subset)
                test_size = 0

        client_train_subset, client_test_subset = torch.utils.data.random_split(
            client_subset, [train_size, test_size]
        )

        # Create loaders for this client
        trainloader = DataLoader(client_train_subset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(client_test_subset, batch_size=BATCH_SIZE, shuffle=False)

        # Each client will use a model with output size equal to the total number of global classes
        num_local_classes_for_client = num_total_classes

        client_data_loaders.append((trainloader, testloader, num_local_classes_for_client))

    return client_data_loaders


def get_global_eval_dataloader(full_dataset, client_partitioned_loaders, num_classes):
    """
    Create a global evaluation set using only samples not used by any client during training.
    """
    all_client_train_indices = set()
    for train_dl, _, _ in client_partitioned_loaders:
        all_client_train_indices.update(train_dl.dataset.indices)

    all_indices = set(range(len(full_dataset)))
    global_eval_indices = list(all_indices - all_client_train_indices)

    eval_sample_size = min(len(global_eval_indices), int(0.1 * len(full_dataset)))

    # If no samples are left, fall back to a small random subset
    if eval_sample_size == 0:
        print("Warning: No remaining data for global evaluation. Using fallback subset.")
        eval_sample_size = min(1000, len(full_dataset) // 10)
        if eval_sample_size == 0:
            raise ValueError("Dataset too small to create evaluation set.")
        global_eval_indices = torch.randperm(len(full_dataset)).tolist()[:eval_sample_size]

    global_eval_dataset = Subset(full_dataset, global_eval_indices)
    global_eval_loader = DataLoader(global_eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Global evaluation dataset size: {len(global_eval_dataset)} samples.")
    return global_eval_loader, num_classes


# --- Global Variables for Use in Other Modules ---
# These lines execute once when this file is imported

FULL_DATASET, _, _ = load_all_data(DATA_ROOT)
GLOBAL_NUM_CLASSES = len(FULL_DATASET.classes)

# --- Local test/debug run ---
if __name__ == "__main__":
    print(f"Attempting to load data from: {os.path.abspath(DATA_ROOT)}")
    print(f"Loaded FULL_DATASET with {len(FULL_DATASET)} samples.")
    print(f"GLOBAL_NUM_CLASSES: {GLOBAL_NUM_CLASSES}")

    num_clients_test = 5
    client_loaders_test = partition_data_for_clients(FULL_DATASET, num_clients_test)

    for i, (train_dl, test_dl, num_lcl_cls) in enumerate(client_loaders_test):
        print(f"\nClient {i}:")
        print(f"  Train samples: {len(train_dl.dataset)}")
        print(f"  Test samples: {len(test_dl.dataset)}")
        print(f"  Number of local classes (model output size): {num_lcl_cls}")

        for images, labels in train_dl:
            print(f"  Train batch image shape: {images.shape}, labels shape: {labels.shape}")
            print(f"  Unique labels in train batch: {torch.unique(labels)}")
            break

    global_eval_dataloader, total_classes = get_global_eval_dataloader(
        FULL_DATASET, client_loaders_test, GLOBAL_NUM_CLASSES
    )
    print(f"\nGlobal Eval Dataloader Samples: {len(global_eval_dataloader.dataset)}")
    print(f"Global Eval Dataloader Total Classes: {total_classes}")
