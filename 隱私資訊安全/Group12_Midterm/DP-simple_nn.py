
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms 
from torch.utils.tensorboard import SummaryWriter 
from opacus import PrivacyEngine 
import os 

# --------------------- 
# Hyperparameters 
# --------------------- 
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.1
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
LOG_DIR = "runs/dp_mnist"
MODEL_PATH = "SimpleNN_model.pth"

# --------------------- 
# Setup 
# --------------------- 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

# --------------------- 
# Dataset 
# --------------------- 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# --------------------- 
# Model - Simple Fully Connected Network 
# --------------------- 
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --------------------- 
# Evaluation Function 
# --------------------- 
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# --------------------- 
# Training Function 
# --------------------- 
def train():
    model = SimpleNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader_private = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.1,
        max_grad_norm=MAX_GRAD_NORM,
    )

    print("Training with DP-SGD...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        batch_count = 0
        for x, y in train_loader_private:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        acc = evaluate(model)
        epsilon = privacy_engine.get_epsilon(DELTA)
        print(f"Epoch {epoch}: Avg Loss={avg_loss:.4f}, Accuracy={acc:.4f}, Epsilon={epsilon:.2f}")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/test", acc, epoch)
        writer.add_scalar("Privacy/epsilon", epsilon, epoch)

        scheduler.step()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    writer.close()
    print("Training complete. Use TensorBoard to visualize logs:")
    print(f"tensorboard --logdir={LOG_DIR}")

# --------------------- 
# Testing Function 
# --------------------- 
def test():
    model = SimpleNN().to(device)
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('_module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    acc = evaluate(model)
    print(f"Loaded model test accuracy: {acc:.4f}")

if __name__ == "__main__":
    # train()     # Uncomment to train the model
    test()       # Uncomment to test the saved model
