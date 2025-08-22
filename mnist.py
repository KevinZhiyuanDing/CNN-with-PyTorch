import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Default hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
CONV_LAYERS = [16, 32]
FC_LAYERS = [128, 64, 10]
SEED = 42

class CNN(nn.Module):
    """Convolutional Neural Network for MNIST classification."""
    def __init__(self, input_shape, conv_channels, layer_sizes, activation=nn.ReLU):
        super().__init__()
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_shape[0]
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                activation(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv_layers(dummy)
            flat_size = conv_out.view(1, -1).shape[1]

        # Build fully connected layers
        fc_layers = []
        in_size = flat_size
        for out_size in layer_sizes[:-1]:
            fc_layers.extend([
                nn.Linear(in_size, out_size),
                activation()
            ])
            in_size = out_size
        
        # Output layer (logits for classification)
        fc_layers.append(nn.Linear(in_size, layer_sizes[-1]))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)

def train(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)
        # Loss computation
        loss = loss_fn(output, targets)
        # Backward pass
        loss.backward()
        # Update parameters (weights and biases)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (output.argmax(1) == targets).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct = 0.0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)

            total_loss += loss.item() * inputs.size(0)
            correct += (output.argmax(1) == targets).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def main(args):
    # Set seed for reproducability
    torch.manual_seed(SEED)

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    input_shape = (1, 28, 28)
    model = CNN(input_shape, conv_channels=CONV_LAYERS, layer_sizes=FC_LAYERS).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.NLLLoss()

    train_losses = []
    test_losses = []

    # Train and Evaluate
    train_loss, train_acc = evaluate(model, train_loader, loss_fn, device)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch 0/{NUM_EPOCHS} - "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(NUM_EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(NUM_EPOCHS + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    main(args)
