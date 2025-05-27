# pattern_optimizer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Neural Network Architecture
class PatternOptimizer(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(PatternOptimizer, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

# Training Loop with L-BFGS Optimization
def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss
            optimizer.step(closure)
        
        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        test_loss /= len(test_loader.dataset)
        print(f'Epoch {epoch+1}:')
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct/len(test_loader.dataset):.0f}%)')

# Main Execution
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    input_dim = 784  # 28x28 pixels
    output_dim = 10  # FashionMNIST classes
    
    # Load Dataset (FashionMNIST)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize Model
    model = PatternOptimizer(input_dim=input_dim, output_dim=output_dim)
    
    # Train and Evaluate
    train_model(model, train_loader, test_loader, epochs=10)
