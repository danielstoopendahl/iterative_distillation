import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from CNN import SimpleCNN


class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.lin1 = nn.Linear(3 * 32 * 32, 16000) 
        self.lin2 = nn.Linear(16000, 128)
        self.lin3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = torch.softmax(x, dim=1)
        return x

    def forward_logits(self, x):
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x
    
class IntermediateSNN1(nn.Module):
    def __init__(self):
        super(IntermediateSNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1) 
        self.softmax = nn.Softmax(1)
        self.added_linear = nn.Linear(28800,10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.added_linear(x)
        x = self.softmax(x)
        return x
    
    def forward_logits(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.added_linear(x)
        return x

def train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher):
    model.train()
    teacher.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            target = teacher.forward_logits(data)

        optimizer.zero_grad()
        output = model.forward_logits(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}')

    
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target_one_hot = F.one_hot(target, num_classes=10).float()
            output = model(data)
            test_loss += criterion(output, target_one_hot).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= test_loader.batch_size
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


    model = IntermediateSNN1().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    teacher = SimpleCNN().to(device)
    teacher.load_state_dict(torch.load('cifar10_cnn.pth'))

    for epoch in range(1, 7):  # 3 epochs for demo
        train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher)
        test(model, device, test_loader, criterion)

    # Save the trained model
    torch.save(model.state_dict(), "intermediate_SNN_cifar10.pth")
    print("Model saved to intermediate_SNN_cifar10.pth")

def main2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


    model = SNN().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    teacher = IntermediateSNN1().to(device)
    teacher.load_state_dict(torch.load('intermediate_SNN_cifar10.pth'))

    for epoch in range(1, 7):  # 3 epochs for demo
        train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher)
        test(model, device, test_loader, criterion)

    # Save the trained model
    torch.save(model.state_dict(), "SNN_cifar10.pth")
    print("Model saved to SNN_cifar10.pth")

if __name__ == '__main__':
    main2()