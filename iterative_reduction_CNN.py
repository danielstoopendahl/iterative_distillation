import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def save_loss_to_file(loss, epoch, file_path="losses.txt"):
    """Append the loss for a given epoch to a file."""
    with open(file_path, "a") as f:
        f.write(f"Epoch {epoch}: Loss {loss:.6f}\n")

def save_test_loss_to_file(loss, accuracy, epoch, file_path="test_losses.txt"):
    """Append the test loss and accuracy for a given epoch to a file."""
    with open(file_path, "a") as f:
        f.write(f"Epoch {epoch}: Test Loss {loss:.6f}, Accuracy {accuracy:.2f}%\n")

from CNN import SimpleCNN


class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.lin1 = nn.Linear(3 * 32 * 32, 32000) 
        self.lin2 = nn.Linear(32000, 2048*2)
        self.lin3 = nn.Linear(2048*2, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        x = self.lin3(x)
        x = torch.softmax(x, dim=1)
        return x

    def forward_logits(self, x):
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        x = self.lin3(x)
        return x
    
# 0.000009752 for 2048 hidden layer
class IntermediateSNN1(nn.Module):
    def __init__(self):
        super(IntermediateSNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1) 
        self.softmax = nn.Softmax(1)
        self.added_linear1 = nn.Linear(28800,4096)
        self.added_linear2 = nn.Linear(4096,10)
        self.added_activation = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.added_linear1(x)
        x = self.added_activation(x)
        x = self.added_linear2(x)
        x = self.softmax(x)
        return x
    
    def forward_logits(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.added_linear1(x)
        x = self.added_activation(x)
        x = self.added_linear2(x)
        return x

def train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher, scheduler):
    model.train()
    teacher.eval()
    
    total_loss = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # Get teacher predictions for the current chunk
        with torch.no_grad():
            target = teacher.forward_logits(data)

        # Forward pass and compute loss
        output = model.forward_logits(data)
        loss = criterion(output, target)
        loss.backward()

        

        total_loss += loss.item()
        
        optimizer.step()
        optimizer.zero_grad()
    average_loss = total_loss / len(train_loader)
    print(f'Train Epoch: {epoch} Training Loss: {average_loss:.9f}')
    scheduler.step(loss)
    # Save the loss to a file
    save_loss_to_file(average_loss, epoch)
    

def train(model, device, train_loader, optimizer, criterion, epoch, scheduler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = F.one_hot(target, num_classes=10).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}')
        scheduler.step(loss)

def test(model, device, test_loader, criterion, epoch):
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
    print(f'\nTest set accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%) Test set loss {test_loss:.6f}\n')

    # Save the test loss and accuracy to a file
    save_test_loss_to_file(test_loss, accuracy, epoch)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


    # model = IntermediateSNN1().to(device)#.to(torch.foat64)


    # print("continuing from previous save")
    # model.load_state_dict(torch.load('models/intermediate_SNN_cifar10_big.pth'))
    model = SNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=100, 
    )  

    # teacher = SimpleCNN().to(device)
    # teacher.load_state_dict(torch.load('cifar10_cnn.pth'))
    teacher = IntermediateSNN1().to(device)
    teacher.load_state_dict(torch.load('models/intermediate_SNN_cifar10_big.pth'))

    for epoch in range(1, 151): 
        # train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher, scheduler)
        train(model, device, train_loader, optimizer, criterion, epoch, scheduler)
        test(model, device, test_loader, criterion, epoch)

    # Save the trained model
    torch.save(model.state_dict(), "models/intermediate_SNN_cifar10_huge.pth")
    print("Model saved to models/intermediate_SNN_cifar10_huge.pth")
    # torch.save(model.state_dict(), "SNN_cifar10.pth")
    # print("Model saved to SNN_cifar10.pth")

if __name__ == '__main__':
    main()