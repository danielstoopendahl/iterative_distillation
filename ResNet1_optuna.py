import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import os
import optuna
from ResNet18 import ResNet18
from ResNet2 import ResNet2
from ResNet6 import ResNet6

TRIALS = 20
TRIAL_EPOCHS = 80
BEST_PARAMS_PATH = "optuna_params/resnet1.json"

# 43% ResNet-2 as teacher
# 59% ResNet-6 as teacher
# 62% ResNet-18 as teacher

class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()

        input_dim = 3 * 32 * 32
        hidden_dim = 16384

        self.net = nn.Sequential(
            nn.Flatten(),                   
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes), 
        )
        self.softmax = nn.Softmax(dim=1)

    def forward_logits(self, x):
        return self.net(x)

    def forward(self, x):
        logits = self.forward_logits(x)
        return self.softmax(logits)


def ResNet1(num_classes=10):
    return ResNetCIFAR(num_classes=num_classes)


def train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher):
    model.train()
    running_loss = 0.0
    teacher.eval()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model.forward_logits(data)
         
        with torch.no_grad():
            target = teacher.forward_logits(data)

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

    return running_loss / len(train_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model.forward_logits(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return test_loss, accuracy


def get_dataloaders(batch_size):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    return train_loader, test_loader


def load_teacher(teacher_name, device):
    teacher_map = {
        "resnet2": (ResNet2, "models/resnet2_cifar10.pth"),
        "resnet6": (ResNet6, "models/resnet6_cifar10.pth"),
        "resnet18": (ResNet18, "models/resnet18_cifar10.pth"),
    }

    teacher_class, teacher_path = teacher_map[teacher_name]
    teacher = teacher_class().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    return teacher


def run_training(model, teacher, device, train_loader, test_loader, optimizer, scheduler, criterion, epochs, trial=None):
    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher)
        print(f"Epoch {epoch}: Train loss {train_loss:.6f}")

        val_loss, val_accuracy = test(model, device, test_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Learning rate {current_lr:.2e}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

        if trial is not None:
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_accuracy


def create_objective(device):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 3e-3, log=True)
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        patience = trial.suggest_int("patience", 3, 15)

        train_loader, test_loader = get_dataloaders(batch_size=batch_size)
        model = ResNet1(num_classes=10).to(device)
        teacher = load_teacher("resnet6", device)

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=patience,
            min_lr=1e-8,
        )
        criterion = nn.MSELoss()

        best_accuracy = run_training(
            model=model,
            teacher=teacher,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epochs=TRIAL_EPOCHS,
            trial=trial,
        )
        return best_accuracy

    return objective


def run_optuna(device):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=10)
    study = optuna.create_study(
        direction="maximize",
        study_name="resnet1_optuna",
        storage=None,
        load_if_exists=False,
        pruner=pruner,
    )

    objective = create_objective(device)
    study.optimize(objective, n_trials=TRIALS, n_jobs=1)

    print("Best trial:")
    print(f"  Value (accuracy): {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "best_value": study.best_trial.value,
                "best_params": study.best_trial.params,
            },
            output_file,
            indent=2,
        )
    print(f"Saved best trial params to {BEST_PARAMS_PATH}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_optuna(device)


if __name__ == "__main__":
    main()
