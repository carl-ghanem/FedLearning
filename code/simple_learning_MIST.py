import matplotlib.pyplot as plt
import torch
import torchvision

import torch.nn as nn

print(f"torch version: {torch.__version__}")

cfg = {"n_epochs":3,
    "batch_size_train":64,
    "batch_size_test":1000,
    "learning_rate":0.01,
    "momentum":0.5,
    "log_interval":10,
}

torch.manual_seed(0)

train_data = torchvision.datasets.MNIST(
    root="./data/",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_data = torchvision.datasets.MNIST(
    root="./data/",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

print(train_data)
print(f"The training dataset has shape: {train_data.data.size()}")
print(test_data)
print(f"The test dataset has shape: {test_data.data.size()}")

x = torch.cat([train_data[i][0] for i in range(len(train_data))], dim=0)


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./data/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((x.mean().item(),), (x.std().item(),)),
            ]
        ),
    ),
    batch_size=cfg["batch_size_train"],
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((x.mean().item(),), (x.std().item(),)),
            ]
        ),
    ),
    batch_size=cfg["batch_size_test"],
    shuffle=True,
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool(self.conv1(x)))
        x = self.relu(self.maxpool(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


clf = Net()
loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    clf.parameters(), lr=cfg["learning_rate"], momentum=cfg["momentum"]
)


train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(cfg["n_epochs"] + 1)]


def train(epoch):
    clf.train()
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = clf(batch_x)
        loss = loss_func(logits, batch_y)
        loss.backward()
        optimizer.step()
        if batch_idx % cfg["log_interval"] == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(batch_x),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * cfg["batch_size_train"])
                + ((epoch - 1) * len(train_loader.dataset))
            )


def test():
    clf.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = clf(batch_x)
            test_loss += loss_func(logits, batch_y).item()
            pred = logits.data.max(1, keepdim=True)[1]
            correct += pred.eq(batch_y.data.view_as(pred)).sum()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


test()
for epoch in range(1, cfg["n_epochs"] + 1):
    train(epoch)
    test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color="blue")
plt.scatter(test_counter, test_losses, color="red")
plt.legend(["Train Loss", "Test Loss"], loc="upper right")
plt.xlabel("number of training examples seen")
plt.ylabel("negative log likelihood loss")
fig.savefig("./training-curve.png")


clf.eval()

fig = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    with torch.no_grad():
        logits = clf(img.unsqueeze(0))
        pred = logits.data.max(1, keepdim=True)[1].item()
    fig.add_subplot(rows, cols, i)
    plt.title(f"{label} (predict: {pred})")
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

fig.savefig("./pred-sample-images.png")
