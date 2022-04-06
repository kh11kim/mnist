import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# set parameters
epochs = 10
lr = 1e-3
batch_size = 32
chkpt_dir = "./checkpoint"
log_dir = "./log"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

# define network


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(in_features=3136, out_features=10, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

# save and load network


def save(chkpt_dir: str, net: Net, optim: optim.Adam, epoch: int):
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    net_dict = net.state_dict()
    optim_dict = optim.state_dict()
    model_dict = dict(net=net_dict, optim=optim_dict)
    save_path = f"{chkpt_dir}/model_epoch_{epoch}.pth"
    torch.save(model_dict, save_path)


def load(chkpt_dir: str, net: Net, optim: optim.Adam):
    chkpt_list = os.listdir(chkpt_dir)
    filename_lst = [os.path.splitext(path)[0] for path in chkpt_list]
    idxs = [filename.split("_")[-1] for filename in filename_lst]
    last_idx = np.argmax(idxs)
    model_dict = torch.load(f"{chkpt_dir}/{chkpt_list[last_idx]}")

    net.load_state_dict(model_dict["net"])
    optim.load_state_dict(model_dict["optim"])


# get datasets
mnist_train = datasets.MNIST(
    root="./",
    download=True,
    train=True,
    transform=transforms.ToTensor()
)
loader = DataLoader(
    mnist_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

# set network, optimizer, tensorboard writer
net = Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(params=net.parameters(), lr=lr)
writer = SummaryWriter(log_dir=log_dir)

# train
for epoch in range(1, epochs+1):
    loss_arr = []
    for batch, (input, label) in enumerate(loader):
        input.to(device)
        label.to(device)

        optimizer.zero_grad()
        pred = net(input)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        loss_arr.append(loss.item())
        print(
            f"train: EPOCH {epoch}/{epochs} | BATCH {batch}/{len(loader)} | Loss {np.mean(loss_arr):.4f}")

    writer.add_scalar("loss", np.mean(loss_arr), epoch)

writer.close()
