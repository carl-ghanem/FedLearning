import matplotlib.pyplot as plt
import torch
import intel_extension_for_pytorch as ipex
import torchvision

import torch.nn as nn
from math import log


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import time
import os


#print(f"torch version: {torch.__version__}")

cfg = {"n_epochs":8,
    "batch_size_train":64,
    "batch_size_test":1000,
    "learning_rate":0.1,
    "momentum":0.5,
    "lr_depreciation":0.7
}
cfg["log_interval"] = (20*64)//cfg["batch_size_train"]

fed_cfg = {"nb_clients":8,
           "subbatch_per_client":1}

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
mnist_mean = x.mean().item()
mnist_std = x.std().item()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((mnist_mean,), (mnist_std,)),
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./data/",
        train=True,
        download=True,
        transform=transform,
    ),
    batch_size=cfg["batch_size_train"],
    shuffle=True,
    num_workers=2,         # Let CPU load data in parallel
    pin_memory=True        # Speeds up CPU-to-GPU memory transfers
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./data/",
        train=False,
        download=True,
        transform=transform,
    ),
    batch_size=cfg["batch_size_test"],
    shuffle=False,         # No need to shuffle test data
    num_workers=2,
    pin_memory=True
)

class nNet(nn.Module):
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

class Personal_Net():
    def __init__(self):
        # --- 4. HARDWARE TARGETING & IPEX OPTIMIZATION ---
        self.clf = nNet()
        self.optimizer = torch.optim.SGD(self.clf.parameters(), lr=cfg["learning_rate"], momentum=cfg["momentum"])

        self.device = torch.device("xpu")
        self.clf = self.clf.to(self.device)

        # ---> COMMENT OUT IPEX OPTIMIZE FOR NOW <---
        # self.clf, self.optimizer = ipex.optimize(self.clf, optimizer=self.optimizer, dtype=torch.bfloat16)
        
        self.loss_func = nn.CrossEntropyLoss().to(self.device) # <-- Moved to XPU

        
        
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [] #[i * len(train_loader.dataset) for i in range(cfg["n_epochs"] + 1)]
        self.lr = cfg["learning_rate"]
        self.computation_costs = []
        self.learning_rates = []
        self.Me = 0
        self.total_examples_trained = 0

    def train(self, epoch, nP):
        """
        Docstring for train
        
        :param self: self
        :param epoch (int): epoch number, used for logging performance
        :param Me (int): local client id, used to seperate train data
        :param nP (int): total number of clients in federated learning
        """

        self.learning_rates.append(self.lr)
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr
        print(f"Set learning rate to: {self.lr}")
        self.lr *= cfg["lr_depreciation"]

        self.clf.train()
        train_batches_amnt = len(train_loader)
        tot_example = len(train_loader.dataset)

        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if not ((self.Me*train_batches_amnt)//nP <= batch_idx < ((self.Me+1)*train_batches_amnt)//nP):
                continue

            #batch_x = batch_x.to(device=self.device, memory_format=torch.channels_last)
            batch_x = batch_x.to(device=self.device, non_blocking=True)
            batch_y = batch_y.to(device=self.device, non_blocking=True)
            
            self.optimizer.zero_grad()

            #with torch.amp.autocast(device_type="xpu", enabled=True, dtype=torch.bfloat16):
            logits = self.clf(batch_x)
            loss = self.loss_func(logits, batch_y)


            loss.backward()
            self.optimizer.step()
            self.total_examples_trained += len(batch_x)

            if batch_idx % cfg["log_interval"] == 0 or ((self.Me+1)*train_batches_amnt)//nP == (batch_idx+1):
                print(
                    "Me: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        self.Me, 
                        epoch,
                        self.total_examples_trained-(epoch-1)*(tot_example//nP),
                        tot_example//nP,
                        100.0 * (self.total_examples_trained-(epoch-1)*(tot_example//nP)) / (tot_example//nP),
                        loss.item(),
                    )
                )
                self.train_losses.append(loss.item())
                self.train_counter.append(self.total_examples_trained)
        self.computation_costs.append(self.train_counter[-1])


    def test(self, epoch, append_to_list=False):
        self.clf.eval()
        test_loss = 0
        correct = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # 1. Push data to XPU
                batch_x = batch_x.to(device=self.device, non_blocking=True)
                batch_y = batch_y.to(device=self.device, non_blocking=True)

                # 2. Forward pass on XPU
                logits = self.clf(batch_x)
                test_loss += self.loss_func(logits, batch_y).item()

                # 3. Get predictions and immediately pull back to CPU
                #pred = logits.data.max(1, keepdim=True).squeeze(1).cpu()
                pred = logits.argmax(dim=1).cpu()
                # And then update the 'correct' calculation just below it to:
                
                batch_y_cpu = batch_y.cpu()

                # 4. Safe CPU math
                correct += pred.eq(batch_y_cpu).sum().item()
                #correct += pred.eq(batch_y_cpu.data.view_as(pred)).sum().item()
                y_true.extend(batch_y_cpu.tolist())
                y_pred.extend(pred.tolist())

            test_loss /= len(test_loader)
            test_loss = round(test_loss, 4)
            if append_to_list:
                self.test_losses.append(test_loss)
                self.test_counter.append(self.total_examples_trained)
            print(
                "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    100.0 * correct / len(test_loader.dataset),
                )
            )

        # Compute and save confusion matrix as PNG for this evaluation
        #try:
        labels = list(range(10))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig_cm = plt.figure(figsize=(8, 6))
        ax = fig_cm.gca()
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        fig_cm.suptitle(f"Confusion Matrix   mod_no: {self.Me}\nepoch: {epoch}   loss: {test_loss}")
        os.makedirs("confusion_matrices", exist_ok=True)
        fname = os.path.join("confusion_matrices", f"confusion_{int(time.time())}.png")
        fig_cm.savefig(fname, bbox_inches="tight")
        plt.close(fig_cm)
        print(f"Saved confusion matrix to {fname}")
        #except Exception as e:
        #    print("Could not compute/save confusion matrix:", e)

    def set_eval(self):
        self.clf.eval()


def do_simple_learning():
    Net = Personal_Net()
    total_op_cost = 0
    global_costs = []
    global_losses = []
    for glob_epoch in range(1, cfg["n_epochs"] + 1):
        for subbatch in range(fed_cfg["subbatch_per_client"]):
            Net.train((glob_epoch-1)*fed_cfg["subbatch_per_client"]+subbatch+1, nP=1)

        Net.test(epoch=(glob_epoch-1)*fed_cfg["subbatch_per_client"]+subbatch+1, append_to_list=True)
        print("Computational cost:{} \n loss: {:.4f} \nloss*comp: {:.1f}".format(Net.computation_costs[-1],
                                                                                Net.test_losses[-1],
                                                                                Net.test_losses[-1]*Net.computation_costs[-1]))
        global_costs.append(Net.computation_costs[-1])
        global_losses.append(Net.test_losses[-1])

    with open(file="./output_simple_{}.txt".format(int(time.time())), mode="w") as f:
        lines = []
        lines.append("cfg:")
        lines.append(str(cfg))

        for glob_epoch in range(1, cfg["n_epochs"] + 1):
            lines.append("Epoch: {}".format(glob_epoch))
            lines.append("Global:")
            lines.append("global_loss: {}".format(global_losses[glob_epoch - 1]))
            lines.append("global_cost: {}".format(global_costs[glob_epoch - 1]))
        f.writelines(line + "\n"for line in lines)
    

    fig = plt.figure()
    plt.plot(Net.train_counter, Net.train_losses, color="blue")
    plt.scatter(Net.test_counter, Net.test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    fig.savefig("./training-curve-single.png")


def do_federated_learning():
    Nets = []
    for Me in range(fed_cfg["nb_clients"]):
        Nets.append(Personal_Net())
        Nets[-1].Me = Me
        # Make all client models identical to the first client's model
        base_state = Nets[0].clf.state_dict()
        for c in range(1, len(Nets)):
            cloned_state = {k: v.clone() for k, v in base_state.items()}
            Nets[c].clf.load_state_dict(cloned_state)
            # recreate optimizer to reference the new parameters
            Nets[c].optimizer = torch.optim.SGD(Nets[c].clf.parameters(), lr=Nets[c].lr, momentum=cfg["momentum"])

    for Net in Nets:
        Net.test(epoch=0)
        
    global_costs = []
    global_losses = []
    for glob_epoch in range(1, cfg["n_epochs"] + 1):
        total_op_cost = 0
        for Net in Nets:
            for subbatch in range(fed_cfg["subbatch_per_client"]):
                Net.train((glob_epoch-1)*fed_cfg["subbatch_per_client"]+subbatch+1, nP=fed_cfg["nb_clients"])

            Net.test(epoch=(glob_epoch-1)*fed_cfg["subbatch_per_client"]+subbatch+1, append_to_list=True)
            print("Computational cost:{} \n loss: {:.4f} \nloss*comp: {:.1f}".format(
                Net.computation_costs[-1],
                Net.test_losses[-1],
                Net.test_losses[-1]*Net.computation_costs[-1]))
            
            total_op_cost += Net.computation_costs[-1]
            
        # Federated averaging: compute global model (simple equal-weight average) and broadcast to clients
        client_states = [Nets[c].clf.state_dict() for c in range(len(Nets))]
        global_state = {}
        for k in client_states[0].keys():
            stacked = torch.stack([client_states[i][k].float() for i in range(len(client_states))], dim=0)
            global_state[k] = torch.mean(stacked, dim=0)

        # Load global state into each client and recreate optimizer to match new params
        for c in range(len(Nets)):
            Nets[c].clf.load_state_dict(global_state)
            Nets[c].optimizer = torch.optim.SGD(Nets[c].clf.parameters(), lr=Nets[c].lr, momentum=cfg["momentum"])
        
        global_net = Personal_Net()
        global_net.Me = -1
        global_net.clf.load_state_dict(global_state)
        #global_net.optimizer = torch.optim.SGD(Nets[c].clf.parameters(), lr=Nets[c].lr, momentum=cfg["momentum"])

        print("Federated averaging done: global model aggregated and distributed to all clients.")
        
        global_net.test(epoch=glob_epoch, append_to_list=True)
        global_losses.append(global_net.test_losses[-1])
        global_costs.append(total_op_cost)

        print("Computational cost:{} \n loss: {:.4f} \nloss*comp: {:.1f}".format(
            total_op_cost,
            global_net.test_losses[-1],
            global_net.test_losses[-1]*total_op_cost))
    
    with open(file="./output_fed_{}.txt".format(int(time.time())), mode="w") as f:
        lines = []
        lines.append("cfg:")
        lines.append(str(cfg))
        lines.append("fed_cfg:")
        lines.append(str(fed_cfg))

        for glob_epoch in range(1, cfg["n_epochs"] + 1):
            lines.append("Epoch: {}".format(glob_epoch))
            for id, Net in enumerate(Nets):
                lines.append("client: {}".format(id))
                lines.append("loss: {}".format(Net.test_losses[glob_epoch - 1]))
            lines.append("Global:")
            lines.append("global_loss: {}".format(global_losses[glob_epoch - 1]))
            lines.append("global_cost: {}".format(global_costs[glob_epoch - 1]))
        f.writelines(line + "\n"for line in lines)

    for Me, Net in enumerate(Nets):
        fig = plt.figure()
        plt.plot(Net.train_counter, Net.train_losses, color="blue")
        plt.scatter(Net.test_counter, Net.test_losses, color="red")
        plt.legend(["Train Loss", "Test Loss"], loc="upper right")
        plt.xlabel("number of training examples seen")
        plt.ylabel("negative log likelihood loss")
        fig.savefig("./training-curve-{}.png".format(Me))

do_simple_learning()
#do_federated_learning()

"""
Net_1.set_eval()

fig = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    with torch.no_grad():
        logits = Net_1.clf(img.unsqueeze(0))
        pred = logits.data.max(1, keepdim=True)[1].item()
    fig.add_subplot(rows, cols, i)
    plt.title(f"{label} (predict: {pred})")
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

fig.savefig("./pred-sample-images.png")"""
