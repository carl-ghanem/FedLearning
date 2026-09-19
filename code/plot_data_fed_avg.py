import argparse
import glob
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np



def find_latest_output(base_dir="."):
    files = glob.glob(os.path.join(base_dir, "output_fed_*.txt"))
    if not files:
        raise FileNotFoundError("No output_fed_*.txt files found in {}".format(base_dir))
    return max(files, key=os.path.getmtime)

def parse_output_file(path):
    clients_per_epoch = []  # list of dicts: {client_id: loss}
    global_losses = []
    global_costs = []
    current_client = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Epoch:"):
                # start a new epoch dict
                clients_per_epoch.append({})
                current_client = None
            elif line.startswith("client:"):
                try:
                    current_client = int(line.split(":", 1)[1].strip())
                except:
                    current_client = None
            elif line.startswith("loss:"):
                try:
                    val = float(line.split(":", 1)[1].strip())
                except:
                    continue
                if current_client is not None and clients_per_epoch:
                    clients_per_epoch[-1][current_client] = val
                else:
                    # could be global loss if not associated with a client
                    pass
            elif line.startswith("global_loss:"):
                try:
                    global_losses.append(float(line.split(":", 1)[1].strip()))
                except:
                    global_losses.append(np.nan)
            elif line.startswith("global_cost:"):
                try:
                    global_costs.append(float(line.split(":", 1)[1].strip()))
                except:
                    global_costs.append(np.nan)
            # else ignore other lines
    # determine number of epochs
    n_epochs = len(clients_per_epoch)
    # determine nb clients
    max_client = -1
    for d in clients_per_epoch:
        if d:
            max_client = max(max_client, max(d.keys()))
    nb_clients = max_client + 1 if max_client >= 0 else 0

    # build per-client loss arrays
    clients_losses = []
    for cid in range(nb_clients):
        arr = []
        for ep in range(n_epochs):
            arr.append(clients_per_epoch[ep].get(cid, np.nan))
        clients_losses.append(np.array(arr, dtype=float))

    # ensure global lists length match n_epochs
    if len(global_losses) != n_epochs:
        # try to align by filling with nan if shorter
        global_losses = (global_losses + [np.nan]*n_epochs)[:n_epochs]
    if len(global_costs) != n_epochs:
        global_costs = (global_costs + [np.nan]*n_epochs)[:n_epochs]

    return clients_losses, np.array(global_losses, dtype=float), np.array(global_costs, dtype=float)

def plot_results(clients_losses, global_losses, global_costs, out_path):
    n_epochs = max(1, max([len(arr) for arr in clients_losses]) if clients_losses else len(global_losses))
    epochs = np.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Left: per-client losses + global loss
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for i, losses in enumerate(clients_losses):
        ax.plot(epochs, losses, marker="o", linestyle="-", label=f"Client {i}", color=cmap(i % 10), alpha=0.9)
    if global_losses.size > 0:
        ax.plot(epochs, global_losses, marker="s", linestyle="--", color="k", linewidth=2.2, label="Federated (global)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_title("Per-client test loss and federated (global) loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    # Right: global loss * cost vs epoch
    ax2 = axes[1]
    product = global_losses * global_costs
    ax2.plot(epochs, product, marker="D", linestyle="-", color="tab:purple")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Global loss × cost")
    ax2.set_title("Global loss × cost per epoch")
    ax2.grid(True, linestyle="--", alpha=0.4)
    # annotate points with values (rounded) for readability
    for x, y in zip(epochs, product):
        if not np.isnan(y):
            ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,6), ha="center", fontsize=8)

    # Save figure
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot federated learning results from output_fed_*.txt")
    parser.add_argument("--file", help="Path to output_fed_*.txt (if not provided, uses latest in cwd)", default=None)
    parser.add_argument("--out", help="Output PNG path", default=None)
    args = parser.parse_args()

    path = args.file if args.file else find_latest_output(".")
    clients_losses, global_losses, global_costs = parse_output_file(path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_default = os.path.splitext(os.path.basename(path))[0] + f"_plot_{ts}.png"
    out_path = args.out if args.out else out_default

    plot_results(clients_losses, global_losses, global_costs, out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()