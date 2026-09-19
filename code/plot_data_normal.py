import argparse
import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

def find_latest_output(base_dir="."):
    files = glob.glob(os.path.join(base_dir, "output_simple_*.txt"))
    if not files:
        raise FileNotFoundError("No output_simple_*.txt files found in {}".format(base_dir))
    return max(files, key=os.path.getmtime)

def parse_output_file(path):
    epochs = []
    clients_per_epoch = []  # list of dicts {client_id: loss} (may be empty for non-federated)
    global_losses = []
    global_costs = []
    current_client = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Epoch:"):
                epochs.append(int(line.split(":",1)[1].strip()))
                clients_per_epoch.append({})
                current_client = None
            elif line.startswith("client:"):
                try:
                    current_client = int(line.split(":",1)[1].strip())
                except:
                    current_client = None
            elif line.startswith("loss:") and current_client is not None:
                try:
                    val = float(line.split(":",1)[1].strip())
                    clients_per_epoch[-1][current_client] = val
                except:
                    pass
            elif line.startswith("global_loss:"):
                try:
                    global_losses.append(float(line.split(":",1)[1].strip()))
                except:
                    global_losses.append(np.nan)
            elif line.startswith("global_cost:"):
                try:
                    global_costs.append(float(line.split(":",1)[1].strip()))
                except:
                    global_costs.append(np.nan)

    n_epochs = len(epochs)
    # determine number of clients if any
    max_client = -1
    for d in clients_per_epoch:
        if d:
            max_client = max(max_client, max(d.keys()))
    nb_clients = max_client + 1 if max_client >= 0 else 0

    clients_losses = []
    for cid in range(nb_clients):
        arr = []
        for ep in range(n_epochs):
            arr.append(clients_per_epoch[ep].get(cid, np.nan))
        clients_losses.append(np.array(arr, dtype=float))

    # pad global arrays if needed
    if len(global_losses) != n_epochs:
        global_losses = (global_losses + [np.nan]*n_epochs)[:n_epochs]
    if len(global_costs) != n_epochs:
        global_costs = (global_costs + [np.nan]*n_epochs)[:n_epochs]

    return np.array(epochs, dtype=int), clients_losses, np.array(global_losses, dtype=float), np.array(global_costs, dtype=float)

def plot_results(epochs, clients_losses, global_losses, global_costs, out_path):
    if len(epochs) == 0:
        raise RuntimeError("No epochs found to plot.")
    x = epochs
    fig, axes = plt.subplots(1, 2, figsize=(14,5), constrained_layout=True)
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    # plot clients if any
    for i, losses in enumerate(clients_losses):
        ax.plot(x, losses, marker="o", linestyle="-", label=f"Client {i}", color=cmap(i%10))
    # plot global (single) model loss
    ax.plot(x, global_losses, marker="s", linestyle="--", color="k", linewidth=2, label="Model (global)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Per-client (if any) and model loss")
    ax.grid(alpha=0.4, linestyle="--")
    ax.legend(loc="best")

    ax2 = axes[1]
    product = global_losses * global_costs
    ax2.plot(x, product, marker="D", color="tab:purple")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Global loss × cost")
    ax2.set_title("Global loss × cost per epoch")
    ax2.grid(alpha=0.4, linestyle="--")
    for xi, yi in zip(x, product):
        if not np.isnan(yi):
            ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0,6), ha="center", fontsize=8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot non-federated (simple) results from output_simple_*.txt")
    parser.add_argument("--file", help="Path to output_simple_*.txt (default: latest)", default=None)
    parser.add_argument("--out", help="Output PNG path", default=None)
    args = parser.parse_args()

    path = args.file if args.file else find_latest_output(".")
    epochs, clients_losses, global_losses, global_costs = parse_output_file(path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_default = os.path.splitext(os.path.basename(path))[0] + f"_plot_{ts}.png"
    out_path = args.out if args.out else out_default
    plot_results(epochs, clients_losses, global_losses, global_costs, out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
