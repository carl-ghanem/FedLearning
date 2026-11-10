# FedLearning

Experiments comparing centralized learning and simulated federated averaging on MNIST using PyTorch. The project trains a convolutional neural network, tracks test loss and a computation-cost proxy, and generates learning curves and confusion matrices.

## Features

- A standalone MNIST classifier and a class-based experiment implementation.
- Equal-weight averaging of client model parameters after local training.
- Configurable client count, training epochs, batch sizes, and learning rate.
- Plotting utilities for centralized and federated experiment logs.
- Separate Intel XPU and Intel Extension for PyTorch variants.
- An exploratory PeMS traffic-data loader.

## Project structure

| Path | Purpose |
| --- | --- |
| `code/simple_learning_MIST.py` | Standalone CNN training and sample predictions |
| `code/simple_learning_MIST_class.py` | Centralized and federated experiments on CPU |
| `code/simple_learning_MIST_class_xpu.py` | Variant selecting XPU when available, otherwise CPU |
| `code/simple_learning_MIST_class_ipex.py` | Variant importing Intel Extension for PyTorch |
| `code/plot_data_normal.py` | Plot centralized experiment logs |
| `code/plot_data_fed_avg.py` | Plot federated experiment logs |
| `smart_dataset/open_dyna.py` | Convert PeMS `.dyna` records into sliding-window tensors |
| `docs/` | Supporting papers and course/project material |
| `output_simple_*.txt`, `output_fed_*.txt` | Included experiment logs |

The filenames use `MIST`; the dataset used by the code is **MNIST**.

## Setup

Use a Python 3 environment with `pip`. From the project root:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the core dependencies:

```sh
python -m pip install torch torchvision matplotlib numpy scikit-learn
```

Dependency versions are not pinned in this repository. The accelerator variants require a compatible hardware/software environment; start with the CPU script. The IPEX variant additionally imports `intel_extension_for_pytorch`.

## Run experiments

Run commands from the project root so downloaded data and generated files stay together. MNIST downloads automatically to `data/` on the first run and requires internet access.

### Centralized training

```sh
python code/simple_learning_MIST_class.py
```

The class-based script runs `do_simple_learning()` by default. To use the smaller standalone example:

```sh
python code/simple_learning_MIST.py
```

### Federated training

In `code/simple_learning_MIST_class.py`, replace the active calls near the bottom with:

```python
# do_simple_learning()
do_federated_learning()
```

Then run the same script. Training mode is selected in the source; there is no training-mode command-line flag.

The CPU class-based defaults are:

| Setting | Value |
| --- | --- |
| Epochs | 4 |
| Training / test batch size | 64 / 1,000 |
| Initial configured learning rate | 0.1 |
| Learning-rate decay factor | 0.5 |
| Manual SGD | Enabled |
| Federated clients | 8 |
| Local passes per global epoch | 1 |

Edit `cfg` and `fed_cfg` at the top of the script to change these values. Other variants have different defaults. In the CPU script, manual SGD uses the learning rate after it is decayed within `train()` and does not apply optimizer momentum.

### Plot results

These commands also work with the included logs, without retraining:

```sh
python code/plot_data_normal.py --out centralized-results.png
python code/plot_data_fed_avg.py --out federated-results.png
```

Each plotter selects the most recently modified matching log in the current directory. Use `--file` to select a specific experiment:

```sh
python code/plot_data_fed_avg.py --file output_fed_1771430362.txt --out federated-results.png
```

## Generated outputs

- Class-based experiments: timestamped `output_simple_*.txt` or `output_fed_*.txt`, confusion matrices in `confusion_matrices/`, and training-curve PNGs.
- Standalone example: `training-curve.png` and `pred-sample-images.png`.
- Plotters: PNGs showing loss and global loss multiplied by the recorded cost.

Some image filenames are reused across runs. Preserve outputs separately when comparing experiments.

## PeMS data exploration

`smart_dataset/open_dyna.py` is independent of the MNIST experiments. Install `pandas`, edit its `directory` variable to point to your own PeMS04 data, then run:

```sh
python -m pip install pandas
python smart_dataset/open_dyna.py
```

The `.dyna` files are not included. The loader expects `time`, `entity_id`, `traffic_flow`, `traffic_occupancy`, and `traffic_speed` columns. It creates windows with 12 time steps and prints a batch shaped as `(batch, sensors, time steps, features)`.

## Experimental scope

Federated clients run sequentially in one process. Each client selects a range of batches from a shuffled loader, so this is not a fixed partition of samples across clients. Aggregation uses an equal-weight parameter average. The recorded computational cost counts training examples processed; it does not measure runtime, energy, or communication overhead.

The repository contains research scripts and saved logs, with no pinned environment or automated test suite.
