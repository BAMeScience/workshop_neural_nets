## Codebase for the Workshop "Your first Neural Network in PyTorch – An Introduction"

This repository contains all the necessary code for the Workshop "Your first Neural Network in PyTorch – An Introduction".
The material follows the workshop structure closely, the final code includes two simple neural networks for classifying digits. For training and validation, the MNIST dataset is downloaded und used.

To get everything going, it is recommended to clone this repository and install all packages with the provided .toml file. Thereto, open the terminal (VDI/Linux) or the command prompt (Windows). In the terminal:
- Navigate to your designated project path (`cd PROJECTPATH`) and create a new folder (`mkdir FOLDERNAME`) 
- Enter the new folder with `cd FOLDERNAME` and clone this repository using `git clone https://github.com/BAMeScience/workshop_neural_nets.git` or download the repository by hand and unzip in the new folder
- In the terminal, from your project path, execute `cd workshop_neural_nets` (you are now in the workshop folder)
- Install uv if it is not already installed (`python -m pip install uv`)
- Create .venv and install the locked dependencies by simply executing `uv sync --locked`. If this message shows up: "The lockfile at `uv.lock` needs to be updated, but `--locked` was provided. To update the lockfile, run `uv lock`", run `uv lock` followed by `uv sync --locked` to install all packages.
- You can now activate the new environment with `source .venv/bin/activate` (Linux) or `.venv\Scripts\activate` (Windows)
- Once activated, we can install new packages with 'uv add PACKAGENAME'. Not necessary for this workshop, all packages should already have been installed
- To find the right kernel for your Jupyter Notebook, you might need to navigate to your workspace folder and run (all in terminal) `uv run python -m ipykernel install --user --name workshop --display-name "Workshop"`. The kernel shoud now appear under the name "Workshop".
- To monitor training progress with TensorBoard, first install TensorBoard (if it is not already installed) and then start it from the project directory:
```bash
uv pip install tensorboard
tensorboard --logdir=runs
``