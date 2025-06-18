# DynamicTTR

# Installation

You'd better to make sure the python version is `3.10`.
Run on the command in the project, and install the dependency:

```shell
pip install -r requirements.txt
```

# Dataset
For the experiment, please pre-download the `EthereumHeist` dataset.
See: https://xblock.pro/#/dataset/46

# Run the code in experiments
Comparative experiments:
```shell
python comp.py --raw_path=/path/to/your/dataset
```

Parameters Sensitivity Analysis:
```shell
python sens.py --raw_path=/path/to/your/dataset
```

Ablation Study:
```shell
python abl.py --raw_path=/path/to/your/dataset
```

# Experiments in the wild
Please refer to the `wild` folder for the experiments in the wild.