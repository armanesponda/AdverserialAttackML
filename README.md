# AdverserialAttackML

A machine learning project that trains a neural network classifier on the UCI **Poker Hand** dataset, then evaluates its robustness against **adversarial attacks** using the Fast Gradient Sign Method (FGSM).

## Overview

The project has two stages:

1. **`NeuralNetwork.py`** — Loads and preprocesses the poker hand dataset, trains several `MLPClassifier` (multi-layer perceptron) configurations from scikit-learn, evaluates them, plots training loss curves, and pickles the best-performing model to disk.
2. **`AdversarialAttack.py`** — Loads the pickled scikit-learn model, converts it into an equivalent PyTorch model (copying over weights and biases layer-by-layer), and runs FGSM and targeted-FGSM adversarial attacks against it across a range of epsilon (perturbation strength) values. It logs results, generates comparison plots, and exports CSVs summarizing attack success rates, perturbation norms, and per-sample predictions.

## Technologies Used

- **Python 3** — primary language for the entire project
- **NumPy** — numerical operations and array handling
- **pandas** — data loading, preprocessing, and results tabulation (`DataFrame`, CSV export)
- **scikit-learn** — model training and evaluation:
  - `MLPClassifier` (neural network)
  - `StandardScaler` (feature scaling)
  - `train_test_split`
  - `accuracy_score`, `mean_squared_error`
- **PyTorch** (`torch`, `torch.nn`) — used to reconstruct the trained scikit-learn model as a differentiable PyTorch model so gradients can be computed for the adversarial attack (`nn.Linear`, `nn.ReLU`/`nn.Tanh`/`nn.Sigmoid`, `nn.CrossEntropyLoss`)
- **Matplotlib** — visualizations, including training loss curves and a 2×2 grid of attack-evaluation plots (accuracy vs. epsilon, attack success rate, perturbation norms, evaluation time)
- **pickle** — serializing/deserializing the trained scikit-learn model between scripts
- **itertools** (`product`) — used for hyperparameter combinations
- **Standard library** — `os`, `time`, `datetime` for logging, timing, and file management

### Dataset

- [UCI Poker Hand dataset](https://archive.ics.uci.edu/dataset/158/poker+hand) — 10 classes representing poker hand rankings, described by 5 card suit/rank pairs (`S1..C5`) plus a `CLASS` label. A copy is included as `poker-hand-testing.data` (1,000,000 rows), and the training script also pulls a copy from a [companion dataset repo](https://github.com/armanesponda/PokerHandDataset).

## Project Structure

```
.
├── NeuralNetwork.py          # Preprocessing + MLPClassifier training/evaluation
├── AdversarialAttack.py      # sklearn → PyTorch conversion + FGSM attacks
└── poker-hand-testing.data   # UCI Poker Hand dataset (CSV, no header)
```

## How It Works

### 1. Training (`NeuralNetwork.py`)

- Loads the poker hand data and labels the 10 feature columns (`S1, C1, ..., S5, C5`) plus `CLASS`.
- Standardizes features with `StandardScaler`.
- Splits data 80/20 into train/test sets.
- Trains three `MLPClassifier` configurations (varying activation, learning rate, and hidden layer depth, each with 64 neurons per layer).
- Tracks accuracy, RMSE, MSE, and R² for each configuration.
- Plots each model's training loss curve to `model_history.png`.
- Saves the best-performing model to `best_sklearn_model.pkl`.

### 2. Adversarial Attack (`AdversarialAttack.py`)

- Loads `best_sklearn_model.pkl` and rebuilds it as an equivalent PyTorch `nn.Sequential` model (`SKLearnToPyTorch`), copying weights/biases directly from the sklearn model's `coefs_` and `intercepts_`.
- Implements FGSM from scratch (`FGSMAttacker.fgsm_attack`): computes the loss gradient with respect to the input, then perturbs the input in the direction of the gradient's sign, scaled by epsilon.
- Also implements a **targeted** variant that tries to force predictions toward a specific class.
- Evaluates attack effectiveness across multiple epsilon values (`0` to `0.3`), tracking accuracy, attack success rate, and average L2/L∞ perturbation magnitude.
- Runs a per-target-class evaluation and a sample-level analysis (10 samples) of original vs. adversarial predictions and confidence.
- Generates a 2×2 comprehensive results plot (`results/fgsm_comprehensive_results.png`) and exports several CSVs (`fgsm_results.csv`, `fgsm_experiment_log.csv`, `sample_analysis.csv`, `targeted_attack_results.csv`).

## Usage

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib torch
```

Train the model:

```bash
python NeuralNetwork.py
```

This produces `best_sklearn_model.pkl` and `model_history.png`.

Run the adversarial attack (requires `best_sklearn_model.pkl`, `X_test.npy`, and `y_test.npy` from the training step):

```bash
python AdversarialAttack.py
```

This produces attack result plots and CSVs under `./results` and an experiment log under `./logs`.

## Notes

- `AdversarialAttack.py` expects `X_test.npy` and `y_test.npy` to exist (saved separately from the training step) in order to run.
- Results were originally intended to support an IEEE conference paper, per the script's closing output message.
