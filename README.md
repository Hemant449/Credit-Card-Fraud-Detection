# Credit Card Fraud Detection

I built this project to understand how different machine-learning approaches behave when the fraud class is extremely rare. I kept the original EDA on transaction amounts, timing, class counts, and correlations, then compared models using cross-validation and precision–recall analysis.

## Project goals

- Explore legitimate and fraudulent transaction patterns.
- Transform the skewed transaction `Amount` feature using `log1p`.
- Establish Logistic Regression as a baseline.
- Compare Random Forest trained on the original class distribution, Random Forest with `class_weight="balanced_subsample"`, and Random Forest with default SMOTE.
- Use five-fold stratified cross-validation and out-of-fold probabilities to compare models and select a decision threshold.
- Evaluate the selected model and threshold once on the held-out test set.

## Dataset

The notebook uses the public [ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The commonly used version contains 284,807 transactions and 492 fraud cases. The `Class` column is the target: `0` means legitimate and `1` means fraud. Features `V1`–`V28` are anonymized PCA-transformed features.

The CSV is **not included** in this repository. The notebook first checks for `data/creditcard.csv`; if it is not there, it attempts to download the dataset with KaggleHub. If automatic download does not work in your environment, download the CSV from Kaggle and put it in the `data/` folder. Review and follow the dataset page's terms before redistributing the data.

## Repository structure

```text
credit-card-fraud-detection-github/
├── credit_card_fraud_detection.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── README.md
```

## Setup

Use Python 3.10 or newer. From the repository folder, create and activate a virtual environment.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `credit_card_fraud_detection.ipynb` and run the cells from top to bottom.

### GPU note

The notebook's first code cell contains `%load_ext cuml.accel`, which is optional and requires a compatible NVIDIA GPU environment with RAPIDS cuML installed. For GPU use, follow the [official cuML installation instructions](https://docs.rapids.ai/install/) for your driver, CUDA, Python, and operating-system combination. If you're running on CPU or do not have cuML installed, comment out the first `%load_ext cuml.accel` line and run the rest of the notebook normally. The standard `requirements.txt` intentionally does not install RAPIDS because its installation depends on the system's GPU/CUDA setup.

## Modeling workflow

1. Split the data into stratified training and test sets.
2. Fit preprocessing within each model pipeline to keep learned transformations inside each training fold.
3. Compare Logistic Regression, standard Random Forest, `balanced_subsample`, and default SMOTE using the same five-fold stratified cross-validation folds.
4. Generate out-of-fold fraud probabilities from training data.
5. Select a model and threshold using out-of-fold results. The current target is fraud precision of at least 90% and recall greater than 80%; these targets are goals, not guaranteed outcomes.
6. Refit the selected model on all of `X_train` and evaluate it on `X_test` once.

SMOTE is contained in an `imblearn` pipeline so it is applied to each fold's training portion only. The validation fold and final test set retain the natural class distribution.

## Metrics

I focus on fraud precision, fraud recall, F1, ROC-AUC, Average Precision, false positives, missed frauds, and the confusion matrix. Accuracy alone is not a useful selection metric for a dataset where legitimate transactions dominate.

The threshold is estimated from out-of-fold scores, while the final model is refitted on all training data. The score distribution can shift after refitting, so the final test metrics show how well that selected threshold transfers. The test set is for final reporting and should not be used to retune the model or threshold.

## Results

Run the notebook to generate the actual metrics and plots for your environment. This README does not hard-code performance numbers, because results should come from the current reproducible run rather than copied outputs from an earlier experiment.

## Limitations

- The dataset represents a limited historical period and its `V1`–`V28` features are anonymized.
- A random stratified split does not fully simulate production performance over time; a time-aware evaluation would be a useful next step.
- The precision/recall target reflects a chosen operating requirement. A real system would also need to consider the cost of false alarms, missed fraud, and human review capacity.
