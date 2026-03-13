# IASML - Integrated Analysis System for Machine Learning

IASML is a machine‑learning toolkit and cloud platform for **genomic selection** and **multi‑omics phenotype prediction**.  
It provides a unified command‑line interface and a Shiny‑based web UI to run classical models (SVM, Ridge, Random Forest, LightGBM, XGBoost, PLS, GBM, etc.) and neural networks (CNN, MLP).

---

## 1. Installation

### 1.1 Install from source (recommended)

```bash
# Clone this repository
git clone https://github.com/yourusername/iasml.git
cd iasml

# (Optional but recommended) create an isolated environment
# conda create -n iasml python=3.10
# conda activate iasml

# Install dependencies and the IASML package (CPU TensorFlow by default)
pip install .
```

### 1.2 TensorFlow / Keras notes

- IASML uses `tensorflow.keras` and **does not depend on the standalone `keras` package**.  
- `requirements.txt` pins `tensorflow-cpu==2.10.1`, which works well on most CPU‑only servers or where CUDA/cuDNN is not installed.
- If you have a GPU with a compatible CUDA/cuDNN stack, you may switch to GPU TensorFlow:

```text
# In requirements.txt, replace
tensorflow-cpu==2.10.1
# with
tensorflow==2.10.1
```

Then reinstall:

```bash
pip install -r requirements.txt
```

### 1.3 Run the cloud platform (Shiny app)

To run the web UI (`app.py`), install the IASML stack **and** the cloud-only dependencies:

```bash
# From the project root, with your environment activated:
pip install -r requirements.txt
pip install -r requirements-cloud.txt
```

- **`requirements.txt`**: core IASML (numpy, pandas, scikit-learn, TensorFlow, etc.) — required so that “Start analysis” can run `IASML.py` on the server.
- **`requirements-cloud.txt`**: `shiny` (web framework) and `nest_asyncio` (used by the app).

Then start the app:

```bash
python app.py
```

By default it listens on `http://0.0.0.0:18001`; open that address in your browser.

---

## 2. Command‑line usage (`IASML.py`)

The main entry point is `IASML.py`. It supports:

- **Genomic selection from PLINK genotypes**
- **Multi‑omics phenotype prediction from TXT feature matrices**
- **Phenotype cross‑validation splitting**
- **Custom dimension reduction by individual similarity (DR)**
- **Automated model search via `--gather`**

### 2.1 Required arguments (training / prediction)

- `--phe <filename>`: Phenotype file (tab‑separated, first row = trait names, first column = individual IDs).
- `--phe-pos <col_number>`: Column position (1‑based, including ID) of the target phenotype.
- One of:
  - `--bfile <prefix>`: PLINK binary genotype prefix (`prefix.bed/.bim/.fam`).
  - `--tfile <filename>`: TXT feature matrix (rows = individuals, columns = markers/features).
- Model specification (choose one of):
  - `--model <model_type>`
  - `--model-params <filename>`
  - `--model-frame <keras_model_file>`
  - `--gather ...` (ensemble search)
- `--out <fileprefix>`: Output prefix (used for predictions, model parameters, logs, etc.).

### 2.2 Optional arguments (covariates, threads, CV, DR)

- `--f <col_number>`: Factor (categorical) covariate columns in phenotype file (comma‑separated, optional).
- `--n <col_number>`: Numeric (continuous) covariate columns (comma‑separated, optional).
- `--threads <number>`: Total threads for model training and DR search (default: 8).
- `--n-iter <number>`: Number of parameter combinations in random search (default: 32).
- `--cv-search <number>`: CV folds in hyper‑parameter search (default: 3).
- `--DR ...`: Individual‑similarity dimension reduction methods.
  - `--DR auto` – try **all supported methods** (`euclidean`, `cosine`, `hamming`, `manhattan`, `pearson`, `van_raden`, `yang_grm`, `kl_divergence`) and, for each candidate DR, run model selection / CV, then **choose the single best‑performing DR**.
  - `--DR euclidean cosine` – restrict the search to the given subset and choose the best among them.
  - If multiple methods are specified (and not `auto`), IASML evaluates each method (possibly with hyper‑parameter search) and selects the one with the best CV score.
- `--val <filename>`: Optional internal validation phenotype file to compute Pearson correlation.
- `--Val <filename>` and `--Val-pos <col_number>`: External validation set and column position for Pearson correlation.

---

## 3. Typical workflows

### 3.1 Genomic selection with PLINK genotypes

```bash
python IASML.py \
  --bfile data/genotype_prefix \
  --phe data/phenotype.txt \
  --phe-pos 3 \
  --model random_forest \
  --threads 8 \
  --out wheat_rf
```

This will:
- Convert PLINK genotypes to an internal NumPy representation.
- Train a Random Forest model on the specified trait (column 3, minus ID).
- Save predictions to `wheat_rf_predict.txt`.
- Save model parameters to `wheat_rf_model.txt`.

### 3.2 Multi‑omics phenotype prediction from TXT features

```bash
python IASML.py \
  --tfile data/omics_features.txt \
  --phe data/phenotype.txt \
  --phe-pos 4 \
  --model ridge \
  --threads 8 \
  --out omics_ridge
```

Here, the feature matrix is directly provided as a tab‑separated TXT file.

### 3.3 Using a parameter file (`--model-params`)

```bash
python IASML.py \
  --bfile data/genotype_prefix \
  --phe data/phenotype.txt \
  --phe-pos 3 \
  --model-params configs/rf_params.txt \
  --threads 8 \
  --out wheat_rf_fixed
```

Example `rf_params.txt`:

```text
random_forest
n_estimators: 300
max_depth: 20
min_samples_split: 2
min_samples_leaf: 1
max_features: sqrt
```

The first line is the model type; the following lines are `key: value` pairs.

### 3.4 Using a pre‑trained Keras CNN/MLP (`--model-frame`)

```bash
python IASML.py \
  --tfile data/omics_features.txt \
  --phe data/phenotype.txt \
  --phe-pos 4 \
  --model-frame models/my_cnn_model.keras \
  --out omics_cnn
```

IASML will load the provided Keras model and use it directly for prediction.

### 3.5 Phenotype cross‑validation splitting only

```bash
python IASML.py \
  --phe data/phenotype.txt \
  --split-seed 2025 \
  --cv-split 5 \
  --out split_only
```

This will generate `Ref1.txt ... Ref5.txt` and `Val1.txt ... Val5.txt` for 5‑fold phenotype CV splits and then exit.

---

## 4. The `--gather` ensemble search

IASML can automatically search across multiple base models and DR methods to select the best combination:

```bash
python IASML.py \
  --bfile data/genotype_prefix \
  --phe data/phenotype.txt \
  --phe-pos 3 \
  --model gather \
  --gather all \
  --DR auto \
  --threads 16 \
  --n-iter 64 \
  --cv-search 5 \
  --out wheat_gather
```

In this mode IASML:
- Evaluates several classical models (SVM, Ridge, Lasso, ElasticNet, Random Forest, LightGBM, XGBoost, Linear, PLS, GBM).
- Optionally combines them with multiple DR kernels.
- Chooses the best model+DR combination based on CV negative MSE.
- Trains a final model and outputs predictions and selected hyper‑parameters.

---

## 5. Web / cloud platform

The Shiny app in `app.py` exposes IASML as an interactive web interface:

- **Data upload**: upload PLINK (`bed/bim/fam`) or TXT feature matrices, plus phenotype files.
- **Covariates**: interactively select factor and numeric covariate columns.
- **Model selection**:
  - Predefined models with internal hyper‑parameter search.
  - Custom hyper‑parameter file upload (`--model-params`).
  - Pre‑trained Keras CNN/MLP upload (`--model-frame`).
  - **Ensemble search (`--gather`)**: optionally enable gather and choose which base models participate, or use `ALL` to search over all supported classical models.
- **DR options (`--DR`)**:
  - Multi‑select individual‑similarity DR methods, or choose `auto` so IASML tries all DRs and selects the best.
  - DR is applied only for supported scikit‑learn style models (SVM/Ridge/Lasso/ElasticNet/Random_forest/Linear/PLS/GBM).
- **Validation**:
  - Optionally upload an independent validation phenotype file as `--Val`, with column index auto‑aligned to the training `--phe-pos`, to compute Pearson correlation.
- **Results**:
  - Live log view of `IASML.log` (last lines while training is running).
  - Download prediction file(s) (e.g. `result_predict.txt`).
  - Download model parameter file or saved Keras model (e.g. `result_model.txt` / `*.keras`).

To launch the Shiny app locally, run (example):

```bash
python app.py
```

and open the given URL in your browser.

---

## 6. Supported models (summary)

- **Linear models**: `linear`, `ridge`, `lasso`, `elasticnet`
- **Tree‑based**: `decision_tree`, `random_forest`, `gbm` (GradientBoostingRegressor)
- **Boosting**: `lightgbm`, `xgboost`
- **Latent‑variable**: `pls`
- **Kernel / margin**: `svm`
- **Neural networks**: `cnn`, `mlp` (via Keras)

Each model can be used alone via `--model`, tuned via `--model-params`, or selected automatically via `--gather`.

For detailed parameter grids and internal behavior, see the source code in `IASML.py`.
