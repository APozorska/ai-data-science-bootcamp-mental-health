# Beyond the Symptoms: AI Models for Depression Classification in Mental Health Data

## Table of Contents

- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Full Pipeline](#running-the-full-pipeline)
  - [Training and Tuning](#training-and-tuning)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
- [Examples](#examples)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Project Description

This repository implements a reproducible machine learning pipeline for depression risk classification using anonymized survey data. 
The workflow covers data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and prediction on external datasets.

**Key Features:**
- Modular, config-driven pipeline
- Custom transformers for advanced preprocessing and feature engineering
- Automated model selection and evaluation
- Clear logging and reporting
- Designed for reproducibility and easy experimentation

**Technologies:** 
- Python (`pandas`, `numpy`, `matplotlib`, `scikit-learn`)
- Jupyter Notebook

**Trained Models:**
- Logistic Regression
- Random Forest

*Future work may include: Multi-Layer Perceptron (MLP), XGBoost*

---

## Project Structure
```
ai-data-science-bootcamp-mental-health/
├── config/                         # Main configuration file
├── data/
│   ├── external/                   # External datasets for prediction
│   └── raw/                        # Raw data file
├── depression_classification/
│   ├── data/                       # Data loading utilities
│   ├── models/                     # Model definitions, evaluation, and parameter grids
│   ├── pipeline/                   # Pipeline assembly
│   ├── preprocessing/              # Custom transformers and preprocessing scripts
│   │   └── custom_transformers/    # Custom feature engineering components
│   ├── utils/                      # Config, logging, and helper functions
├── models/                         # Saved models and metadata
├── notebooks/                      # Notebook
├── results/                        # Evaluation metrics and predictions
├── main.py                         # Run the full pipeline (train, evaluate, predict)
├── train_and_tune.py               # Train models and tune hyperparameters
├── evaluate.py                     # Evaluate the best model on the test set
├── predict.py                      # Predict on external data
└── README.md
```
---

## Installation

**Clone the repository:** 

```bash
git clone https://github.com/APozorska/ai-data-science-bootcamp-mental-health.git
cd ai-data-science-bootcamp-mental-health
```
---

## Configuration

All pipeline parameters are managed via `config.yaml`. This includes:
- Data paths and columns
- Feature engineering and selection settings
- Model hyperparameters and search grids
- Cross-validation setup
- Output paths for models, metrics, and predictions

**Example snippet:**  

```yaml
data:
  raw_data_path: 'data/raw/full_data.csv'
  data_splitted_train_path: 'data/train.csv'
  data_splitted_test_path: 'data/test.csv'
models:
  to_test:
    - 'logistic_regression'
    - 'random_forest'
...
```


> **Note:**  
> The location of your config.yaml file is set using the `CFG_PATH` environment variable, which is loaded from a .env file in the project root.  
>
> If you want to run a new experiment or modify the pipeline settings, you **must update the `config.yaml` file** accordingly.  
> Adjust data paths, feature engineering options, model parameters, or any other relevant settings in `config.yaml` before running the scripts.  

---



---

## Usage

### Running the Full Pipeline

To run all steps (training, evaluation, prediction) in sequence:
```
python main.py
```

### Training and Tuning

Train models and perform hyperparameter tuning: 
```
python train_and_tune.py
```

### Evaluation

Evaluate the best model on the test set: 
``` 
python evaluate.py
```

### Prediction

Predict on external data: 
```
python predict.py
```


> **Note:** By default, scripts use the config path from `depression_classification/utils/settings.py` (`CFG_PATH`).  


---

## Examples

**Loading evaluation metrics in a Jupyter Notebook:**

```python
import pandas as pd
metrics = pd.read_json('results/final_evaluation.json')
print(metrics)
```
**Loading predictions:**

```python
preds = pd.read_csv('results/external_data_predictions.csv')
print(preds.head())
```

---
## Contributing

Contributions, suggestions, and issues are welcome!  
Please open an issue or submit a pull request.

---

## License

[MIT License](LICENSE)

---

## Contact

For questions or collaboration, contact:  

**Aleksandra Pozorska**  

- [Email](a.pozorska9@gmail.com)  
- [LinkedIn profile](https://www.linkedin.com/in/aleksandra-pozorska/)  
- [Kaggle profile](https://www.kaggle.com/aleksandrapozorska)


---
