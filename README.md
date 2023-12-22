# classification-imbalanced-ds
ML training with Optuna HO for a classification task with imbalanced dataset (example: Pima Indians Diabetes dataset; 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv')

- `Tuning hyperparameters` for the Catboost classifier with `Optuna library`
    - exploring hyper-parameters space to tune the model
    - `allowes to pause searches`, try other combinations of hyper-parameters and then continue the optimization process
    - `adaptive search`: supports a tree-based hyper-parameter search TPESampler 'Tree-structured Parzen Estimator'. This approach relies on Bayesian probabilities to determine which hyper-parameter selections are the most promising and iteratively adjust the search.
- Model explainability:
    - feature importance
    - SHAP
- Using the trained model for final predictions

## Installation
Install requirements into your python environment
`pip install -r requirements.txt`

## Folder structure
```
classification-imbalanced-ds/
└─── data/
└─── output/
    └─── models/
    └─── figures/

└─── src/
    |    modelling_cls.ipynb
    |    utils.py
    |    project_dirs.py
    |    requirements.txt
    |    README.md
config.yml

```

- folder `data`: dataset, preprocessed dataset
- folder `output`: figures, models, predictions
- folder `src`: folder with code files

# Use
1. Follow the steps in the modelling_cls.ipynb notebook
