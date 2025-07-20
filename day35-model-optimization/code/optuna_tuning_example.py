# Optuna Tuning Example (LightGBM)
import optuna
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 16, 64),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0)
    }

    lgb_train = lgb.Dataset(X, y)
    cv = lgb.cv(params, lgb_train, nfold=3, early_stopping_rounds=10, seed=42)
    return min(cv['binary_logloss-mean'])

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best Trial:")
print(study.best_trial.params)
