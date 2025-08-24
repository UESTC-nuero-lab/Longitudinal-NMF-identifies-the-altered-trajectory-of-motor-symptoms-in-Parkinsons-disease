import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import pearsonr, spearmanr, kstest, t
from joblib import Parallel, delayed
from itertools import product
from sklearn import metrics

# ================== setting the fixed random seed ==================
RANDOM_STATE = fixed_random_seed
# ===================================================================

scaleName = "MDS-UPDRS-Ⅱ" # or "MDS-UPDRS-Ⅲ"

dataset = pd.read_excel('your/path/to/data')
X = dataset[["W1","W2","W3","W4","W5","W6","W7"]]
y = dataset[scaleName].values
n_features = X.shape[1]

# ================== setting the split and cv seed ==================
split_seed = fixed_split_seed
cv_seed = fixed_cv_seed
# ================================================

# ================== mesh grid for params ==================
N_JOBS = -1
PARAM_GRID = {
    "n_estimators": [50, 100, 150, 200],  # number of trees
    "max_depth": [3, 5, 7, 9],  # maximum tree depth
    "learning_rate": [0.01, 0.05, 0.1, 0.3],  # learning rate
    "subsample": [0.6, 0.8, 1.0],  # sample ratio of training data
    "colsample_bytree": [0.6, 0.8, 1.0],  # sample ratio of features
    "gamma": [0, 0.1, 0.3]  # minimum loss reduction required to make a further partition
}
FIXED_PARAMS = {
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "objective": "reg:squarederror"
}
# ================================================

# generate all parameter combinations
param_combinations = [
    dict(zip(PARAM_GRID.keys(), vals)) for vals in product(*PARAM_GRID.values())
]

def evaluate_params(params, X_train, y_train, cv_seed):
    """ Evaluate a single parameter combination using 5-fold cross-validation."""
    kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    y_true, y_pred = [], []

    # combine fixed and variable parameters
    model_params = {**FIXED_PARAMS, **params}

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # train the model
        model = XGBRegressor(**model_params).fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        y_true.extend(y_val_fold)
        y_pred.extend(model.predict(X_val_fold))
    r, p = spearmanr(y_true, y_pred)
    # calculate R² and MSE
    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    return params, r2, r, p, (y_true, y_pred), mse


# ================== optimizing the params ==================
print(f"\nStarting grid search with {len(param_combinations)} parameter combinations")
final_results = []

# split the dataset into training(60) and testing sets(18)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=18, random_state=split_seed
)
# parallel evaluation of all parameter combinations
results = Parallel(n_jobs=N_JOBS)(
    delayed(evaluate_params)(params, X_train, y_train, X_test, y_test, cv_seed)
    for params in param_combinations
)

# select the best parameters based on MSE
best_r2 = -float('inf')
best_r = best_p = -1
best_mse = float('inf')
best_params = {}
best_y_true, best_y_pred = [], []

for params, r2, r, p, (y_true, y_pred), mse in results:
    if mse > best_mse:
        best_r2, best_r, best_p, best_mse = r2, r, p, mse
        best_params = params
        best_y_true, best_y_pred = y_true, y_pred

# train the final model with the best parameters
final_model = XGBRegressor(
    **{**FIXED_PARAMS, **best_params, "n_jobs": N_JOBS}
).fit(X_train, y_train)

y_pred_test = final_model.predict(X_test)

# test set evaluation
test_r2 = metrics.r2_score(y_test, y_pred_test)
test_y_normal = kstest(y_test, 'norm')
test_pred_normal = kstest(y_pred_test, 'norm')
if test_y_normal.pvalue < 0.05 or test_pred_normal.pvalue < 0.05:
    test_corr, test_p_value = spearmanr(y_test, y_pred_test)
    corr_type = "Spearman rho"
else:
    test_corr, test_p_value = pearsonr(y_test, y_pred_test)
    corr_type = "Pearson r"

final_results.append({
    "split_seed": split_seed,
    "cv_seed": cv_seed,
    "best_r2": best_r2,
    "best_r": best_r,
    "best_p": best_p,
    "best_mse": best_mse,
    "best_params": best_params,
    "best_y_true": best_y_true,
    "best_y_pred": best_y_pred,
    "test_r2": test_r2,
    "test_r": test_corr,
    "test_p": test_p_value,
    "test_mse": metrics.mean_squared_error(y_test, y_pred_test),
    "corr_type": corr_type
})

# ================== print outcomes ==================
print("\n=== Global Optimal Parameters ===")
print(f"Split seed: {final_results['split_seed']}")
print(f"CV seed: {final_results['cv_seed']}")
print(f"n_estimators: {final_results['best_params']['n_estimators']}")
print(f"max_depth: {final_results['best_params']['max_depth']}")
print(f"learning_rate: {final_results['best_params']['learning_rate']}")
print(f"subsample: {final_results['best_params']['subsample']}")
print(f"colsample_bytree: {final_results['best_params']['colsample_bytree']}")
print(f"Gamma: {final_results['best_params']['gamma']}")

# calculate cross-validation metrics
cv_r2 = final_results["best_r2"]
cv_r = final_results["best_r"]
cv_p = final_results["best_p"]
cv_mse = final_results["best_mse"]

# interval for confidence
n = len(final_results["best_y_true"])
df = n - 2
stderr = np.sqrt((1 - cv_r ** 2) / (n - 2))
ci = t.interval(0.95, df, loc=cv_r, scale=stderr)

print("\n=== Cross-Validation Results (5-fold) ===")
print(f"R²: {cv_r2:.4f}")
print(f"MSE: {cv_mse:.4f}")
print(f"Correlation: {cv_r:.4f}, p-value: {cv_p:.4g}")
print(f"95% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")

# 输出测试集结果
test_r2 = final_results["test_r2"]
test_r = final_results["test_r"]
test_p = final_results["test_p"]
test_mse = final_results["test_mse"]

print("\n=== Independent Test Results (18 samples) ===")
print(f"R²: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"Correlation: {test_r:.4f}, p-value: {test_p:.4g}")