import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
import joblib
import sys
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna

# Import feature engineering functions from M2
# Assuming m2_fashion_mnist_feature_engineering.py is in the same directory
sys.path.append('.')
from m2_fashion_mnist_feature_engineering import normalize_images, extract_hog_features

# Import only the models we can optimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Create directories for outputs
os.makedirs('m3_outputs', exist_ok=True)
os.makedirs('m3_outputs/automl', exist_ok=True)
os.makedirs('m3_outputs/hyperopt', exist_ok=True)
os.makedirs('m3_outputs/models', exist_ok=True)

# Define class names for better readability
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# -------------------------------------------------------------------------
# Part 1: Apply Feature Engineering from M2
# -------------------------------------------------------------------------

# Use a smaller sample size for faster processing
sample_size = 5000
random_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
X_train_sample = X_train[random_indices]
y_train_sample = y_train[random_indices]

# Apply feature engineering from M2
print("\nApplying feature engineering from M2...")
# 1. Normalize the data
print("1. Applying normalization...")
X_train_norm, _ = normalize_images(X_train_sample, method='minmax')
X_test_norm, _ = normalize_images(X_test, method='minmax')

# 2. Extract HOG features
print("2. Extracting HOG features...")
X_train_hog, _ = extract_hog_features(X_train_norm)
X_test_hog, _ = extract_hog_features(X_test_norm)

print(f"HOG features shape: {X_train_hog.shape}")

# Split the data for validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_hog, y_train_sample, test_size=0.2, random_state=42, stratify=y_train_sample
)

print(f"Training set: {X_train_final.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test_hog.shape}")

# -------------------------------------------------------------------------
# Part 2: Model Selection with LazyPredict (an open-source AutoML library)
# -------------------------------------------------------------------------

# Define our supported models for optimization
SUPPORTED_MODELS = [
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'SVC',
    'KNeighborsClassifier',
    'LogisticRegression',
    'MLPClassifier',
    'AdaBoostClassifier',
    'SGDClassifier'
]

print("\nStarting model selection with LazyPredict AutoML...")

try:
    from lazypredict.Supervised import LazyClassifier
    
    # Initialize LazyPredict classifier
    lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    
    # Fit multiple models
    print("Training multiple models with LazyPredict AutoML...")
    models, predictions = lazy_clf.fit(X_train_final, X_val, y_train_final, y_val)
    
    # Filter models to only include supported ones
    supported_models = models[models.index.isin(SUPPORTED_MODELS)]
    
    if len(supported_models) == 0:
        print("No supported models found in LazyPredict results. Using fallback approach.")
        raise ValueError("No supported models in results")
    
    # Print model comparison of supported models
    print("\nAutoML Results - Supported Model Comparison:")
    print(supported_models)
    
    # Save AutoML results
    supported_models.to_csv('m3_outputs/automl/lazypredict_supported_comparison.csv')
    models.to_csv('m3_outputs/automl/lazypredict_all_comparison.csv')
    
    # Select the best model based on validation accuracy (from supported models)
    best_model_name = supported_models.index[0]  # First model has the highest accuracy
    best_accuracy = supported_models.iloc[0]['Accuracy']
    print(f"\nBest supported model selected by AutoML: {best_model_name}")
    print(f"Validation accuracy: {best_accuracy:.4f}")
    
    # Save AutoML results as JSON for easier processing
    automl_results = {
        'best_model': best_model_name,
        'validation_accuracy': float(best_accuracy),
        'models_evaluated': len(models),
        'supported_models_evaluated': len(supported_models),
        'top_models': supported_models.head(5).index.tolist(),
        'top_models_accuracy': supported_models.head(5)['Accuracy'].tolist()
    }
    
    with open('m3_outputs/automl/automl_results.json', 'w') as f:
        json.dump(automl_results, f, indent=4)
    
    # Save visual comparison of top models
    plt.figure(figsize=(12, 8))
    top_models = supported_models.head(min(10, len(supported_models)))
    sns.barplot(x=top_models['Accuracy'], y=top_models.index)
    plt.title('Top Supported Models by Accuracy')
    plt.xlabel('Validation Accuracy')
    plt.tight_layout()
    plt.savefig('m3_outputs/automl/top_models_comparison.png')
    
    # Map of model names to their classes (only supported models)
    model_mapping = {
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVC': SVC(probability=True, random_state=42),
        'MLPClassifier': MLPClassifier(random_state=42, max_iter=300),
        'SGDClassifier': SGDClassifier(random_state=42, max_iter=1000)
    }
    
    # Get the best model
    best_model = model_mapping[best_model_name]
    # Train the best model again to save it
    best_model.fit(X_train_final, y_train_final)
    # Save the best model
    joblib.dump(best_model, 'm3_outputs/models/best_automl_model.joblib')
    print(f"Best model saved to 'm3_outputs/models/best_automl_model.joblib'")
    
except ImportError:
    print("LazyPredict not installed. Installing it now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lazypredict"])
    print("LazyPredict installed successfully. Please run the script again.")
    sys.exit(1)
except Exception as e:
    print(f"Error during AutoML model selection: {str(e)}")
    print("Falling back to a simpler approach with supported models only...")
    
    # Define models that we know we can optimize (simplified approach)
    models = {
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'MLPClassifier': MLPClassifier(random_state=42, max_iter=300)
    }
    
    # Evaluate each model
    results = {}
    best_model_name = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        model.fit(X_train_final, y_train_final)
        y_val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        results[name] = {'accuracy': float(accuracy)}
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model
    
    # Save results
    with open('m3_outputs/automl/fallback_model_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save the best model
    joblib.dump(best_model, 'm3_outputs/models/best_automl_model.joblib')
    
    # Create AutoML results
    automl_results = {
        'best_model': best_model_name,
        'validation_accuracy': float(best_accuracy),
        'models_evaluated': len(models),
        'note': 'Fallback approach used with supported models only'
    }
    
    with open('m3_outputs/automl/automl_results.json', 'w') as f:
        json.dump(automl_results, f, indent=4)
    
    print(f"\nBest model: {best_model_name} with validation accuracy: {best_accuracy:.4f}")

# Load the best model selected by AutoML
best_model = joblib.load('m3_outputs/models/best_automl_model.joblib')
best_model_name = best_model.__class__.__name__
print(f"Loaded {best_model_name} for hyperparameter optimization")

# -------------------------------------------------------------------------
# Part 3: Hyperparameter Optimization with Optuna
# -------------------------------------------------------------------------

print("\nPerforming hyperparameter optimization with Optuna on the best model...")

# Define an objective function for Optuna based on model type
def objective(trial):
    if 'RandomForestClassifier' in best_model_name:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    
    elif 'GradientBoostingClassifier' in best_model_name:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }
        model = GradientBoostingClassifier(**params, random_state=42)
    
    elif 'SVC' in best_model_name:
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'degree': trial.suggest_int('degree', 2, 5) if trial.params.get('kernel') == 'poly' else 3
        }
        model = SVC(**params, random_state=42, probability=True)
    
    elif 'KNeighborsClassifier' in best_model_name:
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2)
        }
        model = KNeighborsClassifier(**params)
    
    elif 'LogisticRegression' in best_model_name:
        params = {
            'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
        }
        model = LogisticRegression(**params, random_state=42, max_iter=1000)
    
    elif 'MLPClassifier' in best_model_name:
        params = {
            'hidden_layer_sizes': (trial.suggest_int('hidden_layer_size', 50, 200),),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        }
        model = MLPClassifier(**params, random_state=42, max_iter=300)

    elif 'AdaBoostClassifier' in best_model_name:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0),
            'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
        }
        model = AdaBoostClassifier(**params, random_state=42)
    
    elif 'SGDClassifier' in best_model_name:
        params = {
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'loss': trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber']),
            'max_iter': trial.suggest_int('max_iter', 500, 2000),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
        }
        model = SGDClassifier(**params, random_state=42)
    
    else:
        # This should never happen with our filtered approach
        raise ValueError(f"Model {best_model_name} not supported for hyperparameter tuning")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_final, y_train_final, cv=3, scoring='accuracy')
    return cv_scores.mean()

# Create and optimize study
n_trials = 30  # Number of hyperparameter combinations to try
study = optuna.create_study(direction='maximize')
print(f"Running {n_trials} optimization trials...")

try:
    study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30-minute timeout
except Exception as e:
    print(f"Optimization stopped early: {str(e)}")

# Get best parameters
best_params = study.best_params
best_score = study.best_value
print(f"Best accuracy achieved with Optuna: {best_score:.4f}")
print(f"Best parameters: {best_params}")

# Train final model with best parameters
print(f"Creating final model ({best_model_name}) with optimized parameters...")
if 'RandomForestClassifier' in best_model_name:
    final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
elif 'GradientBoostingClassifier' in best_model_name:
    final_model = GradientBoostingClassifier(**best_params, random_state=42)
elif 'SVC' in best_model_name:
    final_model = SVC(**best_params, random_state=42, probability=True)
elif 'KNeighborsClassifier' in best_model_name:
    final_model = KNeighborsClassifier(**best_params)
elif 'LogisticRegression' in best_model_name:
    final_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
elif 'MLPClassifier' in best_model_name:
    final_model = MLPClassifier(**best_params, random_state=42, max_iter=300)
elif 'AdaBoostClassifier' in best_model_name:
    final_model = AdaBoostClassifier(**best_params, random_state=42)
elif 'SGDClassifier' in best_model_name:
    final_model = SGDClassifier(**best_params, random_state=42)
else:
    # Should never reach here because of our filtering
    raise ValueError(f"Model {best_model_name} not supported for final model creation")

# Train and evaluate the final model
print("Training final model with optimized parameters...")
final_model.fit(X_train_final, y_train_final)
y_val_pred_final = final_model.predict(X_val)
final_val_accuracy = accuracy_score(y_val, y_val_pred_final)
print(f"Final validation accuracy after hyperparameter optimization: {final_val_accuracy:.4f}")

# Evaluate on test set
print("Evaluating on test set...")
y_test_pred = final_model.predict(X_test_hog)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the final optimized model
joblib.dump(final_model, 'm3_outputs/models/optimized_final_model.joblib')
print(f"Saved optimized model to 'm3_outputs/models/optimized_final_model.joblib'")

# -------------------------------------------------------------------------
# Part 4: Evaluation and Visualization
# -------------------------------------------------------------------------

# Generate and save confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('m3_outputs/hyperopt/confusion_matrix.png')

# Plot optimization history with matplotlib
plt.figure(figsize=(10, 6))
trials = study.trials
values = [t.value for t in trials if t.value is not None]
iterations = list(range(len(values)))
best_values = [max(values[:i+1]) for i in range(len(values))]

plt.plot(iterations, values, 'o-', color='blue', alpha=0.5, label='Trial values')
plt.plot(iterations, best_values, 'o-', color='red', label='Best value')
plt.xlabel('Trial')
plt.ylabel('Accuracy')
plt.title('Optuna Optimization History')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('m3_outputs/hyperopt/optimization_history.png')

# Try to generate parameter importance visualization
if len(study.trials) > 5:
    print("Generating parameter importance visualization...")
    try:
        # Get parameter importance
        param_importance = optuna.importance.get_param_importances(study)
        
        # Convert to DataFrame for easier plotting
        importance_df = pd.DataFrame(
            {'Parameter': list(param_importance.keys()), 
             'Importance': list(param_importance.values())}
        )
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot with matplotlib
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Parameter'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Parameter')
        plt.title('Parameter Importance')
        plt.tight_layout()
        plt.savefig('m3_outputs/hyperopt/parameter_importance.png')
    except Exception as e:
        print(f"Could not generate parameter importance: {str(e)}")

# Generate classification report
report = classification_report(y_test, y_test_pred, target_names=class_names)
with open('m3_outputs/hyperopt/classification_report.txt', 'w') as f:
    f.write(report)

# Save hyperparameter optimization results
optuna_results = {
    'best_model': best_model_name,
    'best_params': best_params,
    'best_accuracy_cv': float(best_score),
    'validation_accuracy': float(final_val_accuracy),
    'test_accuracy': float(test_accuracy),
    'improvement_over_base': float(final_val_accuracy) - float(best_accuracy) if 'best_accuracy' in locals() else None,
    'trial_values': values,
    'best_values': best_values
}

with open('m3_outputs/hyperopt/optuna_results.json', 'w') as f:
    json.dump(optuna_results, f, indent=4)

print("Hyperparameter optimization results saved to 'm3_outputs/hyperopt/optuna_results.json'")

# -------------------------------------------------------------------------
# Part 5: Final Summary and Justification
# -------------------------------------------------------------------------

# Create a summary of model selection and hyperparameter optimization
summary = {
    'model_selection_method': 'AutoML with LazyPredict (filtered to supported models)',
    'best_model': best_model_name,
    'initial_validation_accuracy': float(best_accuracy) if 'best_accuracy' in locals() else None,
    'optimized_validation_accuracy': float(final_val_accuracy),
    'final_test_accuracy': float(test_accuracy),
    'model_justification': f"The {best_model_name} model was selected using LazyPredict AutoML, which evaluated multiple ML models "
                         f"to find the one with the best performance on our validation data. We filtered the results to only include "
                         f"models that our hyperparameter optimization pipeline supports, ensuring a smooth end-to-end process.",
    'hyperparameter_justification': "Hyperparameters were optimized using Optuna, which efficiently searched the parameter space "
                                  "using Bayesian optimization. This process identified the optimal configuration that balances "
                                  "model complexity and generalization performance, resulting in improved accuracy on unseen data."
}

# Save final summary to JSON
with open('m3_outputs/final_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("\nModel Selection and Hyperparameter Optimization complete!")
print("Check the 'm3_outputs' directory for all visualizations and results.")
print(f"\nSelected model: {best_model_name}")
print(f"Final test accuracy: {test_accuracy:.4f}")
