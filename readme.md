# Fashion MNIST MLOps Pipeline

This repository implements a comprehensive MLOps pipeline for the Fashion MNIST dataset, demonstrating best practices for continuous improvement and model explainability.

## Project Overview

The pipeline consists of four major components:

1. **Exploratory Data Analysis (M1)**: Automated analysis and visualization of the dataset structure
2. **Feature Engineering & Explainability (M2)**: Preprocessing with HOG features and model interpretability
3. **Model Selection & Hyperparameter Optimization (M3)**: Automated model evaluation and parameter tuning
4. **Model Monitoring & Performance Tracking (M4)**: Drift detection and performance tracking over time

## Repository Structure

```
fashion-mnist-mlops/
├── m1_fashion_mnist_eda.py               # Basic EDA visualizations
├── m1_fashion_mnist_advanced_eda.py      # Advanced EDA with dimensionality reduction
├── m1_fashion_mnist_profiling.py         # Automated EDA with pandas profiling
├── m2_fashion_mnist_feature_engineering.py # Feature engineering pipeline
├── m3_fashion_mnist_automl_simplified.py  # Model selection and optimization
├── m4_fashion_mnist_model_monitoring.py   # Model monitoring and drift detection
├── requirements.txt                       # Project dependencies
└── README.md                              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-mnist-mlops.git
   cd fashion-mnist-mlops
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

### Milestone 1: Exploratory Data Analysis

```bash
# Run basic EDA
python m1_fashion_mnist_eda.py

# Run advanced EDA with dimensionality reduction
python m1_fashion_mnist_advanced_eda.py

# Generate automated profiling report
python m1_fashion_mnist_profiling.py
```

This will generate:
- Basic visualizations in `m1_outputs/`
- Advanced visualizations in `m1_outputs/advanced/`
- Automated profiling report in `m1_outputs/fashion_mnist_profile_report.html`

### Milestone 2: Feature Engineering & Explainability

```bash
python m2_fashion_mnist_feature_engineering.py
```

This will:
1. Apply normalization to images
2. Extract HOG features
3. Train a preliminary model for explainability
4. Generate feature importance visualizations
5. Create LIME and SHAP explanations

Output will be saved to `m2_outputs/` directory.

### Milestone 3: Model Selection & Hyperparameter Optimization

```bash
python m3_fashion_mnist_automl_simplified.py
```

This will:
1. Evaluate multiple models using LazyPredict
2. Select the best model (typically SVC)
3. Optimize hyperparameters using Optuna
4. Generate performance metrics and visualizations

Output will be saved to `m3_outputs/` directory, including:
- Trained models in `m3_outputs/models/`
- AutoML results in `m3_outputs/automl/`
- Hyperparameter optimization results in `m3_outputs/hyperopt/`

### Milestone 4: Model Monitoring & Performance Tracking

```bash
# Start MLflow server in a separate terminal
mlflow ui --port 5000

# Run monitoring script
python m4_fashion_mnist_model_monitoring.py
```

This will:
1. Load the optimized model from M3
2. Simulate model monitoring over time
3. Detect feature, label, and performance drift
4. Generate drift visualizations
5. Log metrics to MLflow

Output will be saved to `m4_outputs/` directory and can be viewed in the MLflow UI at http://localhost:5000.

## Expected Outputs

### Milestone 1 (EDA)
- Class distribution
- Sample images
- Pixel intensity distributions
- Average images per class
- PCA and t-SNE visualizations
- Automated profiling reports

### Milestone 2 (Feature Engineering)
- Normalized images
- HOG feature visualizations
- Edge detection visualizations
- Feature importance plots
- LIME explanations
- SHAP visualizations

### Milestone 3 (Model Selection)
- Model comparison results
- Optimization history
- Parameter importance
- Confusion matrix
- Classification report
- Trained models (best_automl_model.joblib and optimized_final_model.joblib)

### Milestone 4 (Monitoring)
- Accuracy over time plot
- Drift metrics visualization
- Retraining signals
- Drift detection reports
- MLflow experiment tracking

## Troubleshooting

### Common Issues

1. **Memory Errors**: 
   - Reduce the sample sizes in the scripts for datasets or features
   - Use a machine with more RAM

2. **Library Compatibility**:
   - If you encounter issues with SHAP or LIME, try installing them separately:
     ```bash
     pip install shap==0.40.0
     pip install lime==0.2.0
     ```

3. **LazyPredict Installation**:
   - If LazyPredict fails to install automatically, install it manually:
     ```bash
     pip install lazypredict
     ```

4. **MLflow Connection Issues**:
   - Ensure you've started the MLflow server before running the monitoring script
   - Check if port 5000 is already in use; if so, use a different port

## License

[Include your license information here]

## Acknowledgments

- Fashion MNIST dataset by Zalando Research
- The various open-source libraries that made this pipeline possible
