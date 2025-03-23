import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import cv2
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance  # Use sklearn's implementation instead of ELI5
import lime
from lime import lime_image
import shap
from skimage.feature import hog

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories for outputs
os.makedirs('m2_outputs', exist_ok=True)
os.makedirs('m2_outputs/feature_engineering', exist_ok=True)
os.makedirs('m2_outputs/explainability', exist_ok=True)

print("Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define class names for better readability
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Preview the data
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.tight_layout()
plt.savefig('m2_outputs/feature_engineering/original_images.png')
print("Saved sample of original images")

# -------------------------------------------------------------------------
# Part 1: Feature Engineering Pipeline
# -------------------------------------------------------------------------

def normalize_images(images, method='minmax'):
    """Normalize pixel values using different methods"""
    # Reshape to 2D for sklearn transformers
    shape = images.shape
    images_2d = images.reshape(-1, shape[1] * shape[2])
    
    if method == 'minmax':
        scaler = MinMaxScaler()
        images_normalized = scaler.fit_transform(images_2d)
    elif method == 'standard':
        scaler = StandardScaler()
        images_normalized = scaler.fit_transform(images_2d)
    elif method == 'simple':
        # Simple division by 255
        images_normalized = images_2d / 255.0
    
    # Reshape back to original dimensions
    images_normalized = images_normalized.reshape(shape)
    return images_normalized, scaler

def extract_hog_features(images, pixels_per_cell=(8, 8)):
    """Extract Histogram of Oriented Gradients (HOG) features"""
    hog_features = []
    hog_images = []
    
    for image in images:
        # Extract HOG features
        hog_feature, hog_image = hog(
            image, 
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(1, 1),
            visualize=True,
            feature_vector=True,
            block_norm='L2-Hys'
        )
        hog_features.append(hog_feature)
        hog_images.append(hog_image)
        
    return np.array(hog_features), hog_images[0]  # Return first hog image for visualization

def extract_edge_features(images):
    """Extract edge features using Canny edge detection"""
    edge_features = []
    edge_images = []
    
    for image in images:
        # Apply Canny edge detection
        edges = cv2.Canny(image.astype(np.uint8), 100, 200)
        # Flatten the edge image
        edge_feature = edges.flatten()
        edge_features.append(edge_feature)
        edge_images.append(edges)
        
    return np.array(edge_features), edge_images[0]  # Return first edge image for visualization

def apply_pca(features, n_components=50):
    """Apply PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features, pca

# Create a feature engineering pipeline
print("\nFeature Engineering Pipeline:")

# 1. Normalize the data
print("1. Applying normalization...")
X_train_norm, minmax_scaler = normalize_images(X_train, method='minmax')
X_test_norm, _ = normalize_images(X_test, method='minmax')

# Save normalized samples
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train_norm[i], cmap='gray')
    plt.title(f"Normalized: {class_names[y_train[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('m2_outputs/feature_engineering/normalized_images.png')
print("Saved sample of normalized images")

# 2. Extract HOG features
print("2. Extracting HOG features...")
sample_size = 5000  # Use a subset for faster processing
random_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
X_train_sample = X_train_norm[random_indices]
y_train_sample = y_train[random_indices]

X_train_hog, hog_image = extract_hog_features(X_train_sample)
X_test_hog, _ = extract_hog_features(X_test_norm[:1000])  # Use 1000 test samples

print(f"HOG features shape: {X_train_hog.shape}")

# Save HOG visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(X_train[random_indices[0]], cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(X_train_norm[random_indices[0]], cmap='gray')
plt.title('Normalized Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Features')
plt.axis('off')

plt.tight_layout()
plt.savefig('m2_outputs/feature_engineering/hog_features.png')
print("Saved HOG feature visualization")

# 3. Extract edge features
print("3. Extracting edge features...")
X_train_edge, edge_image = extract_edge_features(X_train_sample)
X_test_edge, _ = extract_edge_features(X_test_norm[:1000])

print(f"Edge features shape: {X_train_edge.shape}")

# Save edge detection visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(X_train[random_indices[0]], cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(X_train_norm[random_indices[0]], cmap='gray')
plt.title('Normalized Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edge_image, cmap='gray')
plt.title('Edge Features')
plt.axis('off')

plt.tight_layout()
plt.savefig('m2_outputs/feature_engineering/edge_features.png')
print("Saved edge feature visualization")

# 4. Apply PCA for dimensionality reduction
print("4. Applying PCA...")
# Flatten the normalized images
X_train_flat = X_train_sample.reshape(X_train_sample.shape[0], -1)
X_test_flat = X_test_norm[:1000].reshape(1000, -1)

X_train_pca, pca_model = apply_pca(X_train_flat, n_components=50)
X_test_pca, _ = apply_pca(X_test_flat, n_components=50)

print(f"PCA features shape: {X_train_pca.shape}")

# Visualize explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of PCA Components')
plt.grid(True)
plt.savefig('m2_outputs/feature_engineering/pca_explained_variance.png')
print("Saved PCA explained variance plot")

# -------------------------------------------------------------------------
# Part 2: Train a Simple Model for Explainability
# -------------------------------------------------------------------------

print("\nTraining a Random Forest model for feature explainability...")

# Convert labels to categorical for neural network training
y_train_cat = to_categorical(y_train_sample, num_classes=10)
y_test_cat = to_categorical(y_test[:1000], num_classes=10)

# Train a Random Forest on HOG features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_hog, y_train_sample)

# Evaluate the model
rf_score = rf_model.score(X_test_hog, y_test[:1000])
print(f"Random Forest accuracy on HOG features: {rf_score:.4f}")

# -------------------------------------------------------------------------
# Part 3: Explainability Analysis
# -------------------------------------------------------------------------

print("\nGenerating explainability visualizations...")

# 1. Feature Importance from Random Forest
feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': [f'HOG_{i}' for i in range(len(feature_importances))],
    'Importance': feature_importances
})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Top 20 HOG Features by Importance')
plt.tight_layout()
plt.savefig('m2_outputs/explainability/hog_feature_importance.png')
print("Saved HOG feature importance visualization")

# 2. Permutation Importance (using sklearn's implementation)
perm_imp = permutation_importance(rf_model, X_test_hog, y_test[:1000], n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({
    'Feature': [f'HOG_{i}' for i in range(len(perm_imp.importances_mean))],
    'Importance': perm_imp.importances_mean
})
perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=perm_importance_df)
plt.title('Top 20 HOG Features by Permutation Importance')
plt.tight_layout()
plt.savefig('m2_outputs/explainability/permutation_importance.png')
print("Saved permutation importance visualization")

# 3. LIME Explainability
try:
    # Prepare LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # For LIME, we'll use a simpler approach with the original images
    # Create a wrapper function that converts images to HOG then predicts
    def predict_fn(images):
        # Convert batch of images to HOG features
        processed_images = []
        for img in images:
            # Make sure image is 2D
            if img.shape == (28, 28, 3):  # If LIME sends RGB images
                img = img[:, :, 0]  # Take first channel
            elif len(img.shape) == 3 and img.shape[2] == 1:  # If single channel but 3D
                img = img[:, :, 0]
                
            # Extract HOG features
            hog_feature, _ = hog(
                img, 
                pixels_per_cell=(8, 8),
                cells_per_block=(1, 1),
                visualize=True,
                feature_vector=True
            )
            processed_images.append(hog_feature)
        processed_images = np.array(processed_images)
        return rf_model.predict_proba(processed_images)
    
    # Apply LIME to a few test images
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(range(3)):  # Explain 3 test images
        # Get the image and true label
        img = X_test_norm[idx]
        label = y_test[idx]
        
        # Generate explanation
        explanation = explainer.explain_instance(
            img.astype('double'), 
            predict_fn,
            top_labels=5, 
            hide_color=0, 
            num_samples=500  # Reduced for speed
        )
        
        # Get the explanation for the true label
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=True, 
            num_features=10, 
            hide_rest=False
        )
        
        # Plot original image
        plt.subplot(3, 2, 2*i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Original: {class_names[label]}')
        plt.axis('off')
        
        # Plot explanation
        plt.subplot(3, 2, 2*i+2)
        plt.imshow(mask, cmap='hot', alpha=0.7)
        plt.title('LIME Explanation')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('m2_outputs/explainability/lime_explanations.png')
    print("Saved LIME explanations")
except Exception as e:
    print(f"Error generating LIME explanations: {e}")
    print("Skipping LIME visualization")

# 4. SHAP Analysis
try:
    # Create a background dataset for SHAP
    background = X_train_hog[np.random.choice(X_train_hog.shape[0], 100, replace=False)]
    
    # Initialize the SHAP explainer
    explainer = shap.KernelExplainer(rf_model.predict_proba, background)
    
    # Calculate SHAP values for a small sample (this can be slow)
    shap_values = explainer.shap_values(X_test_hog[:10], nsamples=100)
    
    # Create summary plot of SHAP values
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_hog[:10], plot_type="bar", class_names=class_names, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('m2_outputs/explainability/shap_feature_importance.png')
    print("Saved SHAP feature importance visualization")
    
    # Instead of the dependence plot (which has shape issues), 
    # let's create a different type of SHAP visualization
    plt.figure(figsize=(12, 8))
    # Create a force plot for a single prediction and convert to matplotlib
    sample_idx = 0
    plt.title(f'SHAP Force Plot for Sample {sample_idx} (Class: {class_names[y_test[sample_idx]]})')
    
    # Use SHAP's decision plot which is more robust to shape issues
    shap.decision_plot(explainer.expected_value[0], shap_values[0][0], 
                      feature_names=[f'HOG_{i}' for i in range(X_test_hog.shape[1])],
                      show=False)
    
    plt.tight_layout()
    plt.savefig('m2_outputs/explainability/shap_decision_plot.png')
    print("Saved SHAP decision plot")
except Exception as e:
    print(f"Error generating SHAP explanations: {e}")
    print("Skipping SHAP visualization")

# -------------------------------------------------------------------------
# Part 4: Refined Feature Engineering Based on Explainability
# -------------------------------------------------------------------------

print("\nRefining feature engineering based on explainability insights...")

# Get top important features from Random Forest
top_features = np.argsort(feature_importances)[::-1][:50]  # Top 50 features

# Select only the most important HOG features
X_train_selected = X_train_hog[:, top_features]
X_test_selected = X_test_hog[:, top_features]

print(f"Selected features shape: {X_train_selected.shape}")

# Train a new model with selected features
rf_model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_selected.fit(X_train_selected, y_train_sample)

# Evaluate the refined model
rf_score_selected = rf_model_selected.score(X_test_selected, y_test[:1000])
print(f"Random Forest accuracy with selected features: {rf_score_selected:.4f}")
print(f"Improvement: {(rf_score_selected - rf_score) * 100:.2f}%")

# Compare feature importance distributions before and after selection
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(feature_importances, bins=50)
plt.title('Original Feature Importance Distribution')
plt.xlabel('Importance')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(rf_model_selected.feature_importances_, bins=50)
plt.title('Selected Feature Importance Distribution')
plt.xlabel('Importance')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('m2_outputs/feature_engineering/feature_selection_comparison.png')
print("Saved feature selection comparison")

# Create a final pipeline that combines the feature engineering steps
final_pipeline = Pipeline([
    ('preprocessing', MinMaxScaler()),
    ('feature_selection', SelectFromModel(rf_model, threshold=-np.inf, max_features=50)),
    ('classification', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Document the final feature engineering process
pipeline_summary = {
    'normalization': 'MinMaxScaler',
    'feature_extraction': 'Histogram of Oriented Gradients (HOG)',
    'feature_selection': 'Top 50 features based on Random Forest importance',
    'dimensionality_reduction': 'None (Feature selection used instead)',
    'original_features': X_train_hog.shape[1],
    'selected_features': X_train_selected.shape[1],
    'original_accuracy': float(rf_score),  # Convert to float for JSON serialization
    'refined_accuracy': float(rf_score_selected),
    'improvement': f"{(rf_score_selected - rf_score) * 100:.2f}%"
}

# Save pipeline summary to JSON
import json
with open('m2_outputs/feature_engineering/pipeline_summary.json', 'w') as f:
    json.dump(pipeline_summary, f, indent=4)

print("\nFeature Engineering and Explainability analysis complete!")
print("Check the 'm2_outputs' directory for all visualizations and results.")
print("\nKey Findings:")
print(f"1. The HOG feature extraction reduced dimensionality from {X_train_flat.shape[1]} to {X_train_hog.shape[1]} features.")
print(f"2. Feature selection further reduced the feature set to {X_train_selected.shape[1]} most important features.")
print(f"3. The model accuracy {'improved' if rf_score_selected > rf_score else 'decreased'} from {rf_score:.4f} to {rf_score_selected:.4f}.")
print("4. Explainability analysis provided insights into which HOG features are most predictive for classifying Fashion MNIST items.")