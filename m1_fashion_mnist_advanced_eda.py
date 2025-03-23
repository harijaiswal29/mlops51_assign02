import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# Create output directory
os.makedirs('m1_outputs/advanced', exist_ok=True)

# Load Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Flatten the images for dimensionality reduction
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Normalize pixel values to 0-1 range
X_train_normalized = X_train_flat / 255.0
X_test_normalized = X_test_flat / 255.0

# Sample subset for faster processing
sample_size = 5000
random_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
X_sample = X_train_normalized[random_indices]
y_sample = y_train[random_indices]

print(f"Working with sample of {sample_size} images")

# 1. Class Distribution Analysis
plt.figure(figsize=(12, 6))
class_counts = pd.Series(y_train).value_counts().sort_index()
ax = sns.barplot(x=class_counts.index, y=class_counts.values)
plt.xticks(class_counts.index, [class_names[i] for i in class_counts.index], rotation=45)
plt.title('Class Distribution in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')

# Add count labels on top of each bar
for i, count in enumerate(class_counts.values):
    ax.text(i, count + 100, str(count), ha='center')

plt.tight_layout()
plt.savefig('m1_outputs/advanced/class_distribution_detailed.png')
print("Saved detailed class distribution visualization")

# 2. Pixel Intensity Analysis
# Calculate average pixel value for each class
plt.figure(figsize=(12, 8))
for i in range(len(class_names)):
    class_indices = np.where(y_train == i)[0]
    class_pixels = X_train_flat[class_indices].mean(axis=0)
    plt.plot(class_pixels, label=class_names[i], alpha=0.7)
plt.title('Average Pixel Intensity by Position for Each Class')
plt.xlabel('Pixel Position (Flattened)')
plt.ylabel('Average Intensity')
plt.legend()
plt.savefig('m1_outputs/advanced/pixel_intensity_by_class.png')
print("Saved pixel intensity analysis by class")

# 3. Dimensionality Reduction for Visualization

# PCA
print("Performing PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Class')
plt.title('PCA: 2D Visualization of Fashion MNIST')
plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
plt.savefig('m1_outputs/advanced/pca_visualization.png')
print("Saved PCA visualization")

# Add class labels to PCA plot
plt.figure(figsize=(12, 10))
for i in range(10):
    indices = y_sample == i
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=class_names[i], alpha=0.7)
plt.title('PCA: 2D Visualization with Class Labels')
plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
plt.legend()
plt.savefig('m1_outputs/advanced/pca_visualization_labeled.png')
print("Saved labeled PCA visualization")

# t-SNE (using a smaller sample for faster processing)
print("Performing t-SNE (this may take a few minutes)...")
tsne_sample_size = 2000
tsne_indices = np.random.choice(sample_size, tsne_sample_size, replace=False)
X_tsne_sample = X_sample[tsne_indices]
y_tsne_sample = y_sample[tsne_indices]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_tsne_sample)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne_sample, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Class')
plt.title('t-SNE: 2D Visualization of Fashion MNIST')
plt.savefig('m1_outputs/advanced/tsne_visualization.png')
print("Saved t-SNE visualization")

# Add class labels to t-SNE plot
plt.figure(figsize=(12, 10))
for i in range(10):
    indices = y_tsne_sample == i
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=class_names[i], alpha=0.7)
plt.title('t-SNE: 2D Visualization with Class Labels')
plt.legend()
plt.savefig('m1_outputs/advanced/tsne_visualization_labeled.png')
print("Saved labeled t-SNE visualization")

# 4. Feature correlation analysis
# Compute correlation matrix for a sample of pixels
print("Computing pixel correlation matrix...")
# Sample 100 random pixels to keep the correlation matrix manageable
pixel_sample_size = 100
pixel_indices = np.random.choice(X_train_flat.shape[1], pixel_sample_size, replace=False)
pixel_sample = X_train_flat[:1000, pixel_indices]  # Use 1000 images

# Create a DataFrame with the sampled pixels
pixel_df = pd.DataFrame(pixel_sample, columns=[f'pixel_{i}' for i in pixel_indices])

# Compute correlation matrix
corr_matrix = pixel_df.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Random Pixel Sample')
plt.tight_layout()
plt.savefig('m1_outputs/advanced/pixel_correlation.png')
print("Saved pixel correlation matrix")

# 5. Image variability within classes
plt.figure(figsize=(15, 10))
for i in range(10):
    class_indices = np.where(y_train == i)[0]
    std_image = np.std(X_train[class_indices], axis=0)
    
    plt.subplot(2, 5, i+1)
    plt.imshow(std_image, cmap='viridis')
    plt.title(f'Std Dev: {class_names[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('m1_outputs/advanced/class_variability.png')
print("Saved class variability visualization")

# 6. Save dataset statistics summary
stats = {
    'dataset_name': 'Fashion MNIST',
    'train_size': X_train.shape[0],
    'test_size': X_test.shape[0],
    'image_dimensions': f"{X_train.shape[1]}x{X_train.shape[2]}",
    'num_features': X_train_flat.shape[1],
    'num_classes': len(class_names),
    'class_names': class_names,
    'class_distribution': class_counts.to_dict(),
    'pixel_value_range': f"{X_train.min()}-{X_train.max()}",
    'mean_pixel_value': X_train_flat.mean(),
    'std_pixel_value': X_train_flat.std(),
    'missing_values': 'None',
    'pca_explained_variance': {
        'PC1': pca.explained_variance_ratio_[0],
        'PC2': pca.explained_variance_ratio_[1],
        'cumulative': sum(pca.explained_variance_ratio_[:2])
    }
}

# Save as JSON
import json
with open('m1_outputs/advanced/dataset_statistics.json', 'w') as f:
    json.dump(stats, f, indent=4)

print("Saved dataset statistics summary")
print("\nAdvanced EDA complete. Check the 'm1_outputs/advanced' directory for visualizations.")
