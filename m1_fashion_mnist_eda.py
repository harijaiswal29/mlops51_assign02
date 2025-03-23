import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
import os

# Create output directory for reports and visualizations
os.makedirs('m1_outputs', exist_ok=True)

# Load Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define class names for better readability
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Create a DataFrame for analysis
# Flatten each 28x28 image into a 784-dimensional vector
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Sample a subset for EDA (using all 60k images can be slow for some tools)
sample_size = 10000
random_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
X_sample = X_train_flat[random_indices]
y_sample = y_train[random_indices]

# Create column names for pixel values
pixel_columns = [f'pixel_{i}' for i in range(X_sample.shape[1])]
df_sample = pd.DataFrame(X_sample, columns=pixel_columns)
df_sample['class'] = [class_names[y] for y in y_sample]
df_sample['class_id'] = y_sample

print(f"Created sample DataFrame with shape: {df_sample.shape}")

# Basic statistics about the dataset
print("\nBasic dataset statistics:")
print(f"Number of training examples: {X_train.shape[0]}")
print(f"Number of test examples: {X_test.shape[0]}")
print(f"Image dimensions: {X_train.shape[1]}x{X_train.shape[2]}")
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")

# Check for missing values
missing_values = df_sample.isnull().sum().sum()
print(f"\nMissing values in the dataset: {missing_values}")

# Class distribution visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=df_sample)
plt.title('Class Distribution in Fashion MNIST Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('m1_outputs/class_distribution.png')
print("Saved class distribution visualization")

# Display sample images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[random_indices[i]], cmap='gray')
    plt.title(class_names[y_train[random_indices[i]]])
    plt.axis('off')
plt.tight_layout()
plt.savefig('m1_outputs/sample_images.png')
print("Saved sample images visualization")

# Pixel intensity distribution
plt.figure(figsize=(10, 6))
plt.hist(X_train_flat.flatten(), bins=50)
plt.title('Pixel Intensity Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('m1_outputs/pixel_distribution.png')
print("Saved pixel intensity distribution")

# Average image per class
plt.figure(figsize=(12, 10))
for i in range(len(class_names)):
    class_indices = np.where(y_train == i)[0]
    avg_image = np.mean(X_train[class_indices], axis=0)
    plt.subplot(3, 4, i+1)
    plt.imshow(avg_image, cmap='gray')
    plt.title(f'Avg: {class_names[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('m1_outputs/avg_images_per_class.png')
print("Saved average images per class")

# Now let's use some automated EDA tools

# Option 1: Pandas Profiling
try:
    from ydata_profiling import ProfileReport
    
    print("\nGenerating Pandas Profiling report...")
    # Use a smaller subset for profiling to avoid memory issues
    profiling_sample = df_sample.sample(2000)
    
    # Generate report
    profile = ProfileReport(
        profiling_sample, 
        title="Fashion MNIST Profiling Report",
        minimal=True,  # Set to True for faster processing
        explorative=True
    )
    
    # Save report
    profile.to_file("m1_outputs/fashion_mnist_profile_report.html")
    print("Saved Pandas Profiling report to m1_outputs/fashion_mnist_profile_report.html")
    
except ImportError:
    print("Pandas Profiling (ydata-profiling) not installed. Install with: pip install ydata-profiling")

# Option 2: Sweetviz
try:
    import sweetviz as sv
    
    print("\nGenerating Sweetviz report...")
    # Use a smaller subset for Sweetviz to avoid memory issues
    sweetviz_sample = df_sample.sample(2000)
    
    # Generate report
    sweet_report = sv.analyze(sweetviz_sample)
    
    # Save report
    sweet_report.show_html("m1_outputs/fashion_mnist_sweetviz_report.html")
    print("Saved Sweetviz report to m1_outputs/fashion_mnist_sweetviz_report.html")
    
except ImportError:
    print("Sweetviz not installed. Install with: pip install sweetviz")

# Option 3: D-Tale
try:
    import dtale
    
    print("\nStarting D-Tale session...")
    # Note: D-Tale will start a web server for interactive exploration
    # For this script, we'll just print instructions
    print("To use D-Tale interactively, run the following in your notebook or interactive session:")
    print("import dtale")
    print("dtale_session = dtale.show(df_sample)")
    print("dtale_session.open_browser()")
    
except ImportError:
    print("D-Tale not installed. Install with: pip install dtale")

print("\nEDA process complete. Check the 'm1_outputs' directory for generated reports and visualizations.")
