import numpy as np
import pandas as pd
import os
from tensorflow.keras.datasets import fashion_mnist
from ydata_profiling import ProfileReport

# Create output directory
os.makedirs('m1_outputs', exist_ok=True)

# Load Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define class names for better readability
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Create a smaller sample to make profiling more manageable
sample_size = 5000  # Use 5000 samples to keep memory usage reasonable
print(f"Creating a sample of {sample_size} images for profiling...")

# Randomly select indices for sampling
random_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
X_sample = X_train[random_indices]
y_sample = y_train[random_indices]

# Flatten the images
X_sample_flat = X_sample.reshape(X_sample.shape[0], -1)

# Create column names for pixel values
pixel_columns = [f'pixel_{i}' for i in range(X_sample_flat.shape[1])]

# Create DataFrame
df_sample = pd.DataFrame(X_sample_flat, columns=pixel_columns)

# Add class information
df_sample['class'] = [class_names[y] for y in y_sample]
df_sample['class_id'] = y_sample

print(f"Created sample DataFrame with shape: {df_sample.shape}")

# To reduce the size of the report, we can select a subset of pixels
# For example, we can take every 10th pixel
reduced_df = df_sample.iloc[:, ::10].copy()  # Take every 10th pixel column
reduced_df['class'] = df_sample['class']  # Add back the class column
reduced_df['class_id'] = df_sample['class_id']  # Add back the class_id column

print(f"Created reduced DataFrame with shape: {reduced_df.shape} for faster profiling")

# Generate the profile report with minimal configuration for speed
print("Generating profile report (this may take a few minutes)...")
profile = ProfileReport(
    reduced_df,
    title="Fashion MNIST Dataset Profiling Report",
    minimal=True,  # Set to True for faster processing
    progress_bar=True,
    correlations=None,  # Disable correlations for speed
    explorative=True
)

# Save the report
report_path = "m1_outputs/fashion_mnist_profile_report.html"
profile.to_file(report_path)
print(f"Profile report saved to {report_path}")

print("\nTo generate a more detailed report, you can modify the script to:")
print("1. Use more samples (increase sample_size)")
print("2. Include more pixel features (reduce the subsampling in reduced_df)")
print("3. Set minimal=False in ProfileReport for more comprehensive statistics")
print("Note: These changes will increase processing time and memory usage.")
