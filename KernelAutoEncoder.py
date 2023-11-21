import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load EMNIST data
emnist = fetch_openml(name="Fashion-MNIST", parser='auto', cache = True)
X, y = emnist.data / 255.0, emnist.target.astype(int)

# Flatten images
X_flat = X.values

# Split data into training and testing sets
X_train, X_test, _, _ = train_test_split(X_flat, y, test_size=0.5, random_state=42)

# Apply Kernel PCA for non-linear dimensionality reduction
kpca = KernelPCA(n_components=16, kernel='rbf', gamma=0.01)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)

# Train Kernelized Regression Model for reconstruction
krr = KernelRidge(alpha=1e-6, kernel='rbf', gamma=0.01)
krr.fit(X_train_kpca, X_train)

# Reconstruct images
reconstructed_images = krr.predict(X_test_kpca)

# Calculate reconstruction error
mse = mean_squared_error(X_test, reconstructed_images)
print(f"Mean Squared Error: {mse}")

# Visualize original and reconstructed images
n_samples = 5
plt.figure(figsize=(10, 4))
for i in range(n_samples):
    plt.subplot(2, n_samples, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, n_samples, i + 1 + n_samples)
    plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()
