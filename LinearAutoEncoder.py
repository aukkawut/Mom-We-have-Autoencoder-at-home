import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load MNIST data
mnist = fetch_openml('mnist_784', parser='auto', cache = True)
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Flatten images
X_flat = X.values

# Split data into training and testing sets
X_train, X_test, _, _ = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=16)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train multivariate linear regression model
regression_model = LinearRegression()
regression_model.fit(X_train_pca, X_train)

# Reconstruct images
reconstructed_images = regression_model.predict(X_test_pca)

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
# Select a baseline image index
baseline_index = 0

# Get the latent vector for the baseline image
baseline_latent_vector = X_train_pca[baseline_index]

# Select two random dimensions to perturb
perturb_dimensions = np.random.choice(range(baseline_latent_vector.shape[0]), size=2, replace=False)

# Generate images by perturbing the selected dimensions
n_steps = 10
perturbation_values = np.linspace(-3, 3, n_steps)
generated_images = np.zeros((n_steps, n_steps, X_train.shape[1]))

for i, perturb_val1 in enumerate(perturbation_values):
    for j, perturb_val2 in enumerate(perturbation_values):
        perturbed_latent_vector = baseline_latent_vector.copy()
        perturbed_latent_vector[perturb_dimensions[0]] += perturb_val1
        perturbed_latent_vector[perturb_dimensions[1]] += perturb_val2
        generated_images[i, j] = regression_model.predict(perturbed_latent_vector.reshape(1, -1))

# Plot the results
plt.figure(figsize=(12, 12))

# Baseline Image
plt.subplot(n_steps, n_steps, 1)
plt.imshow(X_train[baseline_index].reshape(28, 28), cmap='gray')
plt.title("Baseline Image")
plt.axis('off')

# Generated Images by Perturbing Dimensions
for i in range(n_steps):
    for j in range(n_steps):
        plt.subplot(n_steps, n_steps, i * n_steps + j + 1)
        plt.imshow(generated_images[i, j].reshape(28, 28), cmap='gray')
        plt.title(f"Perturb {perturb_dimensions[0]}: {perturbation_values[i]:.2f}\n"
                  f"Perturb {perturb_dimensions[1]}: {perturbation_values[j]:.2f}")
        plt.axis('off')

plt.show()
