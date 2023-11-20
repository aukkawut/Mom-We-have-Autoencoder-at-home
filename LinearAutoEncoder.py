#This is autoencoder from PCA and Linear regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression, ElasticNet,ARDRegression, Lasso, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load data from openml
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target

# Split data into train and test
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.1, random_state=42)

# PCA
pca = PCA(n_components=32)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Linear regression on PCA transformed data to get the reconstruction of the intial MNIST image
lr = Lasso(alpha=0.3)
lr.fit(X_train_pca, X_train)

# Randomly pick 10 from test set to show the initial image, PCA transformed image and the reconstructed image
initial_image = X_test.sample(n=10, random_state=42)
initial_image = np.array(initial_image)
pca_image = pca.transform(initial_image)
reconstructed_image = lr.predict(pca_image)

# Plot the images
plt.figure(figsize=(20, 4))
for i in range(10):
    # initial image
    ax = plt.subplot(3, 20, i+1)
    plt.imshow(initial_image[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # PCA transformed image
    ax = plt.subplot(3, 20, i+1+20)
    plt.imshow(pca_image[i].reshape(4, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Reconstructed image
    ax = plt.subplot(3, 20, i+1+40)
    plt.imshow(reconstructed_image[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()


