import pymc as pm
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache = True, parser='auto')
X, y = mnist.data, mnist.target.astype('int32')

# Standardize and perform PCA
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
pca = PCA(n_components=16)
X_pca = pca.fit_transform(X_std)

# Generate new data function
def generate_new_data(coefficients, intercept, n_samples=5):
    Z_new = np.random.randn(n_samples, 16)  # New latent variables
    X_new = np.matmul(Z_new, coefficients.values) + intercept.values  # Linear regression to generate new data
    X_new_unscaled = scaler.inverse_transform(X_new)  # Unscale the data
    return X_new_unscaled
def plot_images(images, rows, cols, title):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.ravel()

    for i in range(rows * cols):
        if i < len(images):
            axes[i].imshow(images[i].reshape(28, 28))
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    plt.suptitle(title)
    plt.savefig(title + '.png')
coef = []
inte = []
for i in tqdm(range(10)):
    # Select a class for variational inference
    class_label = i
    X_pca_class = X_pca[y == class_label]
    X_class = X_std[y == class_label]
    with pm.Model() as model:
        # Priors
        coefficients = pm.Normal('coefficients', mu=0, sigma=1, shape=(16, 784))
        intercept = pm.Normal('intercept', mu=0, sigma=1, shape=784)
        # Regression model
        mu = pm.math.dot(X_pca_class, coefficients) + intercept
        # Likelihood of observations fixing the class
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=1, observed=X_class)
        # VI
        approx = pm.fit(n=10000, method='advi', obj_optimizer = pm.adadelta())
    # Draw samples from posterior
    trace = approx.sample(draws=5000)
    # Coefficient is just a expectation on that posterior
    posterior_coefficients = np.mean(trace['posterior']['coefficients'][0], axis = 0)
    coef.append(posterior_coefficients)
    posterior_intercept = np.mean(trace['posterior']['intercept'][0], axis = 0)
    inte.append(posterior_intercept)
    images = generate_new_data(posterior_coefficients, posterior_intercept, n_samples=10)
    plot_images(images, rows=2, cols=5, title=f'Generated Images for label {i}')
import pickle
with open('coef.pkl', 'wb') as f:
    pickle.dump(coef, f)
with open('inte.pkl', 'wb') as f:
    pickle.dump(inte, f)
    
def thresholding(data_array,threshold = 1e-2):
  # Apply one-sided thresholding
  thresholded_data = np.where(data_array < threshold, 0, 1.2*data_array)
  thresholded_data_rescaled = (thresholded_data - np.min(thresholded_data)) / (np.max(thresholded_data) - np.min(thresholded_data))

  return thresholded_data_rescaled
def interpolate_and_generate_ppca(coefficients, intercept, start_point, end_point, n_samples=10, k=9):
    interpolated_points = interpolate_points(start_point, end_point, n_samples)
    generated_images = []
    for point in interpolated_points:
        Z_new = point.reshape(1, k)
        X_new = np.matmul(Z_new, coefficients.values) + intercept.values
        X_new_unscaled = thresholding(scaler.inverse_transform(X_new), 4e-3)  # Assuming 'scaler' is your StandardScaler instance
        generated_images.append(X_new_unscaled.squeeze())
    return generated_images

def interpolate_points(p1, p2, n_steps=10):
    return np.linspace(p1, p2, num=n_steps)


coefficients_0 = coef[0]  # Coefficients for digit 0
intercept_0 = inte[0]    # Intercept for digit 0
# Select two points in the latent space for digit 0
start_point = np.random.randn(k)  # Random point in latent space
end_point = np.random.randn(k)    # Another random point
generated_images = interpolate_and_generate_ppca(coefficients_0, intercept_0, start_point, end_point, n_samples=10, k=k)

# Visualize the results
plot_images(generated_images, rows=1, cols=len(generated_images), title="Latent Space Interpolation of LDA and MMLR for Label 6")
