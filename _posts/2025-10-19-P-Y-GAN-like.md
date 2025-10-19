---
layout: post
title: "GAN-like Synthetic Data Generation Examples with DistroSimulator"
description: "Examples of synthetic data generation using DistroSimulator for univariate, multivariate distributions, digits, Fashion-MNIST, and Olivetti faces."
date: 2025-10-19
categories: [R, Python]
comments: true
---

In [Generating Synthetic Data with R-vine Copulas using esgtoolkit in R](https://thierrymoudiki.github.io/blog/2025/09/21/r/synthetic-copulas), I presented a method to generate synthetic stock returns data using R-vine copulas with the `esgtoolkit` package in R. 

This post demonstrates how to use the `DistroSimulator` class from [Techtonique's `synthe` package](https://github.com/Techtonique/synthe/tree/main) to generate synthetic data. The examples cover **univariate normal distributions, multivariate distributions, digits dataset, Fashion-MNIST, and Olivetti faces**.

The results showcase the effectiveness of the `DistroSimulator` in capturing the underlying distribution of the data through diverse  metrics and visualizations.

But first, please sign the petition "Stop torturing T. Moudiki": [https://www.change.org/stop_torturing_T_Moudiki](https://www.change.org/stop_torturing_T_Moudiki)

```python
!pip install git+https://github.com/Techtonique/synthe.git
```


```python
import numpy as np
import matplotlib.pyplot as plt
import optuna

from scipy import stats
from sklearn.datasets import fetch_olivetti_faces, load_digits
from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm
from synthe import DistroSimulator  # Assuming your package name is synthe


# Now let's reproduce the exact same examples using DistroSimulator

print("=" * 60)
print("EXAMPLE 1: Univariate Normal Distribution")
print("=" * 60)

# Example 1: Univariate normal
# Example usage (univariate)
np.random.seed(42)
n = 200
Y_uni = np.random.normal(0, 1, n)

# Create and fit the simulator
simulator_uni = DistroSimulator(
    random_state=42,
    residual_sampling="bootstrap",
    use_rff=False  # Small dataset, no need for approximation
)

simulator_uni.fit(Y_uni, metric='wasserstein', n_trials=50)
Y_sim_uni = simulator_uni.sample(500)

print("Univariate Results:")
print(f"Best sigma: {simulator_uni.best_params_['sigma']:.3f}, "
      f"lambda: {simulator_uni.best_params_['lambd']:.3f}, "
      f"dist: {simulator_uni.best_score_:.3f}")

# Compare distributions
simulator_uni.compare_distributions(Y_uni, Y_sim_uni)

print("\n" + "=" * 60)
print("EXAMPLE 2: Bivariate Normal Distribution")
print("=" * 60)

# Example 2: Bivariate normal
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
Y_multi = np.random.multivariate_normal(mean, cov, n)

simulator_multi = DistroSimulator(
    random_state=42,
    residual_sampling="bootstrap",
    use_rff=False
)

simulator_multi.fit(Y_multi, metric='mmd', n_trials=50)
Y_sim_multi = simulator_multi.sample(500)

print("Bivariate Results:")
print(f"Best sigma: {simulator_multi.best_params_['sigma']:.3f}, "
      f"lambda: {simulator_multi.best_params_['lambd']:.3f}, "
      f"MMD: {simulator_multi.best_score_:.3f}")

# Compare distributions
simulator_multi.compare_distributions(Y_multi, Y_sim_multi)

print("\n" + "=" * 60)
print("EXAMPLE 3: 3D Multivariate Mixture")
print("=" * 60)

# Example 3: 3D multivariate mixture
# Example usage (3D multivariate mixture)
np.random.seed(42)
n_samples = 800

# Generate multivariate target distribution
cov_matrix1 = np.array([[1.0, 0.5, 0.3],
                        [0.5, 1.5, 0.4],
                        [0.3, 0.4, 0.8]])
component1 = np.random.multivariate_normal([0, 1, -0.5], cov_matrix1, int(0.6 * n_samples))

cov_matrix2 = np.array([[0.8, -0.3, 0.1],
                        [-0.3, 1.2, -0.2],
                        [0.1, -0.2, 1.0]])
component2 = np.random.multivariate_normal([2, -1, 1], cov_matrix2, int(0.4 * n_samples))

Y_mixture = np.vstack([component1, component2])
np.random.shuffle(Y_mixture)

print("Target Distribution: 3D mixture of multivariate normals")
print(f"Sample size: {len(Y_mixture)}")
print("Original correlation matrix:")
print(np.corrcoef(Y_mixture.T))

simulator_mixture = DistroSimulator(
    random_state=42,
    residual_sampling="bootstrap",
    use_rff=False
)

simulator_mixture.fit(Y_mixture, metric='mmd', n_trials=50)
Y_sim_mixture = simulator_mixture.sample(800)

print("\n3D Mixture Results:")
print(f"Best sigma: {simulator_mixture.best_params_['sigma']:.3f}, "
      f"lambda: {simulator_mixture.best_params_['lambd']:.3f}, "
      f"MMD: {simulator_mixture.best_score_:.3f}")

# Compare distributions
simulator_mixture.compare_distributions(Y_mixture, Y_sim_mixture)

print("\n" + "=" * 60)
print("EXAMPLE 4: Testing Different Residual Sampling Methods")
print("=" * 60)

# Test different sampling methods on univariate data
sampling_methods = ["bootstrap", "kde", "gmm"]
sampling_results = {}

for method in sampling_methods:
    print(f"\nTesting {method.upper()} sampling...")
    simulator_test = DistroSimulator(
        random_state=42,
        residual_sampling=method,
        use_rff=False
    )

    simulator_test.fit(Y_uni, metric='wasserstein', n_trials=30)
    Y_sim_test = simulator_test.sample(500)

    # Calculate KS test for comparison
    ks_stat, ks_pvalue = stats.ks_2samp(Y_uni.flatten(), Y_sim_test.flatten())
    sampling_results[method] = {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'best_score': simulator_test.best_score_
    }

    print(f"  KS statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")
    print(f"  Best distance: {simulator_test.best_score_:.4f}")

print("\nSampling Method Comparison:")
for method, results in sampling_results.items():
    print(f"{method.upper()}: KS={results['ks_statistic']:.4f}, p={results['ks_pvalue']:.4f}, dist={results['best_score']:.4f}")


# Continue with the DistroSimulator class definition from previous code...

print("\n" + "=" * 60)
print("EXAMPLE 5: Digits Dataset (MNIST)")
print("=" * 60)

# Load digits dataset
digits = load_digits()
pca_digits = PCA(n_components=15, whiten=False)
Y_digits = pca_digits.fit_transform(digits.data)
print(f"Target Distribution: PCA-transformed digits dataset (15D)")
print(f"Sample size: {len(Y_digits)}")

# Create and fit the simulator for digits
simulator_digits = DistroSimulator(
    random_state=42,
    residual_sampling="bootstrap",
    use_rff=True,  # Enable RFF for larger dataset
)

simulator_digits.fit(Y_digits, metric='mmd', n_trials=50)
Y_sim_digits = simulator_digits.sample(len(Y_digits))

print("\nDigits Dataset Results:")
print(f"Best sigma: {simulator_digits.best_params_['sigma']:.3f}, "
      f"lambda: {simulator_digits.best_params_['lambd']:.3f}, "
      f"MMD: {simulator_digits.best_score_:.3f}")

def visualize_digits(real_data, sim_data, pca, n_samples=44):
    """Visualize original and simulated digits"""
    # Inverse transform simulated data to original 64D space
    sim_data_orig = pca.inverse_transform(sim_data)

    # Select first n_samples from real and simulated data
    real_data = real_data[:n_samples].reshape((4, 11, -1))
    sim_data_orig = sim_data_orig[:n_samples].reshape((4, 11, -1))

    # Plot 4x11 grid
    fig, ax = plt.subplots(9, 11, figsize=(11, 9), subplot_kw=dict(xticks=[], yticks=[]))

    for j in range(11):
        ax[4, j].set_visible(False)
        for i in range(4):
            im = ax[i, j].imshow(
                real_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest"
            )
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(
                sim_data_orig[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest"
            )
            im.set_clim(0, 16)

    ax[0, 5].set_title("Selection from the input data", fontsize=12)
    ax[5, 5].set_title("Simulated digits from residual-resampling model", fontsize=12)
    plt.tight_layout()
    plt.savefig('digits_comparison_distrosimulator.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize digits comparison
visualize_digits(digits.data, Y_sim_digits, pca_digits)

print("\n" + "=" * 60)
print("EXAMPLE 6: Olivetti Faces")
print("=" * 60)

# Load Olivetti faces
olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
Y_olivetti = olivetti.data
pca_olivetti = PCA(n_components=15, whiten=False)
Y_olivetti_pca = pca_olivetti.fit_transform(Y_olivetti)
print(f"Target Distribution: PCA-transformed Olivetti faces (15D)")
print(f"Sample size: {len(Y_olivetti_pca)}")

# Create and fit the simulator for Olivetti
simulator_olivetti = DistroSimulator(
    random_state=42,
    residual_sampling="bootstrap",
    use_rff=True,
)

simulator_olivetti.fit(Y_olivetti_pca, metric='mmd', n_trials=50)
Y_sim_olivetti = simulator_olivetti.sample(len(Y_olivetti_pca))

print("\nOlivetti Faces Results:")
print(f"Best sigma: {simulator_olivetti.best_params_['sigma']:.3f}, "
      f"lambda: {simulator_olivetti.best_params_['lambd']:.3f}, "
      f"MMD: {simulator_olivetti.best_score_:.3f}")

def visualize_images(real_data, sim_data, pca, img_shape, dataset_name, n_samples=44):
    """Visualize original and simulated images"""
    sim_data_orig = pca.inverse_transform(sim_data)
    real_data = real_data[:n_samples].reshape((4, 11, -1))
    sim_data_orig = sim_data_orig[:n_samples].reshape((4, 11, -1))

    fig, ax = plt.subplots(9, 11, figsize=(11, 9), subplot_kw=dict(xticks=[], yticks=[]))

    for j in range(11):
        ax[4, j].set_visible(False)
        for i in range(4):
            im = ax[i, j].imshow(
                real_data[i, j].reshape(img_shape), cmap=plt.cm.binary, interpolation="nearest"
            )
            im.set_clim(real_data.min(), real_data.max())
            im = ax[i + 5, j].imshow(
                sim_data_orig[i, j].reshape(img_shape), cmap=plt.cm.binary, interpolation="nearest"
            )
            im.set_clim(real_data.min(), real_data.max())

    ax[0, 5].set_title(f"Selection from {dataset_name} data", fontsize=12)
    ax[5, 5].set_title(f"Simulated {dataset_name} from DistroSimulator", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_comparison_distrosimulator.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize Olivetti comparison
visualize_images(Y_olivetti, Y_sim_olivetti, pca_olivetti, img_shape=(64, 64), dataset_name="Olivetti")

print("\n" + "=" * 60)
print("EXAMPLE 7: Fashion-MNIST")
print("=" * 60)

# Load Fashion-MNIST (subsample for efficiency)
(X_train_fmnist, _), (_, _) = fashion_mnist.load_data()
Y_fmnist = X_train_fmnist.reshape(-1, 784) / 255.0  # Normalize to [0, 1]
n_subsample = 5000
idx = np.random.choice(len(Y_fmnist), n_subsample, replace=False)
Y_fmnist = Y_fmnist[idx]

pca_fmnist = PCA(n_components=15, whiten=False)
Y_fmnist_pca = pca_fmnist.fit_transform(Y_fmnist)
print(f"Target Distribution: PCA-transformed Fashion-MNIST (15D, subsampled)")
print(f"Sample size: {len(Y_fmnist_pca)}")

# Create and fit the simulator for Fashion-MNIST
simulator_fmnist = DistroSimulator(
    random_state=42,
    residual_sampling="bootstrap",
    use_rff=True,
)

simulator_fmnist.fit(Y_fmnist_pca, metric='mmd', n_trials=50)
Y_sim_fmnist = simulator_fmnist.sample(len(Y_fmnist_pca))

print("\nFashion-MNIST Results:")
print(f"Best sigma: {simulator_fmnist.best_params_['sigma']:.3f}, "
      f"lambda: {simulator_fmnist.best_params_['lambd']:.3f}, "
      f"MMD: {simulator_fmnist.best_score_:.3f}")

# Visualize Fashion-MNIST comparison
visualize_images(Y_fmnist, Y_sim_fmnist, pca_fmnist, img_shape=(28, 28), dataset_name="Fashion-MNIST")

print("\n" + "=" * 60)
print("EXAMPLE 8: Advanced Sampling Methods Comparison")
print("=" * 60)

# Compare different sampling methods on Fashion-MNIST
sampling_methods_fmnist = ["bootstrap", "kde", "gmm"]
fmnist_results = {}

print("Comparing sampling methods on Fashion-MNIST...")
for method in sampling_methods_fmnist:
    print(f"\nTesting {method.upper()} sampling...")

    simulator_compare = DistroSimulator(
        random_state=42,
        residual_sampling=method,
        use_rff=True,
    )

    # Use smaller subset for faster comparison
    Y_fmnist_subset = Y_fmnist_pca[:1000]
    simulator_compare.fit(Y_fmnist_subset, metric='mmd', n_trials=20)
    Y_sim_compare = simulator_compare.sample(1000)

    # Calculate distribution similarity metrics
    energy_dist = simulator_compare._custom_energy_distance(Y_fmnist_subset, Y_sim_compare)
    mmd_dist = simulator_compare._mmd(Y_fmnist_subset, Y_sim_compare)

    fmnist_results[method] = {
        'best_score': simulator_compare.best_score_,
        'energy_distance': energy_dist,
        'mmd': mmd_dist,
        'sigma': simulator_compare.best_params_['sigma'],
        'lambda': simulator_compare.best_params_['lambd']
    }

    print(f"  Best MMD: {simulator_compare.best_score_:.4f}")
    print(f"  Energy distance: {energy_dist:.4f}")
    print(f"  Current MMD: {mmd_dist:.4f}")

print("\nFashion-MNIST Sampling Method Comparison:")
print("-" * 50)
for method, results in fmnist_results.items():
    print(f"{method.upper():<12} | MMD: {results['best_score']:.4f} | "
          f"Energy: {results['energy_distance']:.4f} | "
          f"σ: {results['sigma']:.3f} | λ: {results['lambda']:.3f}")

print("\n" + "=" * 60)
print("EXAMPLE 9: Statistical Tests on Image Datasets")
print("=" * 60)

def perform_statistical_tests(Y_orig, Y_sim, dataset_name):
    """Perform comprehensive statistical tests on the results"""
    print(f"\nStatistical Tests for {dataset_name}:")
    print("-" * 40)

    # Energy distance
    energy_dist = simulator_digits._custom_energy_distance(Y_orig, Y_sim)
    print(f"Energy Distance: {energy_dist:.6f}")

    # MMD
    mmd_dist = simulator_digits._mmd(Y_orig, Y_sim)
    print(f"MMD: {mmd_dist:.6f}")

    # Marginal KS tests
    n_dims = Y_orig.shape[1]
    ks_pvalues = []
    for i in range(min(5, n_dims)):  # Test first 5 dimensions
        ks_stat, ks_pvalue = stats.ks_2samp(Y_orig[:, i], Y_sim[:, i])
        ks_pvalues.append(ks_pvalue)

    print(f"KS test p-values (first 5 dims): {[f'{p:.4f}' for p in ks_pvalues]}")
    print(f"Min KS p-value: {min(ks_pvalues):.4f}")

    # Correlation preservation
    orig_corr = np.corrcoef(Y_orig.T)
    sim_corr = np.corrcoef(Y_sim.T)
    corr_diff = np.mean(np.abs(orig_corr - sim_corr))
    print(f"Average correlation difference: {corr_diff:.4f}")

    return {
        'energy_distance': energy_dist,
        'mmd': mmd_dist,
        'min_ks_pvalue': min(ks_pvalues),
        'avg_corr_diff': corr_diff
    }

# Perform statistical tests on all datasets
datasets = {
    'Digits': (Y_digits, Y_sim_digits),
    'Olivetti': (Y_olivetti_pca, Y_sim_olivetti),
    'Fashion-MNIST': (Y_fmnist_pca, Y_sim_fmnist)
}

statistical_results = {}
for name, (Y_orig, Y_sim) in datasets.items():
    statistical_results[name] = perform_statistical_tests(Y_orig, Y_sim, name)

print("\n" + "=" * 60)
print("SUMMARY: All Dataset Results")
print("=" * 60)

print("\nPerformance Summary:")
print("-" * 80)
print(f"{'Dataset':<15} {'MMD':<10} {'Energy Dist':<12} {'Min KS p-val':<12} {'Avg Corr Diff':<12}")
print("-" * 80)

for name in statistical_results.keys():
    results = statistical_results[name]
    print(f"{name:<15} {results['mmd']:<10.6f} {results['energy_distance']:<12.6f} "
          f"{results['min_ks_pvalue']:<12.4f} {results['avg_corr_diff']:<12.4f}")

print("\n" + "=" * 60)
print("EXAMPLE 10: Quality Assessment with Different PCA Dimensions")
print("=" * 60)

# Test with different PCA dimensions on Fashion-MNIST
pca_dims = [5, 10, 15, 20]
pca_results = {}

print("Testing different PCA dimensions on Fashion-MNIST...")
for n_components in pca_dims:
    print(f"\nPCA with {n_components} components:")

    # Apply PCA
    pca_test = PCA(n_components=n_components, whiten=False)
    Y_fmnist_test = pca_test.fit_transform(Y_fmnist[:2000])  # Use subset for speed

    # Fit simulator
    simulator_pca = DistroSimulator(
        random_state=42,
        residual_sampling="bootstrap",
        use_rff=True,
    )

    simulator_pca.fit(Y_fmnist_test, metric='mmd', n_trials=20)
    Y_sim_pca = simulator_pca.sample(len(Y_fmnist_test))

    # Calculate metrics
    energy_dist = simulator_pca._custom_energy_distance(Y_fmnist_test, Y_sim_pca)
    mmd_dist = simulator_pca._mmd(Y_fmnist_test, Y_sim_pca)

    pca_results[n_components] = {
        'mmd': mmd_dist,
        'energy': energy_dist,
        'best_score': simulator_pca.best_score_,
        'explained_variance': np.sum(pca_test.explained_variance_ratio_)
    }

    print(f"  Explained variance: {pca_results[n_components]['explained_variance']:.3f}")
    print(f"  Best MMD: {simulator_pca.best_score_:.4f}")
    print(f"  Current MMD: {mmd_dist:.4f}")

print("\nPCA Dimension Comparison:")
print("-" * 50)
print(f"{'Components':<12} {'Explained Var':<14} {'Best MMD':<10} {'Current MMD':<12} {'Energy Dist':<12}")
print("-" * 50)
for n_comp, results in pca_results.items():
    print(f"{n_comp:<12} {results['explained_variance']:<14.3f} "
          f"{results['best_score']:<10.4f} {results['mmd']:<12.4f} {results['energy']:<12.4f}")

print("\n" + "=" * 60)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 60)






import pandas as pd

log_returns = pd.read_csv("https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/multivariate/log_returns.csv")
log_returns.drop(columns=["Unnamed: 0"], inplace=True)
log_returns.index = pd.date_range(start="2024-04-24", periods=len(log_returns), freq="B")
display(log_returns.head())
display(log_returns.tail())

simulator_multi = DistroSimulator(
    random_state=42,
    n_clusters=1,
    residual_sampling="bootstrap",
    clustering_method="kmeans",
    use_rff=True
)

# Pass the values (NumPy array) of the DataFrame to the fit method
simulator_multi.fit(log_returns.values, metric='mmd', n_trials=100)
Y_sim_multi = simulator_multi.sample(500)

print("Bivariate Results:")
print(f"Best sigma: {simulator_multi.best_params_['sigma']:.3f}, "
      f"lambda: {simulator_multi.best_params_['lambd']:.3f}, "
      f"MMD: {simulator_multi.best_score_:.3f}")

# Compare distributions
simulator_multi.compare_distributions(log_returns.values, Y_sim_multi)
```

    [I 2025-10-17 17:10:36,729] A new study created in memory with name: no-name-efe52968-3328-4d29-9f90-53e6cfef1dac
    [I 2025-10-17 17:10:36,741] Trial 0 finished with value: 0.13865139056120598 and parameters: {'sigma': 7.939529961485126, 'lambd': 0.00025564139075974225}. Best is trial 0 with value: 0.13865139056120598.
    [I 2025-10-17 17:10:36,754] Trial 1 finished with value: 0.12509044170632658 and parameters: {'sigma': 0.016326612419707866, 'lambd': 0.10952025363677034}. Best is trial 1 with value: 0.12509044170632658.
    [I 2025-10-17 17:10:36,765] Trial 2 finished with value: 0.11061164802462088 and parameters: {'sigma': 0.19798283142271447, 'lambd': 0.0011258694916476417}. Best is trial 2 with value: 0.11061164802462088.
    [I 2025-10-17 17:10:36,778] Trial 3 finished with value: 0.1303733014116722 and parameters: {'sigma': 0.31407455296972825, 'lambd': 4.6783733209865167e-05}. Best is trial 2 with value: 0.11061164802462088.
    [I 2025-10-17 17:10:36,786] Trial 4 finished with value: 0.10519036748633664 and parameters: {'sigma': 0.15294561056611258, 'lambd': 2.4914088337259825e-05}. Best is trial 4 with value: 0.10519036748633664.
    [I 2025-10-17 17:10:36,794] Trial 5 finished with value: 0.09667308702726289 and parameters: {'sigma': 1.3081913048401494, 'lambd': 0.41987706182176376}. Best is trial 5 with value: 0.09667308702726289.
    [I 2025-10-17 17:10:36,802] Trial 6 finished with value: 0.09351359591898555 and parameters: {'sigma': 1.2159760687292298, 'lambd': 0.010594581126969739}. Best is trial 6 with value: 0.09351359591898555.
    [I 2025-10-17 17:10:36,810] Trial 7 finished with value: 0.09877115961508015 and parameters: {'sigma': 0.04323853207868255, 'lambd': 0.0006176112901161659}. Best is trial 6 with value: 0.09351359591898555.
    [I 2025-10-17 17:10:36,818] Trial 8 finished with value: 0.13433598955243806 and parameters: {'sigma': 0.7818889178221696, 'lambd': 0.06829451533968077}. Best is trial 6 with value: 0.09351359591898555.
    [I 2025-10-17 17:10:36,826] Trial 9 finished with value: 0.12295779498265826 and parameters: {'sigma': 0.29474225951883504, 'lambd': 0.0025226560420775787}. Best is trial 6 with value: 0.09351359591898555.
    [I 2025-10-17 17:10:36,852] Trial 10 finished with value: 0.08831227645565284 and parameters: {'sigma': 8.072659644947068, 'lambd': 0.01692234323423254}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:36,866] Trial 11 finished with value: 0.12399196132507113 and parameters: {'sigma': 8.775162887575267, 'lambd': 0.017767342506139133}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:36,881] Trial 12 finished with value: 0.17818770991640567 and parameters: {'sigma': 2.527737204578802, 'lambd': 0.011641679359383473}. Best is trial 10 with value: 0.08831227645565284.


    ============================================================
    EXAMPLE 1: Univariate Normal Distribution
    ============================================================


    [I 2025-10-17 17:10:36,898] Trial 13 finished with value: 0.1278497464603523 and parameters: {'sigma': 2.717042401012473, 'lambd': 0.010335722904489154}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:36,913] Trial 14 finished with value: 0.15680270713277725 and parameters: {'sigma': 3.137505302734161, 'lambd': 0.8555898896591118}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:36,927] Trial 15 finished with value: 0.16245174417575825 and parameters: {'sigma': 0.7442012774099896, 'lambd': 0.044675924439867315}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:36,941] Trial 16 finished with value: 0.14322015475342245 and parameters: {'sigma': 4.698999529531017, 'lambd': 0.004964623961467108}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:36,956] Trial 17 finished with value: 0.12845078252055192 and parameters: {'sigma': 1.2880707463963539, 'lambd': 0.17594776427257966}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:36,975] Trial 18 finished with value: 0.13304115164019475 and parameters: {'sigma': 0.06484836062638796, 'lambd': 0.0001721243805060685}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:37,006] Trial 19 finished with value: 0.1376648176841346 and parameters: {'sigma': 0.6512141529898293, 'lambd': 0.028422846053845405}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:37,034] Trial 20 finished with value: 0.26553812848983 and parameters: {'sigma': 5.335786405065405, 'lambd': 0.002976720235114457}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:37,048] Trial 21 finished with value: 0.1379374770558697 and parameters: {'sigma': 1.3533690602502408, 'lambd': 0.5886737221572172}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:37,069] Trial 22 finished with value: 0.27616105484722264 and parameters: {'sigma': 1.7054854650386475, 'lambd': 0.17577601309842297}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:37,100] Trial 23 finished with value: 0.1423762460625137 and parameters: {'sigma': 0.6336686133486901, 'lambd': 0.3139003914271717}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:37,151] Trial 24 finished with value: 0.21270922380830065 and parameters: {'sigma': 4.2961436965412, 'lambd': 0.008000391630157054}. Best is trial 10 with value: 0.08831227645565284.
    [I 2025-10-17 17:10:37,181] Trial 25 finished with value: 0.08773323157186091 and parameters: {'sigma': 1.8104612179407673, 'lambd': 0.03139386210623193}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,196] Trial 26 finished with value: 0.11326691381546121 and parameters: {'sigma': 2.211652991334354, 'lambd': 0.031043362187921742}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,214] Trial 27 finished with value: 0.17529467587665049 and parameters: {'sigma': 8.506835194616686, 'lambd': 0.0020254013143509593}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,236] Trial 28 finished with value: 0.10931573878689667 and parameters: {'sigma': 0.47663533895800214, 'lambd': 0.07273288977190787}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,258] Trial 29 finished with value: 0.13173542934293836 and parameters: {'sigma': 9.946548381791501, 'lambd': 0.00534649988918561}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,278] Trial 30 finished with value: 0.13466615353116962 and parameters: {'sigma': 5.390013361438354, 'lambd': 0.0004352873479542061}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,295] Trial 31 finished with value: 0.12739845765625216 and parameters: {'sigma': 1.1456962259848296, 'lambd': 0.47253514796617924}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,312] Trial 32 finished with value: 0.13523530561318498 and parameters: {'sigma': 1.6879829636620967, 'lambd': 0.021875474652266837}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,329] Trial 33 finished with value: 0.1543861983885528 and parameters: {'sigma': 3.5918928264067462, 'lambd': 0.12371426509412914}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,346] Trial 34 finished with value: 0.124087423999485 and parameters: {'sigma': 0.436327260616358, 'lambd': 0.26956045816759056}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,365] Trial 35 finished with value: 0.11419262249207385 and parameters: {'sigma': 0.12484226407061048, 'lambd': 0.05404856495829252}. Best is trial 25 with value: 0.08773323157186091.
    [I 2025-10-17 17:10:37,386] Trial 36 finished with value: 0.07989423450503166 and parameters: {'sigma': 1.0373669923347824, 'lambd': 0.005860634936872617}. Best is trial 36 with value: 0.07989423450503166.
    [I 2025-10-17 17:10:37,407] Trial 37 finished with value: 0.12309923159666655 and parameters: {'sigma': 0.016312322069009774, 'lambd': 0.001432166432744615}. Best is trial 36 with value: 0.07989423450503166.
    [I 2025-10-17 17:10:37,424] Trial 38 finished with value: 0.10592598518285125 and parameters: {'sigma': 0.22018842698237467, 'lambd': 0.004747961704996959}. Best is trial 36 with value: 0.07989423450503166.
    [I 2025-10-17 17:10:37,442] Trial 39 finished with value: 0.18387560583024581 and parameters: {'sigma': 0.8945922279320206, 'lambd': 1.013740576689216e-05}. Best is trial 36 with value: 0.07989423450503166.
    [I 2025-10-17 17:10:37,474] Trial 40 finished with value: 0.0841440659778046 and parameters: {'sigma': 0.39255975213677985, 'lambd': 0.0008365913245914038}. Best is trial 36 with value: 0.07989423450503166.
    [I 2025-10-17 17:10:37,503] Trial 41 finished with value: 0.1384268688656396 and parameters: {'sigma': 0.46620383062453347, 'lambd': 0.0011463911050532305}. Best is trial 36 with value: 0.07989423450503166.
    [I 2025-10-17 17:10:37,530] Trial 42 finished with value: 0.17560429712757222 and parameters: {'sigma': 0.2707802864803906, 'lambd': 9.678803461845194e-05}. Best is trial 36 with value: 0.07989423450503166.
    [I 2025-10-17 17:10:37,548] Trial 43 finished with value: 0.07859017899393293 and parameters: {'sigma': 0.1379048338144211, 'lambd': 0.014620117962116035}. Best is trial 43 with value: 0.07859017899393293.
    [I 2025-10-17 17:10:37,570] Trial 44 finished with value: 0.14443117280787746 and parameters: {'sigma': 0.10618826118096122, 'lambd': 0.0006493692751477353}. Best is trial 43 with value: 0.07859017899393293.
    [I 2025-10-17 17:10:37,588] Trial 45 finished with value: 0.11948151918682495 and parameters: {'sigma': 0.02658031202524431, 'lambd': 0.012720206673101468}. Best is trial 43 with value: 0.07859017899393293.
    [I 2025-10-17 17:10:37,608] Trial 46 finished with value: 0.1504836103202269 and parameters: {'sigma': 0.07667874163490111, 'lambd': 0.007952656111460798}. Best is trial 43 with value: 0.07859017899393293.
    [I 2025-10-17 17:10:37,625] Trial 47 finished with value: 0.15057885623741438 and parameters: {'sigma': 0.19661515571692617, 'lambd': 0.02108466739207806}. Best is trial 43 with value: 0.07859017899393293.
    [I 2025-10-17 17:10:37,646] Trial 48 finished with value: 0.106031846594626 and parameters: {'sigma': 0.36582581781031204, 'lambd': 0.0006843719509585771}. Best is trial 43 with value: 0.07859017899393293.
    [I 2025-10-17 17:10:37,663] Trial 49 finished with value: 0.15447925883266217 and parameters: {'sigma': 2.027974253634049, 'lambd': 0.0002741800125024495}. Best is trial 43 with value: 0.07859017899393293.
    /usr/local/lib/python3.12/dist-packages/synthe/distro_simulator.py:840: UserWarning: p-value capped: true value larger than 0.25. Consider specifying `method` (e.g. `method=stats.PermutationMethod()`.)
      ad_result = stats.anderson_ksamp([Y_orig[:, i], Y_sim[:, i]])


      Using standard kernel method
    Univariate Results:
    Best sigma: 0.138, lambda: 0.015, dist: 0.079



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_4.png){:class="img-responsive"}
    


    
    ============================================================
    COMPREHENSIVE STATISTICAL TEST RESULTS
    ============================================================
    
    Dimension 1:
      Kolmogorov-Smirnov Test:
        Statistic: 0.085000
        p-value: 0.466286
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: -0.231175
        Significance level: 0.250
        Interpretation: Distributions similar



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_6.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_7.png){:class="img-responsive"}
    


    [I 2025-10-17 17:10:39,785] A new study created in memory with name: no-name-10680fd7-63aa-40f0-bc75-7b35ff67937f
    [I 2025-10-17 17:10:39,795] Trial 0 finished with value: 0.00757912609023359 and parameters: {'sigma': 0.45159380533833304, 'lambd': 0.015285458488091496}. Best is trial 0 with value: 0.00757912609023359.
    [I 2025-10-17 17:10:39,804] Trial 1 finished with value: 0.0059033877506062815 and parameters: {'sigma': 0.12790641295619623, 'lambd': 0.0004048349417403313}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:39,813] Trial 2 finished with value: 0.0059681315590341955 and parameters: {'sigma': 0.15660249417198127, 'lambd': 3.821498240714095e-05}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:39,831] Trial 3 finished with value: 0.019895296434778387 and parameters: {'sigma': 0.4539775047736735, 'lambd': 0.2195055073038003}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:39,853] Trial 4 finished with value: 0.009334331736636337 and parameters: {'sigma': 0.2084698360391535, 'lambd': 0.021156584230896905}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:39,873] Trial 5 finished with value: 0.006078835083101253 and parameters: {'sigma': 0.07582284188991656, 'lambd': 0.00352790339986226}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:39,889] Trial 6 finished with value: 0.007823377016179744 and parameters: {'sigma': 0.09788729151873958, 'lambd': 0.037845505055796216}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:39,905] Trial 7 finished with value: 0.005905455655558112 and parameters: {'sigma': 0.11161285143637067, 'lambd': 0.0009087693109192781}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:39,914] Trial 8 finished with value: 0.03108532342364745 and parameters: {'sigma': 0.16926811143879536, 'lambd': 0.5215610123378658}. Best is trial 1 with value: 0.0059033877506062815.


    
    ============================================================
    EXAMPLE 2: Bivariate Normal Distribution
    ============================================================


    [I 2025-10-17 17:10:39,948] Trial 9 finished with value: 0.014651863548524768 and parameters: {'sigma': 2.9317020954021613, 'lambd': 0.00021401291624394104}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,002] Trial 10 finished with value: 0.005972010919200521 and parameters: {'sigma': 0.011390996824054764, 'lambd': 1.4186864699397386e-05}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,043] Trial 11 finished with value: 0.005978898510451969 and parameters: {'sigma': 0.025226449193894822, 'lambd': 0.00038349171532250323}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,078] Trial 12 finished with value: 0.012282238125339018 and parameters: {'sigma': 1.0918349668661071, 'lambd': 0.00034852584464863524}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,132] Trial 13 finished with value: 0.005982311706606014 and parameters: {'sigma': 0.0368393574840666, 'lambd': 0.0017638872045388284}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,155] Trial 14 finished with value: 0.012738519376381374 and parameters: {'sigma': 6.338899865357751, 'lambd': 0.0020615333755626543}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,173] Trial 15 finished with value: 0.005971251329131011 and parameters: {'sigma': 0.05073355003082267, 'lambd': 8.112402084964673e-05}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,197] Trial 16 finished with value: 0.01986741617468679 and parameters: {'sigma': 1.0294183706643432, 'lambd': 0.0008616034400650235}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,215] Trial 17 finished with value: 0.005940423212321133 and parameters: {'sigma': 0.017894636618105097, 'lambd': 0.006382033372172324}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,232] Trial 18 finished with value: 0.013846329689912285 and parameters: {'sigma': 0.7847306673543072, 'lambd': 9.391867894409297e-05}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,248] Trial 19 finished with value: 0.005984791904521614 and parameters: {'sigma': 0.07498170902050637, 'lambd': 0.0008964814539169048}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,268] Trial 20 finished with value: 0.00590493798581071 and parameters: {'sigma': 2.303180498591104, 'lambd': 1.2692262285785782e-05}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,289] Trial 21 finished with value: 0.013284733040663244 and parameters: {'sigma': 2.547375209843448, 'lambd': 1.4361814513361906e-05}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,307] Trial 22 finished with value: 0.006192467239163224 and parameters: {'sigma': 0.2849470595265868, 'lambd': 3.993264000267063e-05}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,331] Trial 23 finished with value: 0.00646916050945523 and parameters: {'sigma': 2.1092080221678526, 'lambd': 0.00013694959408941666}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,348] Trial 24 finished with value: 0.016612418722344535 and parameters: {'sigma': 5.368605745160852, 'lambd': 0.07926997981131215}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,375] Trial 25 finished with value: 0.006080770866824792 and parameters: {'sigma': 0.11121754671622948, 'lambd': 0.005843634506968388}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,399] Trial 26 finished with value: 0.008931797593683455 and parameters: {'sigma': 0.5097612584555968, 'lambd': 0.0005546335560550102}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,426] Trial 27 finished with value: 0.0059721512004148325 and parameters: {'sigma': 0.04917088818781612, 'lambd': 2.678805141519147e-05}. Best is trial 1 with value: 0.0059033877506062815.
    [I 2025-10-17 17:10:40,460] Trial 28 finished with value: 0.005263531112211672 and parameters: {'sigma': 9.830200296841944, 'lambd': 1.0460311142198797e-05}. Best is trial 28 with value: 0.005263531112211672.
    [I 2025-10-17 17:10:40,486] Trial 29 finished with value: 0.007010917012259199 and parameters: {'sigma': 4.033077781417283, 'lambd': 1.212958213094797e-05}. Best is trial 28 with value: 0.005263531112211672.
    [I 2025-10-17 17:10:40,509] Trial 30 finished with value: 0.00666128384170428 and parameters: {'sigma': 8.341705342911675, 'lambd': 5.973974522910358e-05}. Best is trial 28 with value: 0.005263531112211672.
    [I 2025-10-17 17:10:40,531] Trial 31 finished with value: 0.00818908721297329 and parameters: {'sigma': 9.715800951907788, 'lambd': 2.803787526331052e-05}. Best is trial 28 with value: 0.005263531112211672.
    [I 2025-10-17 17:10:40,556] Trial 32 finished with value: 0.00405157901540576 and parameters: {'sigma': 1.5469422602408454, 'lambd': 0.00018314584836766346}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,573] Trial 33 finished with value: 0.005993794092372662 and parameters: {'sigma': 1.7348207795912711, 'lambd': 0.00017926463377286288}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,598] Trial 34 finished with value: 0.020376976136829428 and parameters: {'sigma': 1.5560232693096554, 'lambd': 2.7527416134195413e-05}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,619] Trial 35 finished with value: 0.0054895924055294865 and parameters: {'sigma': 3.5522060828011224, 'lambd': 7.590833231219877e-05}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,643] Trial 36 finished with value: 0.010598500165818092 and parameters: {'sigma': 3.4112371058073636, 'lambd': 0.00023766404115756784}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,661] Trial 37 finished with value: 0.007392157990843029 and parameters: {'sigma': 4.8586312553589694, 'lambd': 5.895004118979393e-05}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,687] Trial 38 finished with value: 0.005789366125415141 and parameters: {'sigma': 0.2814669231042328, 'lambd': 4.400105129973596e-05}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,705] Trial 39 finished with value: 0.012906009342960045 and parameters: {'sigma': 0.5440717502398419, 'lambd': 3.884538531848992e-05}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,722] Trial 40 finished with value: 0.007536426186959266 and parameters: {'sigma': 7.602461372035633, 'lambd': 0.0001271466709381745}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,743] Trial 41 finished with value: 0.005936508295067311 and parameters: {'sigma': 0.2443773894402853, 'lambd': 5.8855140656128954e-05}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,762] Trial 42 finished with value: 0.005937198137503041 and parameters: {'sigma': 0.18273195400186693, 'lambd': 2.1614468808108084e-05}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,780] Trial 43 finished with value: 0.006066602937505827 and parameters: {'sigma': 0.15030276803006165, 'lambd': 0.00026135901348521383}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,798] Trial 44 finished with value: 0.010078373248892158 and parameters: {'sigma': 0.6281725628829741, 'lambd': 0.0004978931386984565}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,818] Trial 45 finished with value: 0.0061335698164645125 and parameters: {'sigma': 0.3572496236200006, 'lambd': 0.00011132690566631603}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,835] Trial 46 finished with value: 0.010811604839440436 and parameters: {'sigma': 0.3519755597235105, 'lambd': 0.0015599793121436572}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,852] Trial 47 finished with value: 0.010744682406860329 and parameters: {'sigma': 1.3683232459045307, 'lambd': 0.9318246793967383}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,874] Trial 48 finished with value: 0.012614978950124422 and parameters: {'sigma': 3.7355753354296075, 'lambd': 5.2850162852263505e-05}. Best is trial 32 with value: 0.00405157901540576.
    [I 2025-10-17 17:10:40,894] Trial 49 finished with value: 0.007101821024179 and parameters: {'sigma': 0.911147202871469, 'lambd': 1.944994649386531e-05}. Best is trial 32 with value: 0.00405157901540576.


      Using standard kernel method
    Bivariate Results:
    Best sigma: 1.547, lambda: 0.000, MMD: 0.004



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_12.png){:class="img-responsive"}
    


    
    ============================================================
    COMPREHENSIVE STATISTICAL TEST RESULTS
    ============================================================
    
    Dimension 1:
      Kolmogorov-Smirnov Test:
        Statistic: 0.135000
        p-value: 0.052139
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: 0.967242
        Significance level: 0.131
        Interpretation: Distributions similar
    
    Dimension 2:
      Kolmogorov-Smirnov Test:
        Statistic: 0.115000
        p-value: 0.142075
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: 1.716362
        Significance level: 0.064
        Interpretation: Distributions similar



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_14.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_15.png){:class="img-responsive"}
    


    [I 2025-10-17 17:10:42,624] A new study created in memory with name: no-name-5b2b3fcf-175e-48a3-9322-489de8a57193
    [I 2025-10-17 17:10:42,668] Trial 0 finished with value: 0.002322687734283846 and parameters: {'sigma': 0.16553926571933472, 'lambd': 0.004991740104386651}. Best is trial 0 with value: 0.002322687734283846.
    [I 2025-10-17 17:10:42,775] Trial 1 finished with value: 0.0023946794765971335 and parameters: {'sigma': 0.012643730233777423, 'lambd': 8.911299157347148e-05}. Best is trial 0 with value: 0.002322687734283846.


    
    ============================================================
    EXAMPLE 3: 3D Multivariate Mixture
    ============================================================
    Target Distribution: 3D mixture of multivariate normals
    Sample size: 800
    Original correlation matrix:
    [[ 1.         -0.43924198  0.57623006]
     [-0.43924198  1.         -0.33956797]
     [ 0.57623006 -0.33956797  1.        ]]


    [I 2025-10-17 17:10:42,975] Trial 2 finished with value: 0.03428236149102548 and parameters: {'sigma': 0.06916930987044892, 'lambd': 0.47429778061506705}. Best is trial 0 with value: 0.002322687734283846.
    [I 2025-10-17 17:10:43,078] Trial 3 finished with value: 0.002315504743073793 and parameters: {'sigma': 0.09484851007747586, 'lambd': 0.008642303735943588}. Best is trial 3 with value: 0.002315504743073793.
    [I 2025-10-17 17:10:43,168] Trial 4 finished with value: 0.007531830389713973 and parameters: {'sigma': 2.6004293571016053, 'lambd': 0.007058778679729241}. Best is trial 3 with value: 0.002315504743073793.
    [I 2025-10-17 17:10:43,298] Trial 5 finished with value: 0.004672777165316999 and parameters: {'sigma': 9.091841031092143, 'lambd': 0.0031342957672057946}. Best is trial 3 with value: 0.002315504743073793.
    [I 2025-10-17 17:10:43,390] Trial 6 finished with value: 0.004402539950321804 and parameters: {'sigma': 0.15266972392493955, 'lambd': 0.0911534996451324}. Best is trial 3 with value: 0.002315504743073793.
    [I 2025-10-17 17:10:43,488] Trial 7 finished with value: 0.002296862036165137 and parameters: {'sigma': 0.3039963885287504, 'lambd': 0.0008347858479977647}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:43,581] Trial 8 finished with value: 0.01010787875621838 and parameters: {'sigma': 0.8234828574603542, 'lambd': 0.0033817821608881477}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:43,773] Trial 9 finished with value: 0.0025190831444356077 and parameters: {'sigma': 0.49601533744733156, 'lambd': 1.9645713944030023e-05}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:43,853] Trial 10 finished with value: 0.0023921407517389914 and parameters: {'sigma': 0.02188134511517753, 'lambd': 0.00026650085454558943}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:43,943] Trial 11 finished with value: 0.0023887182652336802 and parameters: {'sigma': 0.05578309007583408, 'lambd': 0.0004854168964132118}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:44,000] Trial 12 finished with value: 0.016425948557974357 and parameters: {'sigma': 0.9107453578601367, 'lambd': 0.08763754049672833}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:44,057] Trial 13 finished with value: 0.0027746914908431952 and parameters: {'sigma': 0.20720810125813763, 'lambd': 0.030693671202997774}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:44,148] Trial 14 finished with value: 0.0023900597358126607 and parameters: {'sigma': 0.05266483670168091, 'lambd': 0.0006249228920858126}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:44,214] Trial 15 finished with value: 0.00895641286226373 and parameters: {'sigma': 1.7759425853939916, 'lambd': 0.016387526247155528}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:44,267] Trial 16 finished with value: 0.0023905725403942646 and parameters: {'sigma': 0.3509681793535357, 'lambd': 0.001350311650664697}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:44,333] Trial 17 finished with value: 0.002394464905209309 and parameters: {'sigma': 0.10815721085259188, 'lambd': 8.086837012160703e-05}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:44,391] Trial 18 finished with value: 0.04069665159895297 and parameters: {'sigma': 0.017250748510764396, 'lambd': 0.8851512816326714}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:44,494] Trial 19 finished with value: 0.002395567079509525 and parameters: {'sigma': 0.028997839096446053, 'lambd': 1.1638018685172846e-05}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:44,577] Trial 20 finished with value: 0.010111222005180498 and parameters: {'sigma': 0.38593675532040145, 'lambd': 0.0387229196515953}. Best is trial 7 with value: 0.002296862036165137.
    [I 2025-10-17 17:10:44,678] Trial 21 finished with value: 0.0021541318928401942 and parameters: {'sigma': 0.19762623024925685, 'lambd': 0.007929816182119864}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:44,787] Trial 22 finished with value: 0.002378061423397304 and parameters: {'sigma': 0.0923780023235256, 'lambd': 0.0014748972670946183}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:44,884] Trial 23 finished with value: 0.0024238001246560947 and parameters: {'sigma': 0.23261746644467263, 'lambd': 0.012795959534371971}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:45,167] Trial 24 finished with value: 0.0023794237221544667 and parameters: {'sigma': 0.042975479596360014, 'lambd': 0.0015704994024101707}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:45,275] Trial 25 finished with value: 0.02017891290883156 and parameters: {'sigma': 0.67040565221192, 'lambd': 0.16299274982749287}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:45,335] Trial 26 finished with value: 0.016678392124417712 and parameters: {'sigma': 1.2780132866715221, 'lambd': 0.00015657209367814595}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:45,403] Trial 27 finished with value: 0.010600164468300366 and parameters: {'sigma': 3.56649190273539, 'lambd': 0.01134919961042482}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:45,473] Trial 28 finished with value: 0.0023862955791874585 and parameters: {'sigma': 0.10857999584264684, 'lambd': 0.0007134387854167455}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:45,570] Trial 29 finished with value: 0.0022612649295274956 and parameters: {'sigma': 0.21024736111290368, 'lambd': 0.0039453342062811736}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:45,713] Trial 30 finished with value: 0.0022429111066801233 and parameters: {'sigma': 0.22907683017109987, 'lambd': 0.003102083029909277}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:45,816] Trial 31 finished with value: 0.0021719459773751337 and parameters: {'sigma': 0.25679546697079264, 'lambd': 0.003654438948376939}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:45,942] Trial 32 finished with value: 0.002311676035517296 and parameters: {'sigma': 0.1799856753167514, 'lambd': 0.003536379169112489}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:46,174] Trial 33 finished with value: 0.01586367428450669 and parameters: {'sigma': 0.5189115754346222, 'lambd': 0.024506294597311785}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:46,500] Trial 34 finished with value: 0.0022823057189398366 and parameters: {'sigma': 0.15106047175615514, 'lambd': 0.0064560917829212675}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:46,821] Trial 35 finished with value: 0.002207439625224683 and parameters: {'sigma': 0.2690957863407665, 'lambd': 0.0020582317709508667}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:46,969] Trial 36 finished with value: 0.0023656518707455 and parameters: {'sigma': 0.27958225592286023, 'lambd': 0.0018147599554297201}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:47,094] Trial 37 finished with value: 0.012055678104367196 and parameters: {'sigma': 0.4972048377087801, 'lambd': 0.006337420027963177}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:47,278] Trial 38 finished with value: 0.0023668711082727167 and parameters: {'sigma': 0.0774340797758398, 'lambd': 0.002364113441903739}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:47,405] Trial 39 finished with value: 0.002391048476465374 and parameters: {'sigma': 0.13351772565481423, 'lambd': 0.0002459427092328238}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:47,529] Trial 40 finished with value: 0.012000504553038305 and parameters: {'sigma': 1.127068459413218, 'lambd': 0.06143073665103903}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:47,769] Trial 41 finished with value: 0.002171391196809036 and parameters: {'sigma': 0.27043830682403536, 'lambd': 0.004400975348743673}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:47,886] Trial 42 finished with value: 0.0024139646574208307 and parameters: {'sigma': 0.3127641492997159, 'lambd': 0.004706344537476109}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:48,012] Trial 43 finished with value: 0.005614565541952721 and parameters: {'sigma': 0.39694888243280346, 'lambd': 0.009149700235344078}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:48,125] Trial 44 finished with value: 0.009745641709428488 and parameters: {'sigma': 0.608386250374016, 'lambd': 0.002584190023936653}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:48,236] Trial 45 finished with value: 0.002342787177038952 and parameters: {'sigma': 0.25801610436011324, 'lambd': 0.0010419308418912311}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:48,324] Trial 46 finished with value: 0.0023821192222296117 and parameters: {'sigma': 0.18553819450020023, 'lambd': 0.00043855094598624204}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:48,422] Trial 47 finished with value: 0.0023344955892924957 and parameters: {'sigma': 0.13677907726737057, 'lambd': 0.006507007221160386}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:48,544] Trial 48 finished with value: 0.0022756570759907002 and parameters: {'sigma': 0.06709836279337231, 'lambd': 0.01828455117024619}. Best is trial 21 with value: 0.0021541318928401942.
    [I 2025-10-17 17:10:48,630] Trial 49 finished with value: 0.011383993727818953 and parameters: {'sigma': 0.7736542207608815, 'lambd': 0.00103240120427897}. Best is trial 21 with value: 0.0021541318928401942.


      Using standard kernel method
    
    3D Mixture Results:
    Best sigma: 0.198, lambda: 0.008, MMD: 0.002


    /usr/local/lib/python3.12/dist-packages/synthe/distro_simulator.py:840: UserWarning: p-value floored: true value smaller than 0.001. Consider specifying `method` (e.g. `method=stats.PermutationMethod()`.)
      ad_result = stats.anderson_ksamp([Y_orig[:, i], Y_sim[:, i]])



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_21.png){:class="img-responsive"}
    


    
    ============================================================
    COMPREHENSIVE STATISTICAL TEST RESULTS
    ============================================================
    
    Dimension 1:
      Kolmogorov-Smirnov Test:
        Statistic: 0.152500
        p-value: 0.000000
        Significance: SIGNIFICANT
      Anderson-Darling Test:
        Statistic: 18.737273
        Significance level: 0.001
        Interpretation: Distributions differ
    
    Dimension 2:
      Kolmogorov-Smirnov Test:
        Statistic: 0.088750
        p-value: 0.003652
        Significance: SIGNIFICANT
      Anderson-Darling Test:
        Statistic: 10.100227
        Significance level: 0.001
        Interpretation: Distributions differ
    
    Dimension 3:
      Kolmogorov-Smirnov Test:
        Statistic: 0.103750
        p-value: 0.000360
        Significance: SIGNIFICANT
      Anderson-Darling Test:
        Statistic: 13.258901
        Significance level: 0.001
        Interpretation: Distributions differ



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_23.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_24.png){:class="img-responsive"}
    


    [I 2025-10-17 17:10:51,551] A new study created in memory with name: no-name-5ddfb6ed-edf9-45ed-8cd7-5658c8a522bb
    [I 2025-10-17 17:10:51,560] Trial 0 finished with value: 0.126090061994121 and parameters: {'sigma': 0.2643067570756047, 'lambd': 0.018681351433384473}. Best is trial 0 with value: 0.126090061994121.
    [I 2025-10-17 17:10:51,569] Trial 1 finished with value: 0.10712502789702817 and parameters: {'sigma': 3.501479257993283, 'lambd': 0.04839053846593362}. Best is trial 1 with value: 0.10712502789702817.
    [I 2025-10-17 17:10:51,576] Trial 2 finished with value: 0.11946731152209171 and parameters: {'sigma': 0.7505384099474822, 'lambd': 0.00045853257340076154}. Best is trial 1 with value: 0.10712502789702817.
    [I 2025-10-17 17:10:51,584] Trial 3 finished with value: 0.16591846773442662 and parameters: {'sigma': 0.02695576864818756, 'lambd': 0.8550465412648137}. Best is trial 1 with value: 0.10712502789702817.
    [I 2025-10-17 17:10:51,593] Trial 4 finished with value: 0.20850412805744512 and parameters: {'sigma': 7.289134593325246, 'lambd': 1.7316492571233098e-05}. Best is trial 1 with value: 0.10712502789702817.
    [I 2025-10-17 17:10:51,601] Trial 5 finished with value: 0.11293804338917358 and parameters: {'sigma': 0.07384305773658001, 'lambd': 2.8565747721623684e-05}. Best is trial 1 with value: 0.10712502789702817.
    [I 2025-10-17 17:10:51,611] Trial 6 finished with value: 0.08655468214633288 and parameters: {'sigma': 0.17105047550211236, 'lambd': 0.23018094864773014}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,623] Trial 7 finished with value: 0.11584616852137554 and parameters: {'sigma': 0.09982331960790443, 'lambd': 0.005062264263452878}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,635] Trial 8 finished with value: 0.12160725498433832 and parameters: {'sigma': 0.8830408060312631, 'lambd': 0.00241427928594918}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,645] Trial 9 finished with value: 0.15553142512785495 and parameters: {'sigma': 0.06241990840840007, 'lambd': 0.0030959894718095835}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,662] Trial 10 finished with value: 0.15168833166037637 and parameters: {'sigma': 0.016275969780852027, 'lambd': 0.915025474741849}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,680] Trial 11 finished with value: 0.1239641361130665 and parameters: {'sigma': 4.321476418174071, 'lambd': 0.08993403898193073}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,699] Trial 12 finished with value: 0.17527401096984818 and parameters: {'sigma': 2.014707166904962, 'lambd': 0.09732224811375152}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,717] Trial 13 finished with value: 0.15465280900054099 and parameters: {'sigma': 0.24549872596169775, 'lambd': 0.10706178721011705}. Best is trial 6 with value: 0.08655468214633288.


    
    ============================================================
    EXAMPLE 4: Testing Different Residual Sampling Methods
    ============================================================
    
    Testing BOOTSTRAP sampling...


    [I 2025-10-17 17:10:51,738] Trial 14 finished with value: 0.1831847103127754 and parameters: {'sigma': 1.137209792503289, 'lambd': 0.026380668509087234}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,755] Trial 15 finished with value: 0.16250863469927654 and parameters: {'sigma': 2.949104595930344, 'lambd': 0.23705851826736943}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,771] Trial 16 finished with value: 0.14320336053562907 and parameters: {'sigma': 9.61264510321515, 'lambd': 0.018702015797929416}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,787] Trial 17 finished with value: 0.11849372500700509 and parameters: {'sigma': 0.5727287941566578, 'lambd': 0.0004336809056763512}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,807] Trial 18 finished with value: 0.09103771134558808 and parameters: {'sigma': 0.14078625480721782, 'lambd': 0.35759592791332584}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,825] Trial 19 finished with value: 0.15529309047299367 and parameters: {'sigma': 0.15687598766759772, 'lambd': 0.32098342049411455}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,843] Trial 20 finished with value: 0.21249698896648145 and parameters: {'sigma': 0.04352645155211563, 'lambd': 0.2986655259013703}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,860] Trial 21 finished with value: 0.12446920809848944 and parameters: {'sigma': 0.40896873575816406, 'lambd': 0.04025128172659398}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,877] Trial 22 finished with value: 0.214246109021799 and parameters: {'sigma': 0.1227572367464457, 'lambd': 0.007795627557608306}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,890] Trial 23 finished with value: 0.12724597136541754 and parameters: {'sigma': 0.1955139561393979, 'lambd': 0.432052449706986}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,905] Trial 24 finished with value: 0.22618259142434807 and parameters: {'sigma': 1.5338131890431885, 'lambd': 0.05673321983411569}. Best is trial 6 with value: 0.08655468214633288.
    [I 2025-10-17 17:10:51,917] Trial 25 finished with value: 0.0839396644595838 and parameters: {'sigma': 0.5062447835342266, 'lambd': 0.1641721928230246}. Best is trial 25 with value: 0.0839396644595838.
    [I 2025-10-17 17:10:51,931] Trial 26 finished with value: 0.09461913952075693 and parameters: {'sigma': 0.38279734449507724, 'lambd': 0.1612976072489705}. Best is trial 25 with value: 0.0839396644595838.
    [I 2025-10-17 17:10:51,945] Trial 27 finished with value: 0.11048779571366252 and parameters: {'sigma': 0.010154030999880265, 'lambd': 0.000993428532341461}. Best is trial 25 with value: 0.0839396644595838.
    [I 2025-10-17 17:10:51,959] Trial 28 finished with value: 0.10137898104773796 and parameters: {'sigma': 0.03629209965428483, 'lambd': 6.870102759716288e-05}. Best is trial 25 with value: 0.0839396644595838.
    [I 2025-10-17 17:10:51,972] Trial 29 finished with value: 0.18324432731185777 and parameters: {'sigma': 0.28343457752655665, 'lambd': 0.011804422837614473}. Best is trial 25 with value: 0.0839396644595838.
    [I 2025-10-17 17:10:52,011] A new study created in memory with name: no-name-6efbb49d-d2de-472c-981d-6e83b97032a4


      Using standard kernel method
      KS statistic: 0.0500, p-value: 0.9647
      Best distance: 0.0839
    
    Testing KDE sampling...


    [I 2025-10-17 17:10:52,910] Trial 0 finished with value: 0.18385361197237915 and parameters: {'sigma': 2.0612469473171453, 'lambd': 2.4909683832761306e-05}. Best is trial 0 with value: 0.18385361197237915.
    [I 2025-10-17 17:10:53,730] Trial 1 finished with value: 0.21070755305280472 and parameters: {'sigma': 6.280863368189164, 'lambd': 0.5104498334793268}. Best is trial 0 with value: 0.18385361197237915.
    [I 2025-10-17 17:10:54,576] Trial 2 finished with value: 0.11336026172619304 and parameters: {'sigma': 0.25898949220151785, 'lambd': 0.060873011711652145}. Best is trial 2 with value: 0.11336026172619304.
    [I 2025-10-17 17:10:55,451] Trial 3 finished with value: 0.20188973079013808 and parameters: {'sigma': 1.2249323997904993, 'lambd': 0.0915417778787923}. Best is trial 2 with value: 0.11336026172619304.
    [I 2025-10-17 17:10:56,298] Trial 4 finished with value: 0.16684439544614266 and parameters: {'sigma': 0.09413655744355204, 'lambd': 0.009379793504264945}. Best is trial 2 with value: 0.11336026172619304.
    [I 2025-10-17 17:10:57,198] Trial 5 finished with value: 0.1182555390873471 and parameters: {'sigma': 0.8046438706015777, 'lambd': 1.3801715753622091e-05}. Best is trial 2 with value: 0.11336026172619304.
    [I 2025-10-17 17:10:58,044] Trial 6 finished with value: 0.10863816883871437 and parameters: {'sigma': 0.34747167545210006, 'lambd': 0.7061696877748173}. Best is trial 6 with value: 0.10863816883871437.
    [I 2025-10-17 17:10:59,240] Trial 7 finished with value: 0.19725290909078744 and parameters: {'sigma': 1.1832211323438686, 'lambd': 0.0001842091702679514}. Best is trial 6 with value: 0.10863816883871437.
    [I 2025-10-17 17:11:00,406] Trial 8 finished with value: 0.12501266804695518 and parameters: {'sigma': 0.028255240522301224, 'lambd': 0.0011678640730050848}. Best is trial 6 with value: 0.10863816883871437.
    [I 2025-10-17 17:11:01,456] Trial 9 finished with value: 0.15785329783545088 and parameters: {'sigma': 1.8413423766063775, 'lambd': 0.0003887933265015829}. Best is trial 6 with value: 0.10863816883871437.
    [I 2025-10-17 17:11:02,309] Trial 10 finished with value: 0.1510896215446208 and parameters: {'sigma': 0.026108990764570215, 'lambd': 0.6438638027013004}. Best is trial 6 with value: 0.10863816883871437.
    [I 2025-10-17 17:11:03,186] Trial 11 finished with value: 0.11596141856556216 and parameters: {'sigma': 0.21671102465202588, 'lambd': 0.0357151370986787}. Best is trial 6 with value: 0.10863816883871437.
    [I 2025-10-17 17:11:03,999] Trial 12 finished with value: 0.12141474014778143 and parameters: {'sigma': 0.2043593825870261, 'lambd': 0.08679014364545526}. Best is trial 6 with value: 0.10863816883871437.
    [I 2025-10-17 17:11:04,867] Trial 13 finished with value: 0.09675032326467109 and parameters: {'sigma': 0.0886563038646677, 'lambd': 0.8805290478682786}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:05,698] Trial 14 finished with value: 0.1973624160096657 and parameters: {'sigma': 0.010512908495556349, 'lambd': 0.9613604694579202}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:06,539] Trial 15 finished with value: 0.13042649286063526 and parameters: {'sigma': 0.0758498761540105, 'lambd': 0.016657529151235657}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:07,380] Trial 16 finished with value: 0.1792959182699163 and parameters: {'sigma': 0.08331243794323515, 'lambd': 0.23997762822584448}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:08,325] Trial 17 finished with value: 0.3320075625048077 and parameters: {'sigma': 0.6942526467341413, 'lambd': 0.0040423964395044434}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:09,204] Trial 18 finished with value: 0.1616004751811014 and parameters: {'sigma': 0.3995802106880106, 'lambd': 0.24024205722570893}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:10,044] Trial 19 finished with value: 0.15240247777947877 and parameters: {'sigma': 5.415923969234125, 'lambd': 0.18033475147020625}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:10,899] Trial 20 finished with value: 0.1731717621885773 and parameters: {'sigma': 0.03985433182233417, 'lambd': 0.010387836002364932}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:12,222] Trial 21 finished with value: 0.1940115309553195 and parameters: {'sigma': 0.3009915195912629, 'lambd': 0.05085417560452917}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:13,581] Trial 22 finished with value: 0.12760028994228972 and parameters: {'sigma': 0.1335554861064011, 'lambd': 0.9408771609444139}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:14,465] Trial 23 finished with value: 0.13024616344003676 and parameters: {'sigma': 0.5118824999102928, 'lambd': 0.2825250351194016}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:15,315] Trial 24 finished with value: 0.11215053387501217 and parameters: {'sigma': 0.14455403335916048, 'lambd': 0.10210538660477085}. Best is trial 13 with value: 0.09675032326467109.
    [I 2025-10-17 17:11:16,175] Trial 25 finished with value: 0.08451317043238248 and parameters: {'sigma': 0.0589975521527248, 'lambd': 0.1502134247521258}. Best is trial 25 with value: 0.08451317043238248.
    [I 2025-10-17 17:11:17,020] Trial 26 finished with value: 0.12313487566964666 and parameters: {'sigma': 0.04840561135849796, 'lambd': 0.02435692615210083}. Best is trial 25 with value: 0.08451317043238248.
    [I 2025-10-17 17:11:17,859] Trial 27 finished with value: 0.1590439693216088 and parameters: {'sigma': 0.011361945668579652, 'lambd': 0.39780572632219235}. Best is trial 25 with value: 0.08451317043238248.
    [I 2025-10-17 17:11:18,685] Trial 28 finished with value: 0.11980431712826911 and parameters: {'sigma': 0.05556590247652049, 'lambd': 0.13535457154715133}. Best is trial 25 with value: 0.08451317043238248.
    [I 2025-10-17 17:11:19,589] Trial 29 finished with value: 0.10008414358303983 and parameters: {'sigma': 0.02289576259952689, 'lambd': 0.4103907878072501}. Best is trial 25 with value: 0.08451317043238248.
    [I 2025-10-17 17:11:20,462] A new study created in memory with name: no-name-a7b1dd24-2262-4902-a421-4d2bc5e48046
    [I 2025-10-17 17:11:20,489] Trial 0 finished with value: 0.12895822722134 and parameters: {'sigma': 0.034956995532638625, 'lambd': 0.007768189772006189}. Best is trial 0 with value: 0.12895822722134.
    [I 2025-10-17 17:11:20,518] Trial 1 finished with value: 0.10417172192386831 and parameters: {'sigma': 0.8152401289059421, 'lambd': 0.0033618843302610944}. Best is trial 1 with value: 0.10417172192386831.
    [I 2025-10-17 17:11:20,541] Trial 2 finished with value: 0.06776287428027966 and parameters: {'sigma': 0.07788698658867288, 'lambd': 0.00020831610492576808}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,565] Trial 3 finished with value: 0.1077168798669381 and parameters: {'sigma': 1.490238958421132, 'lambd': 0.0005209680013121514}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,583] Trial 4 finished with value: 0.10025780251540026 and parameters: {'sigma': 0.053443767766548016, 'lambd': 0.00031806576470318437}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,603] Trial 5 finished with value: 0.11806944790498562 and parameters: {'sigma': 9.082728795422774, 'lambd': 0.006817214009927173}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,624] Trial 6 finished with value: 0.10196049032854135 and parameters: {'sigma': 0.8119003791779015, 'lambd': 0.7065028718618618}. Best is trial 2 with value: 0.06776287428027966.


      Using standard kernel method
      KS statistic: 0.1200, p-value: 0.1123
      Best distance: 0.0845
    
    Testing GMM sampling...


    [I 2025-10-17 17:11:20,642] Trial 7 finished with value: 0.10202054230513408 and parameters: {'sigma': 2.589123879684797, 'lambd': 2.4336643995689135e-05}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,660] Trial 8 finished with value: 0.1003501179857539 and parameters: {'sigma': 0.0552752083256954, 'lambd': 0.0002502895728108336}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,680] Trial 9 finished with value: 0.10809038073916517 and parameters: {'sigma': 8.422341568148855, 'lambd': 0.0003928486608969992}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,708] Trial 10 finished with value: 0.14732569462266382 and parameters: {'sigma': 0.012587046949933213, 'lambd': 0.1049139971196517}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,735] Trial 11 finished with value: 0.09386910122247859 and parameters: {'sigma': 0.1564532987232499, 'lambd': 1.3934647091395659e-05}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,761] Trial 12 finished with value: 0.10598893547755553 and parameters: {'sigma': 0.20861387816271568, 'lambd': 2.202146100130993e-05}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,785] Trial 13 finished with value: 0.0986328617664545 and parameters: {'sigma': 0.19677259757685694, 'lambd': 1.102207604133049e-05}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,810] Trial 14 finished with value: 0.12630751082269925 and parameters: {'sigma': 0.12008043375193504, 'lambd': 7.786733521813783e-05}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,845] Trial 15 finished with value: 0.11591375246403887 and parameters: {'sigma': 0.02070761676606431, 'lambd': 7.179804335285579e-05}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,870] Trial 16 finished with value: 0.13059390144544086 and parameters: {'sigma': 0.0942105503135734, 'lambd': 0.0014212975686517322}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,893] Trial 17 finished with value: 0.08303117103400376 and parameters: {'sigma': 0.3851831407226602, 'lambd': 7.213512645076955e-05}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,916] Trial 18 finished with value: 0.08223060133836718 and parameters: {'sigma': 0.4452446939335862, 'lambd': 7.139984488891664e-05}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,936] Trial 19 finished with value: 0.11035227798863596 and parameters: {'sigma': 0.5443717555300995, 'lambd': 0.023567188884567946}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,962] Trial 20 finished with value: 0.10738579095353792 and parameters: {'sigma': 2.851906004987334, 'lambd': 0.0015137747155576971}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:20,986] Trial 21 finished with value: 0.08030228699654501 and parameters: {'sigma': 0.32713692510309056, 'lambd': 8.578804007864842e-05}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:21,008] Trial 22 finished with value: 0.07975553397844479 and parameters: {'sigma': 0.3210493333712269, 'lambd': 0.0001050747360090953}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:21,034] Trial 23 finished with value: 0.07150706957137618 and parameters: {'sigma': 0.07879776085288524, 'lambd': 0.00011192326319081242}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:21,068] Trial 24 finished with value: 0.09076820400204753 and parameters: {'sigma': 0.06922051661132361, 'lambd': 0.0001718821605505469}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:21,104] Trial 25 finished with value: 0.08840740565757359 and parameters: {'sigma': 0.02829332027288815, 'lambd': 0.001387647511286572}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:21,128] Trial 26 finished with value: 0.0814600955186567 and parameters: {'sigma': 0.09395131319161984, 'lambd': 3.1704984474800014e-05}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:21,158] Trial 27 finished with value: 0.08646667647772127 and parameters: {'sigma': 0.01098210917121599, 'lambd': 0.0008173590538287275}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:21,184] Trial 28 finished with value: 0.0908550219064965 and parameters: {'sigma': 0.2182248010075849, 'lambd': 0.0001563781462644853}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:21,203] Trial 29 finished with value: 0.11379626346663022 and parameters: {'sigma': 0.03516192307078593, 'lambd': 0.013991486102733007}. Best is trial 2 with value: 0.06776287428027966.
    [I 2025-10-17 17:11:21,323] A new study created in memory with name: no-name-6c4e24c5-1a41-4674-b186-c105fef2ceaf


      Using standard kernel method
      KS statistic: 0.0800, p-value: 0.5453
      Best distance: 0.0678
    
    Sampling Method Comparison:
    BOOTSTRAP: KS=0.0500, p=0.9647, dist=0.0839
    KDE: KS=0.1200, p=0.1123, dist=0.0845
    GMM: KS=0.0800, p=0.5453, dist=0.0678
    
    ============================================================
    EXAMPLE 5: Digits Dataset (MNIST)
    ============================================================
    Target Distribution: PCA-transformed digits dataset (15D)
    Sample size: 1797


    [I 2025-10-17 17:11:21,567] Trial 0 finished with value: 0.0022246941588308575 and parameters: {'sigma': 3.034386762615776, 'lambd': 0.08547154108887714}. Best is trial 0 with value: 0.0022246941588308575.
    [I 2025-10-17 17:11:21,760] Trial 1 finished with value: 0.0022246941600081375 and parameters: {'sigma': 2.6892900930634025, 'lambd': 0.0018697249698837633}. Best is trial 0 with value: 0.0022246941588308575.
    [I 2025-10-17 17:11:21,968] Trial 2 finished with value: 0.0022246941588200805 and parameters: {'sigma': 0.889465150111555, 'lambd': 0.48565954959270946}. Best is trial 2 with value: 0.0022246941588200805.
    [I 2025-10-17 17:11:22,178] Trial 3 finished with value: 0.0022246986066284187 and parameters: {'sigma': 8.068733682000948, 'lambd': 0.020064757947210014}. Best is trial 2 with value: 0.0022246941588200805.
    [I 2025-10-17 17:11:22,399] Trial 4 finished with value: 0.0022246941588200527 and parameters: {'sigma': 0.06991870532527267, 'lambd': 0.03776240489033323}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:22,622] Trial 5 finished with value: 0.002224694159657839 and parameters: {'sigma': 2.6677164337394768, 'lambd': 0.17553543358987805}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:22,886] Trial 6 finished with value: 0.0022246941588214306 and parameters: {'sigma': 0.5670945028721163, 'lambd': 0.06280326347063207}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:23,120] Trial 7 finished with value: 0.002224694190579349 and parameters: {'sigma': 5.367524461468203, 'lambd': 0.019428842867155276}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:23,317] Trial 8 finished with value: 0.0022246941604723023 and parameters: {'sigma': 0.7421592164533201, 'lambd': 0.01683944803100256}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:23,518] Trial 9 finished with value: 0.0022246941604796506 and parameters: {'sigma': 0.01660105760939393, 'lambd': 1.3350914072279816e-05}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:23,708] Trial 10 finished with value: 0.0022246941589765387 and parameters: {'sigma': 0.04759508404840813, 'lambd': 0.0008718697897702893}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:24,039] Trial 11 finished with value: 0.002224694166250335 and parameters: {'sigma': 0.11470831084824455, 'lambd': 0.6223626873882091}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:24,423] Trial 12 finished with value: 0.002224694158825918 and parameters: {'sigma': 0.27224632936324333, 'lambd': 0.8530735370368501}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:24,711] Trial 13 finished with value: 0.002224694160014713 and parameters: {'sigma': 0.08769247963719867, 'lambd': 0.006130568855976838}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:25,119] Trial 14 finished with value: 0.0022246941588200614 and parameters: {'sigma': 0.018447926289387753, 'lambd': 0.00023877390564767826}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:25,520] Trial 15 finished with value: 0.002224694158826815 and parameters: {'sigma': 0.012186740347780034, 'lambd': 0.00022213301895163704}. Best is trial 4 with value: 0.0022246941588200527.
    [I 2025-10-17 17:11:25,924] Trial 16 finished with value: 0.002224694158817377 and parameters: {'sigma': 0.03401792645922921, 'lambd': 9.898071446851201e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:26,265] Trial 17 finished with value: 0.0022246941588200684 and parameters: {'sigma': 0.03902077296757051, 'lambd': 3.4867190239840804e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:26,524] Trial 18 finished with value: 0.0022246941589003514 and parameters: {'sigma': 0.17781505377400345, 'lambd': 0.00014157303160355938}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:26,818] Trial 19 finished with value: 0.002224694158820057 and parameters: {'sigma': 0.05377121644486395, 'lambd': 0.0008192052604722724}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:27,174] Trial 20 finished with value: 0.0022246941588213547 and parameters: {'sigma': 0.030336430831989416, 'lambd': 0.003284072101731787}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:27,419] Trial 21 finished with value: 0.0022246941588312547 and parameters: {'sigma': 0.07373804292792707, 'lambd': 0.0008294642173975806}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:27,611] Trial 22 finished with value: 0.0022246941588200423 and parameters: {'sigma': 0.025744918564091994, 'lambd': 6.310002769749461e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:27,847] Trial 23 finished with value: 0.002224694158824855 and parameters: {'sigma': 0.02981165985544494, 'lambd': 8.337431157056064e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:28,087] Trial 24 finished with value: 0.0022246941588204847 and parameters: {'sigma': 0.0104728551992276, 'lambd': 1.6256994180625782e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:28,273] Trial 25 finished with value: 0.0022246941588193888 and parameters: {'sigma': 0.14013822845376228, 'lambd': 3.611108306657078e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:28,457] Trial 26 finished with value: 0.0022246941588729644 and parameters: {'sigma': 0.18268196344408066, 'lambd': 5.544768817791607e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:28,632] Trial 27 finished with value: 0.0022246941588200566 and parameters: {'sigma': 0.02341972624222852, 'lambd': 2.7157784465617063e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:28,815] Trial 28 finished with value: 0.0022246941588230477 and parameters: {'sigma': 0.1312416924179274, 'lambd': 0.00038189561481603016}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:28,991] Trial 29 finished with value: 0.0022246941593316817 and parameters: {'sigma': 0.3773644330842283, 'lambd': 9.73932863300472e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:29,185] Trial 30 finished with value: 0.0022246941590743233 and parameters: {'sigma': 0.038422938997017854, 'lambd': 3.921652037235994e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:29,429] Trial 31 finished with value: 0.0022246941588201117 and parameters: {'sigma': 0.058574346273035295, 'lambd': 0.07357418505880296}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:29,661] Trial 32 finished with value: 0.002224694159092144 and parameters: {'sigma': 0.09082863476857933, 'lambd': 1.1149717120966804e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:29,830] Trial 33 finished with value: 0.002224694160437738 and parameters: {'sigma': 0.19063855419325113, 'lambd': 0.00046572078087394936}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:30,030] Trial 34 finished with value: 0.0022246941591147667 and parameters: {'sigma': 0.01933940574950359, 'lambd': 0.007226103955815252}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:30,270] Trial 35 finished with value: 0.002224694158820147 and parameters: {'sigma': 0.062298029383583135, 'lambd': 0.235775711582688}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:30,500] Trial 36 finished with value: 0.002224694158842376 and parameters: {'sigma': 1.9188264123017775, 'lambd': 0.0022705665746547725}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:30,694] Trial 37 finished with value: 0.002224694158823318 and parameters: {'sigma': 0.40467281236659813, 'lambd': 0.037436367060527}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:30,873] Trial 38 finished with value: 0.002224694200054734 and parameters: {'sigma': 0.033795769292273146, 'lambd': 2.5016005314165526e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:31,079] Trial 39 finished with value: 0.002224694158822757 and parameters: {'sigma': 1.291594149988876, 'lambd': 0.18122950360054146}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:31,278] Trial 40 finished with value: 0.0022246941588200553 and parameters: {'sigma': 0.12617871268307465, 'lambd': 6.851256755320141e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:31,518] Trial 41 finished with value: 0.00222469415939099 and parameters: {'sigma': 0.12678691458145042, 'lambd': 6.967723330829376e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:31,718] Trial 42 finished with value: 0.0022246941588202943 and parameters: {'sigma': 0.25254074290814343, 'lambd': 0.00016226347792602138}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:31,953] Trial 43 finished with value: 0.0022246941588200697 and parameters: {'sigma': 0.08079159212986328, 'lambd': 5.760031826931511e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:32,146] Trial 44 finished with value: 0.0022246941588667025 and parameters: {'sigma': 0.11010432156733155, 'lambd': 2.3808723164489988e-05}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:32,413] Trial 45 finished with value: 0.0022246941588211382 and parameters: {'sigma': 0.015636713301948484, 'lambd': 0.00011499414842614425}. Best is trial 16 with value: 0.002224694158817377.
    [I 2025-10-17 17:11:32,608] Trial 46 finished with value: 0.002224694158816916 and parameters: {'sigma': 0.5623977698929833, 'lambd': 0.00043037111468971186}. Best is trial 46 with value: 0.002224694158816916.
    [I 2025-10-17 17:11:32,779] Trial 47 finished with value: 0.0022246941588248553 and parameters: {'sigma': 1.2677994873423124, 'lambd': 0.0005538439511840655}. Best is trial 46 with value: 0.002224694158816916.
    [I 2025-10-17 17:11:32,994] Trial 48 finished with value: 0.002224694158822946 and parameters: {'sigma': 0.6731709461562156, 'lambd': 0.0002794778567212086}. Best is trial 46 with value: 0.002224694158816916.
    [I 2025-10-17 17:11:33,174] Trial 49 finished with value: 0.002224695224673578 and parameters: {'sigma': 9.307515795379192, 'lambd': 0.0011759603836786502}. Best is trial 46 with value: 0.002224694158816916.


      Using RFF with 100 components
    
    Digits Dataset Results:
    Best sigma: 0.562, lambda: 0.000, MMD: 0.002



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_35.png){:class="img-responsive"}
    


    
    ============================================================
    EXAMPLE 6: Olivetti Faces
    ============================================================
    downloading Olivetti faces from https://ndownloader.figshare.com/files/5976027 to /root/scikit_learn_data


    [I 2025-10-17 17:11:40,432] A new study created in memory with name: no-name-096355e6-8b00-4192-8312-5144f28a3d8a
    [I 2025-10-17 17:11:40,459] Trial 0 finished with value: 0.010435174309232848 and parameters: {'sigma': 0.08575473865841494, 'lambd': 0.3448916285289377}. Best is trial 0 with value: 0.010435174309232848.
    [I 2025-10-17 17:11:40,482] Trial 1 finished with value: 0.010396987276751084 and parameters: {'sigma': 4.251763014164457, 'lambd': 0.0012281541434627527}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,511] Trial 2 finished with value: 0.010450219218812317 and parameters: {'sigma': 1.7133407605319777, 'lambd': 0.031089295632681833}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,533] Trial 3 finished with value: 0.010464499360681586 and parameters: {'sigma': 3.247609947261713, 'lambd': 3.0321885123628217e-05}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,550] Trial 4 finished with value: 0.010435731488000251 and parameters: {'sigma': 0.2345259613053879, 'lambd': 0.000559694964639838}. Best is trial 1 with value: 0.010396987276751084.


    Target Distribution: PCA-transformed Olivetti faces (15D)
    Sample size: 400


    [I 2025-10-17 17:11:40,571] Trial 5 finished with value: 0.010444051789784447 and parameters: {'sigma': 8.23622754906347, 'lambd': 0.00036567467073964386}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,602] Trial 6 finished with value: 0.010447883914246479 and parameters: {'sigma': 0.2515797545437043, 'lambd': 0.00018296030758637157}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,623] Trial 7 finished with value: 0.010452019752840585 and parameters: {'sigma': 0.07173852413967749, 'lambd': 0.0007523520408916687}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,645] Trial 8 finished with value: 0.010436178352782974 and parameters: {'sigma': 4.22225061445181, 'lambd': 0.015432428515163334}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,674] Trial 9 finished with value: 0.010422188281427469 and parameters: {'sigma': 5.5726772850495045, 'lambd': 0.0004563977330636227}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,717] Trial 10 finished with value: 0.010440371800466914 and parameters: {'sigma': 0.018046008238406196, 'lambd': 1.6238934343507206e-05}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,770] Trial 11 finished with value: 0.010453554354975799 and parameters: {'sigma': 1.281359288633284, 'lambd': 0.004652337521695351}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,833] Trial 12 finished with value: 0.010401552412228779 and parameters: {'sigma': 0.7854881139254786, 'lambd': 0.002620064932462582}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,888] Trial 13 finished with value: 0.010433883410589796 and parameters: {'sigma': 0.7958272984253381, 'lambd': 0.004834519553497132}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,913] Trial 14 finished with value: 0.010439233501419141 and parameters: {'sigma': 0.5796286551093658, 'lambd': 0.10959705622302783}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:40,992] Trial 15 finished with value: 0.010440590120051598 and parameters: {'sigma': 2.1442241678013136, 'lambd': 0.0031330468155277846}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,101] Trial 16 finished with value: 0.010436889250768404 and parameters: {'sigma': 0.5593065461984615, 'lambd': 9.075606666788495e-05}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,137] Trial 17 finished with value: 0.010442092982733277 and parameters: {'sigma': 0.10166481013065426, 'lambd': 0.002036746514637743}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,161] Trial 18 finished with value: 0.010508034467105846 and parameters: {'sigma': 9.874525894147546, 'lambd': 0.01750156250132312}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,189] Trial 19 finished with value: 0.010439074960179982 and parameters: {'sigma': 1.0627042674566747, 'lambd': 0.06395855981351582}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,222] Trial 20 finished with value: 0.01043385711060315 and parameters: {'sigma': 2.874794989015365, 'lambd': 0.00010390006742902309}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,245] Trial 21 finished with value: 0.01044790444325715 and parameters: {'sigma': 5.037640011505679, 'lambd': 0.0010031019025123064}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,279] Trial 22 finished with value: 0.010447604083123061 and parameters: {'sigma': 5.677105603062788, 'lambd': 0.0019113736990036467}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,305] Trial 23 finished with value: 0.0104349081151009 and parameters: {'sigma': 2.031766240260359, 'lambd': 0.00774511205826035}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,330] Trial 24 finished with value: 0.010442679787851957 and parameters: {'sigma': 0.45829806327463346, 'lambd': 0.00023731606934031718}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,357] Trial 25 finished with value: 0.010445269031566931 and parameters: {'sigma': 6.483581054982775, 'lambd': 0.0015137564412805958}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,384] Trial 26 finished with value: 0.010435846637064025 and parameters: {'sigma': 0.015280565709917874, 'lambd': 3.568880495988801e-05}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,421] Trial 27 finished with value: 0.01044770950724172 and parameters: {'sigma': 2.6909403880892695, 'lambd': 0.010225633558399694}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,457] Trial 28 finished with value: 0.010449011006768994 and parameters: {'sigma': 1.3733458606957063, 'lambd': 0.000554136046212649}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,487] Trial 29 finished with value: 0.010431252279812625 and parameters: {'sigma': 0.12238013946078506, 'lambd': 0.2516335734758182}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,524] Trial 30 finished with value: 0.01043721707511792 and parameters: {'sigma': 0.0413576081539471, 'lambd': 9.987982047900427e-05}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,552] Trial 31 finished with value: 0.010445326697029822 and parameters: {'sigma': 0.12902342585393034, 'lambd': 0.23889171339231788}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,578] Trial 32 finished with value: 0.010434102752783373 and parameters: {'sigma': 0.19271690001146902, 'lambd': 0.2684699065946425}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,599] Trial 33 finished with value: 0.010452925848488446 and parameters: {'sigma': 0.05317575395216421, 'lambd': 0.7783569372177462}. Best is trial 1 with value: 0.010396987276751084.
    [I 2025-10-17 17:11:41,632] Trial 34 finished with value: 0.010394132679594987 and parameters: {'sigma': 0.026995750985447292, 'lambd': 0.0397536584244774}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,676] Trial 35 finished with value: 0.010441856247930295 and parameters: {'sigma': 0.024283613603171016, 'lambd': 0.033738539063730394}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,706] Trial 36 finished with value: 0.010434240480429537 and parameters: {'sigma': 0.37474848372493474, 'lambd': 0.0011850558047535656}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,733] Trial 37 finished with value: 0.010431494084163167 and parameters: {'sigma': 0.01095005905324633, 'lambd': 0.0004194657348118424}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,757] Trial 38 finished with value: 0.010460956981428963 and parameters: {'sigma': 3.759902300722846, 'lambd': 0.0002340159312791432}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,797] Trial 39 finished with value: 0.010514925344105377 and parameters: {'sigma': 7.269427417543369, 'lambd': 0.03517762196196378}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,826] Trial 40 finished with value: 0.010434414132081033 and parameters: {'sigma': 0.8383776739390186, 'lambd': 0.0037852317817186187}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,858] Trial 41 finished with value: 0.010439225873140086 and parameters: {'sigma': 0.03466255729584727, 'lambd': 0.5622451843823391}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,886] Trial 42 finished with value: 0.010431547476700151 and parameters: {'sigma': 0.11191533229171073, 'lambd': 0.14852781982635344}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,931] Trial 43 finished with value: 0.010437760783759222 and parameters: {'sigma': 0.29304410576073947, 'lambd': 0.06585795193628019}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,966] Trial 44 finished with value: 0.010447128853138707 and parameters: {'sigma': 0.1761579648540614, 'lambd': 0.0007114886734287087}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:41,992] Trial 45 finished with value: 0.010437070676912158 and parameters: {'sigma': 0.06317698294158555, 'lambd': 0.016917928131179486}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:42,017] Trial 46 finished with value: 0.010443381264997198 and parameters: {'sigma': 4.2736235594362135, 'lambd': 0.0026270957776364364}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:42,044] Trial 47 finished with value: 0.010402955193909982 and parameters: {'sigma': 1.5282541142329686, 'lambd': 0.0065234536327331005}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:42,078] Trial 48 finished with value: 0.010447562502403496 and parameters: {'sigma': 1.623582841688127, 'lambd': 0.0053930867400643805}. Best is trial 34 with value: 0.010394132679594987.
    [I 2025-10-17 17:11:42,114] Trial 49 finished with value: 0.01043688379425865 and parameters: {'sigma': 0.7117082932402922, 'lambd': 0.008416896619186718}. Best is trial 34 with value: 0.010394132679594987.


      Using RFF with 50 components
    
    Olivetti Faces Results:
    Best sigma: 0.027, lambda: 0.040, MMD: 0.010



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_41.png){:class="img-responsive"}
    


    
    ============================================================
    EXAMPLE 7: Fashion-MNIST
    ============================================================
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    [1m29515/29515[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    [1m26421880/26421880[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    [1m5148/5148[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    [1m4422102/4422102[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step


    [I 2025-10-17 17:11:48,784] A new study created in memory with name: no-name-8db119a6-35a2-49b3-a3d1-12e0472aaabb


    Target Distribution: PCA-transformed Fashion-MNIST (15D, subsampled)
    Sample size: 5000


    [I 2025-10-17 17:11:49,644] Trial 0 finished with value: 0.0019892355629470304 and parameters: {'sigma': 0.06946171483984556, 'lambd': 1.1725369530801314e-05}. Best is trial 0 with value: 0.0019892355629470304.
    [I 2025-10-17 17:11:50,714] Trial 1 finished with value: 0.0019702610793439356 and parameters: {'sigma': 0.02272505476707354, 'lambd': 1.5754730460516142e-05}. Best is trial 1 with value: 0.0019702610793439356.
    [I 2025-10-17 17:11:51,841] Trial 2 finished with value: 0.001989514022158744 and parameters: {'sigma': 0.028114742237218656, 'lambd': 0.0009307603249683455}. Best is trial 1 with value: 0.0019702610793439356.
    [I 2025-10-17 17:11:52,771] Trial 3 finished with value: 0.001973252583294894 and parameters: {'sigma': 1.40580431451803, 'lambd': 0.0028856420300471834}. Best is trial 1 with value: 0.0019702610793439356.
    [I 2025-10-17 17:11:53,526] Trial 4 finished with value: 0.0018773022155804731 and parameters: {'sigma': 0.1376705383897862, 'lambd': 0.6973118219875718}. Best is trial 4 with value: 0.0018773022155804731.
    [I 2025-10-17 17:11:54,285] Trial 5 finished with value: 0.0018731660015017335 and parameters: {'sigma': 7.39106998903997, 'lambd': 6.27565645902574e-05}. Best is trial 5 with value: 0.0018731660015017335.
    [I 2025-10-17 17:11:54,982] Trial 6 finished with value: 0.001985048317795435 and parameters: {'sigma': 1.0901548172124569, 'lambd': 7.965857353435175e-05}. Best is trial 5 with value: 0.0018731660015017335.
    [I 2025-10-17 17:11:55,740] Trial 7 finished with value: 0.001982513553393475 and parameters: {'sigma': 0.022789118667394043, 'lambd': 0.004474395853512804}. Best is trial 5 with value: 0.0018731660015017335.
    [I 2025-10-17 17:11:56,570] Trial 8 finished with value: 0.0019716712414303327 and parameters: {'sigma': 0.02596272593329039, 'lambd': 4.92277136803802e-05}. Best is trial 5 with value: 0.0018731660015017335.
    [I 2025-10-17 17:11:57,398] Trial 9 finished with value: 0.001960767021038922 and parameters: {'sigma': 0.15525724234544244, 'lambd': 7.430033577500031e-05}. Best is trial 5 with value: 0.0018731660015017335.
    [I 2025-10-17 17:11:58,239] Trial 10 finished with value: 0.0014793507566249933 and parameters: {'sigma': 9.117639887039616, 'lambd': 0.043217613167101744}. Best is trial 10 with value: 0.0014793507566249933.
    [I 2025-10-17 17:11:59,028] Trial 11 finished with value: 0.0013878407121193533 and parameters: {'sigma': 9.492128792325992, 'lambd': 0.09336243679214411}. Best is trial 11 with value: 0.0013878407121193533.
    [I 2025-10-17 17:11:59,808] Trial 12 finished with value: 0.0012810204657137088 and parameters: {'sigma': 9.738194318632576, 'lambd': 0.16494592118166912}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:00,657] Trial 13 finished with value: 0.0018927586239235608 and parameters: {'sigma': 2.5344523583164564, 'lambd': 0.4175736276179542}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:01,610] Trial 14 finished with value: 0.001897025807817745 and parameters: {'sigma': 3.158112631133682, 'lambd': 0.07246515094937574}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:03,475] Trial 15 finished with value: 0.0020135272898608165 and parameters: {'sigma': 0.5221437267862932, 'lambd': 0.06717174803386658}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:04,474] Trial 16 finished with value: 0.0018594707948543124 and parameters: {'sigma': 4.6588292167218315, 'lambd': 0.021130863340629694}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:05,779] Trial 17 finished with value: 0.0019614090778039776 and parameters: {'sigma': 0.5509081731758566, 'lambd': 0.23795346316040708}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:06,835] Trial 18 finished with value: 0.0019380179212026703 and parameters: {'sigma': 1.8842244445193719, 'lambd': 0.01667131383285827}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:07,515] Trial 19 finished with value: 0.001607707996818578 and parameters: {'sigma': 5.482595666961997, 'lambd': 0.16407746355950564}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:08,309] Trial 20 finished with value: 0.0019358883929473888 and parameters: {'sigma': 0.9019852744346644, 'lambd': 0.9832639160515017}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:09,113] Trial 21 finished with value: 0.0015611664049048923 and parameters: {'sigma': 9.334474358275688, 'lambd': 0.018894884129579785}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:09,923] Trial 22 finished with value: 0.001829313637646465 and parameters: {'sigma': 4.136331055975914, 'lambd': 0.10052416670989114}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:10,762] Trial 23 finished with value: 0.0016732588468057349 and parameters: {'sigma': 9.919800560249579, 'lambd': 0.006385292104556479}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:11,461] Trial 24 finished with value: 0.0018758695869781606 and parameters: {'sigma': 3.03767040736032, 'lambd': 0.03895925656872617}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:12,151] Trial 25 finished with value: 0.0019001465945362676 and parameters: {'sigma': 5.670537859845475, 'lambd': 0.0012319630765093871}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:12,861] Trial 26 finished with value: 0.0019268799871362285 and parameters: {'sigma': 0.012101029381927713, 'lambd': 0.16046304280677928}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:13,666] Trial 27 finished with value: 0.0019284133737639034 and parameters: {'sigma': 2.0551311458817247, 'lambd': 0.39649851351461846}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:14,420] Trial 28 finished with value: 0.001710034554445046 and parameters: {'sigma': 6.257217180311695, 'lambd': 0.03992916352638639}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:15,310] Trial 29 finished with value: 0.001872444439359557 and parameters: {'sigma': 3.629118962892809, 'lambd': 0.009268783531128078}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:16,102] Trial 30 finished with value: 0.0020163497759163127 and parameters: {'sigma': 0.300597011501817, 'lambd': 0.00038157649881761385}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:17,053] Trial 31 finished with value: 0.0015368357494719542 and parameters: {'sigma': 9.21084977900447, 'lambd': 0.02648947114752872}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:18,214] Trial 32 finished with value: 0.0017010484094551628 and parameters: {'sigma': 6.857208650757891, 'lambd': 0.03831528998706033}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:19,297] Trial 33 finished with value: 0.0017025343977046072 and parameters: {'sigma': 8.309657775299506, 'lambd': 0.011428601245222198}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:20,221] Trial 34 finished with value: 0.0017652731953977706 and parameters: {'sigma': 4.434092269449606, 'lambd': 0.10299067642565476}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:21,024] Trial 35 finished with value: 0.0017411870050461696 and parameters: {'sigma': 9.925779416030334, 'lambd': 0.0020407423323542413}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:21,715] Trial 36 finished with value: 0.0019528985222339187 and parameters: {'sigma': 0.04521047937328109, 'lambd': 0.408370884316286}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:22,493] Trial 37 finished with value: 0.0019594436991868 and parameters: {'sigma': 1.706948602497944, 'lambd': 0.03545115322520242}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:23,188] Trial 38 finished with value: 0.0019243263230261107 and parameters: {'sigma': 2.6878096214179528, 'lambd': 0.18465834357758582}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:23,875] Trial 39 finished with value: 0.0019761385666847518 and parameters: {'sigma': 0.980198423177681, 'lambd': 0.07477947057649174}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:24,557] Trial 40 finished with value: 0.001991593237422492 and parameters: {'sigma': 0.09851736704679352, 'lambd': 0.00034334955865135105}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:25,335] Trial 41 finished with value: 0.0014559032192252325 and parameters: {'sigma': 9.98864807637654, 'lambd': 0.023961143105579463}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:26,049] Trial 42 finished with value: 0.001709994760638889 and parameters: {'sigma': 6.740702432452856, 'lambd': 0.023172027419786564}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:26,781] Trial 43 finished with value: 0.001811446319828209 and parameters: {'sigma': 6.647520233772108, 'lambd': 0.005797009864829972}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:27,526] Trial 44 finished with value: 0.0016527830242910712 and parameters: {'sigma': 4.71102586739665, 'lambd': 0.26573718566238563}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:28,211] Trial 45 finished with value: 0.0015873377172766216 and parameters: {'sigma': 9.959416399110289, 'lambd': 0.010077289678790829}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:29,140] Trial 46 finished with value: 0.0019241300118093489 and parameters: {'sigma': 3.6267191091637003, 'lambd': 0.0029455297813174275}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:29,880] Trial 47 finished with value: 0.001537920259246175 and parameters: {'sigma': 7.066567258887439, 'lambd': 0.059219228224031265}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:30,858] Trial 48 finished with value: 0.001936861941320821 and parameters: {'sigma': 2.346524115527226, 'lambd': 0.10097414279514345}. Best is trial 12 with value: 0.0012810204657137088.
    [I 2025-10-17 17:12:31,973] Trial 49 finished with value: 0.0019038140707764376 and parameters: {'sigma': 1.3197362693838952, 'lambd': 0.6363414187898893}. Best is trial 12 with value: 0.0012810204657137088.


      Using RFF with 200 components
    
    Fashion-MNIST Results:
    Best sigma: 9.738, lambda: 0.165, MMD: 0.001



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_47.png){:class="img-responsive"}
    


    [I 2025-10-17 17:12:36,466] A new study created in memory with name: no-name-5c72157a-0f2d-457f-8170-64171fd87a63
    [I 2025-10-17 17:12:36,565] Trial 0 finished with value: 0.00553569766506351 and parameters: {'sigma': 0.8460877586011375, 'lambd': 4.709165032932877e-05}. Best is trial 0 with value: 0.00553569766506351.


    
    ============================================================
    EXAMPLE 8: Advanced Sampling Methods Comparison
    ============================================================
    Comparing sampling methods on Fashion-MNIST...
    
    Testing BOOTSTRAP sampling...


    [I 2025-10-17 17:12:36,723] Trial 1 finished with value: 0.005509638711471382 and parameters: {'sigma': 0.14478590769189262, 'lambd': 0.6196548348377662}. Best is trial 1 with value: 0.005509638711471382.
    [I 2025-10-17 17:12:36,821] Trial 2 finished with value: 0.0055321322710836385 and parameters: {'sigma': 0.2769139882838393, 'lambd': 0.00011247903709532683}. Best is trial 1 with value: 0.005509638711471382.
    [I 2025-10-17 17:12:36,945] Trial 3 finished with value: 0.005401401174134694 and parameters: {'sigma': 0.10753643693509313, 'lambd': 0.8536164744141534}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:37,065] Trial 4 finished with value: 0.00549882856138778 and parameters: {'sigma': 3.86686406243945, 'lambd': 1.7172241932031487e-05}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:37,180] Trial 5 finished with value: 0.005535739625601262 and parameters: {'sigma': 0.18816013314617402, 'lambd': 0.2164279340068768}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:37,331] Trial 6 finished with value: 0.005469026960345116 and parameters: {'sigma': 0.32620613277980826, 'lambd': 2.6105033627519313e-05}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:37,458] Trial 7 finished with value: 0.005559284600610345 and parameters: {'sigma': 0.20367817690633153, 'lambd': 0.0069515511771439924}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:37,564] Trial 8 finished with value: 0.005492148588952955 and parameters: {'sigma': 0.06173457292575633, 'lambd': 0.5197030795580642}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:37,651] Trial 9 finished with value: 0.005539966578358715 and parameters: {'sigma': 1.2293632503402854, 'lambd': 0.005132088798932314}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:37,759] Trial 10 finished with value: 0.005521340083253729 and parameters: {'sigma': 0.010797092605158308, 'lambd': 0.03332860434586925}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:37,885] Trial 11 finished with value: 0.0055460791170889034 and parameters: {'sigma': 0.03204986680421774, 'lambd': 0.0002549316372087186}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:38,003] Trial 12 finished with value: 0.005517433601275061 and parameters: {'sigma': 1.1168370928094389, 'lambd': 0.0006951172174833457}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:38,134] Trial 13 finished with value: 0.005568358284339592 and parameters: {'sigma': 0.05014279408405909, 'lambd': 0.055774253823918415}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:38,246] Trial 14 finished with value: 0.005506408047416418 and parameters: {'sigma': 8.520856137042768, 'lambd': 0.001500026494658055}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:38,368] Trial 15 finished with value: 0.005511222408664484 and parameters: {'sigma': 0.6253100626979734, 'lambd': 1.4115939685122403e-05}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:38,463] Trial 16 finished with value: 0.005589581524695385 and parameters: {'sigma': 0.08858842107358524, 'lambd': 0.016136128956645714}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:38,587] Trial 17 finished with value: 0.005559239838292292 and parameters: {'sigma': 0.02444503100597354, 'lambd': 0.11717078426640247}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:38,674] Trial 18 finished with value: 0.005502502576419136 and parameters: {'sigma': 2.420408875099419, 'lambd': 0.0005584812576642319}. Best is trial 3 with value: 0.005401401174134694.
    [I 2025-10-17 17:12:38,784] Trial 19 finished with value: 0.0055109032771257956 and parameters: {'sigma': 0.3882836682661641, 'lambd': 8.687751417748745e-05}. Best is trial 3 with value: 0.005401401174134694.


      Using RFF with 100 components


    [I 2025-10-17 17:12:39,113] A new study created in memory with name: no-name-af6be05f-72ae-48f8-b230-f35104dfb193


      Best MMD: 0.0054
      Energy distance: 0.0452
      Current MMD: 0.0034
    
    Testing KDE sampling...


    [I 2025-10-17 17:12:42,721] Trial 0 finished with value: 0.005658769499816391 and parameters: {'sigma': 1.380467255278301, 'lambd': 3.31866815767531e-05}. Best is trial 0 with value: 0.005658769499816391.
    [I 2025-10-17 17:12:47,274] Trial 1 finished with value: 0.005651414286277011 and parameters: {'sigma': 0.1034818414930744, 'lambd': 0.0002829179325931842}. Best is trial 1 with value: 0.005651414286277011.
    [I 2025-10-17 17:12:50,921] Trial 2 finished with value: 0.005633985649330602 and parameters: {'sigma': 0.17764969068388775, 'lambd': 0.2862297609501407}. Best is trial 2 with value: 0.005633985649330602.
    [I 2025-10-17 17:12:54,532] Trial 3 finished with value: 0.005649495177212006 and parameters: {'sigma': 0.012764194388169134, 'lambd': 0.0019287445133986416}. Best is trial 2 with value: 0.005633985649330602.
    [I 2025-10-17 17:12:58,809] Trial 4 finished with value: 0.005643061551161458 and parameters: {'sigma': 2.122483243662381, 'lambd': 0.13284001778471444}. Best is trial 2 with value: 0.005633985649330602.
    [I 2025-10-17 17:13:02,377] Trial 5 finished with value: 0.005656296901449886 and parameters: {'sigma': 1.6242171124934222, 'lambd': 0.03950247464015279}. Best is trial 2 with value: 0.005633985649330602.
    [I 2025-10-17 17:13:06,052] Trial 6 finished with value: 0.005648690210064469 and parameters: {'sigma': 0.010054105629979196, 'lambd': 0.0010696250626137896}. Best is trial 2 with value: 0.005633985649330602.
    [I 2025-10-17 17:13:10,352] Trial 7 finished with value: 0.005633042804987078 and parameters: {'sigma': 0.6280274306224791, 'lambd': 0.018085931470243875}. Best is trial 7 with value: 0.005633042804987078.
    [I 2025-10-17 17:13:14,146] Trial 8 finished with value: 0.0056129821806538225 and parameters: {'sigma': 0.036822710620171985, 'lambd': 0.002715752616062836}. Best is trial 8 with value: 0.0056129821806538225.
    [I 2025-10-17 17:13:17,681] Trial 9 finished with value: 0.005630913091217001 and parameters: {'sigma': 0.01686930040278382, 'lambd': 0.676834720966076}. Best is trial 8 with value: 0.0056129821806538225.
    [I 2025-10-17 17:13:21,470] Trial 10 finished with value: 0.005655041183885723 and parameters: {'sigma': 0.0596156409229364, 'lambd': 3.620407210771079e-05}. Best is trial 8 with value: 0.0056129821806538225.
    [I 2025-10-17 17:13:25,848] Trial 11 finished with value: 0.005653092421975471 and parameters: {'sigma': 0.03691241361516929, 'lambd': 0.010721804534624233}. Best is trial 8 with value: 0.0056129821806538225.
    [I 2025-10-17 17:13:29,420] Trial 12 finished with value: 0.005395144808252892 and parameters: {'sigma': 8.568999733994703, 'lambd': 0.9084036519711657}. Best is trial 12 with value: 0.005395144808252892.
    [I 2025-10-17 17:13:32,849] Trial 13 finished with value: 0.0056472251854840436 and parameters: {'sigma': 8.275377117604526, 'lambd': 0.0003793435521643158}. Best is trial 12 with value: 0.005395144808252892.
    [I 2025-10-17 17:13:37,143] Trial 14 finished with value: 0.005628793022537269 and parameters: {'sigma': 4.916768664009718, 'lambd': 0.006096316548000802}. Best is trial 12 with value: 0.005395144808252892.
    [I 2025-10-17 17:13:40,723] Trial 15 finished with value: 0.005643001870929179 and parameters: {'sigma': 0.3531879300337029, 'lambd': 0.07955500692518286}. Best is trial 12 with value: 0.005395144808252892.
    [I 2025-10-17 17:13:44,494] Trial 16 finished with value: 0.005648789267993222 and parameters: {'sigma': 0.03268222695503872, 'lambd': 0.00015056409841733735}. Best is trial 12 with value: 0.005395144808252892.
    [I 2025-10-17 17:13:48,743] Trial 17 finished with value: 0.005644687804291291 and parameters: {'sigma': 0.2727085341898026, 'lambd': 0.0037210685832827508}. Best is trial 12 with value: 0.005395144808252892.
    [I 2025-10-17 17:13:53,104] Trial 18 finished with value: 0.005652515433235846 and parameters: {'sigma': 3.5771067989720704, 'lambd': 0.0007802415244630936}. Best is trial 12 with value: 0.005395144808252892.
    [I 2025-10-17 17:13:56,637] Trial 19 finished with value: 0.0056480259133483 and parameters: {'sigma': 0.6707852721252372, 'lambd': 0.02799307859284754}. Best is trial 12 with value: 0.005395144808252892.


      Using RFF with 100 components


    [I 2025-10-17 17:14:01,101] A new study created in memory with name: no-name-ce50086f-5f22-4b5c-ba67-553c3f03098f


      Best MMD: 0.0054
      Energy distance: 0.0273
      Current MMD: 0.0033
    
    Testing GMM sampling...


    [I 2025-10-17 17:14:01,428] Trial 0 finished with value: 0.005610397333872637 and parameters: {'sigma': 0.07280555116630144, 'lambd': 0.0012309102713446358}. Best is trial 0 with value: 0.005610397333872637.
    [I 2025-10-17 17:14:01,696] Trial 1 finished with value: 0.0056035839727165415 and parameters: {'sigma': 3.162844495844304, 'lambd': 0.0020446159967166867}. Best is trial 1 with value: 0.0056035839727165415.
    [I 2025-10-17 17:14:01,973] Trial 2 finished with value: 0.005613874472728539 and parameters: {'sigma': 0.044606341640329436, 'lambd': 5.911256706499439e-05}. Best is trial 1 with value: 0.0056035839727165415.
    [I 2025-10-17 17:14:02,237] Trial 3 finished with value: 0.005596109545553306 and parameters: {'sigma': 0.17194990826192433, 'lambd': 0.003913730958511346}. Best is trial 3 with value: 0.005596109545553306.
    [I 2025-10-17 17:14:02,498] Trial 4 finished with value: 0.005629639522832094 and parameters: {'sigma': 0.0856887563234124, 'lambd': 0.002271310626424931}. Best is trial 3 with value: 0.005596109545553306.
    [I 2025-10-17 17:14:02,674] Trial 5 finished with value: 0.005623484644812446 and parameters: {'sigma': 0.035021367307924926, 'lambd': 0.26514591276506466}. Best is trial 3 with value: 0.005596109545553306.
    [I 2025-10-17 17:14:02,940] Trial 6 finished with value: 0.0056150522397781745 and parameters: {'sigma': 0.08744193523381705, 'lambd': 9.25220170532846e-05}. Best is trial 3 with value: 0.005596109545553306.
    [I 2025-10-17 17:14:03,085] Trial 7 finished with value: 0.00557640200939819 and parameters: {'sigma': 0.011002280636680253, 'lambd': 0.03263121416274191}. Best is trial 7 with value: 0.00557640200939819.
    [I 2025-10-17 17:14:03,211] Trial 8 finished with value: 0.005619949758078718 and parameters: {'sigma': 2.6771723876682816, 'lambd': 1.0189431257801014e-05}. Best is trial 7 with value: 0.00557640200939819.
    [I 2025-10-17 17:14:03,375] Trial 9 finished with value: 0.005620407244002331 and parameters: {'sigma': 0.0676024291017315, 'lambd': 0.0002727084877246502}. Best is trial 7 with value: 0.00557640200939819.
    [I 2025-10-17 17:14:03,519] Trial 10 finished with value: 0.005619817943597596 and parameters: {'sigma': 0.7825415091133227, 'lambd': 0.0900147489497579}. Best is trial 7 with value: 0.00557640200939819.
    [I 2025-10-17 17:14:03,721] Trial 11 finished with value: 0.005604591572310991 and parameters: {'sigma': 0.010150342980155434, 'lambd': 0.017121444818376148}. Best is trial 7 with value: 0.00557640200939819.
    [I 2025-10-17 17:14:03,955] Trial 12 finished with value: 0.0056236327098506695 and parameters: {'sigma': 0.4414904063425065, 'lambd': 0.024887094783313253}. Best is trial 7 with value: 0.00557640200939819.
    [I 2025-10-17 17:14:04,187] Trial 13 finished with value: 0.005475731758217264 and parameters: {'sigma': 9.332659232824437, 'lambd': 0.677775250132715}. Best is trial 13 with value: 0.005475731758217264.
    [I 2025-10-17 17:14:04,314] Trial 14 finished with value: 0.005481190582833648 and parameters: {'sigma': 8.352026764652255, 'lambd': 0.6674063513102156}. Best is trial 13 with value: 0.005475731758217264.
    [I 2025-10-17 17:14:04,509] Trial 15 finished with value: 0.005472998290923025 and parameters: {'sigma': 8.907311767116616, 'lambd': 0.9267586560980672}. Best is trial 15 with value: 0.005472998290923025.
    [I 2025-10-17 17:14:04,710] Trial 16 finished with value: 0.00547682095777651 and parameters: {'sigma': 7.1786399692592555, 'lambd': 0.8212175741302802}. Best is trial 15 with value: 0.005472998290923025.
    [I 2025-10-17 17:14:04,989] Trial 17 finished with value: 0.005604382310450571 and parameters: {'sigma': 1.5026300953541887, 'lambd': 0.15593225368282737}. Best is trial 15 with value: 0.005472998290923025.
    [I 2025-10-17 17:14:05,187] Trial 18 finished with value: 0.0055362349336269484 and parameters: {'sigma': 4.304015788116848, 'lambd': 0.260550096892479}. Best is trial 15 with value: 0.005472998290923025.
    [I 2025-10-17 17:14:05,437] Trial 19 finished with value: 0.005617675095359431 and parameters: {'sigma': 1.2164751789525714, 'lambd': 0.8929401849821599}. Best is trial 15 with value: 0.005472998290923025.


      Using RFF with 100 components
      Best MMD: 0.0055
      Energy distance: 0.0429
      Current MMD: 0.0034
    
    Fashion-MNIST Sampling Method Comparison:
    --------------------------------------------------
    BOOTSTRAP    | MMD: 0.0054 | Energy: 0.0452 | σ: 0.108 | λ: 0.854
    KDE          | MMD: 0.0054 | Energy: 0.0273 | σ: 8.569 | λ: 0.908
    GMM          | MMD: 0.0055 | Energy: 0.0429 | σ: 8.907 | λ: 0.927
    
    ============================================================
    EXAMPLE 9: Statistical Tests on Image Datasets
    ============================================================
    
    Statistical Tests for Digits:
    ----------------------------------------
    Energy Distance: 0.071668
    MMD: 0.001113
    KS test p-values (first 5 dims): ['0.5432', '0.1117', '0.1644', '0.2692', '0.1525']
    Min KS p-value: 0.1117
    Average correlation difference: 0.0231
    
    Statistical Tests for Olivetti:
    ----------------------------------------
    Energy Distance: 0.042560
    MMD: 0.005548
    KS test p-values (first 5 dims): ['0.3222', '0.4680', '0.8134', '0.6405', '0.1314']
    Min KS p-value: 0.1314
    Average correlation difference: 0.0512
    
    Statistical Tests for Fashion-MNIST:
    ----------------------------------------
    Energy Distance: 0.005224


    [I 2025-10-17 17:14:09,658] A new study created in memory with name: no-name-27e43835-4718-438b-8e26-699221c5e683


    MMD: 0.000703
    KS test p-values (first 5 dims): ['0.0174', '0.4355', '0.9229', '0.0197', '0.8897']
    Min KS p-value: 0.0174
    Average correlation difference: 0.0178
    
    ============================================================
    SUMMARY: All Dataset Results
    ============================================================
    
    Performance Summary:
    --------------------------------------------------------------------------------
    Dataset         MMD        Energy Dist  Min KS p-val Avg Corr Diff
    --------------------------------------------------------------------------------
    Digits          0.001113   0.071668     0.1117       0.0231      
    Olivetti        0.005548   0.042560     0.1314       0.0512      
    Fashion-MNIST   0.000703   0.005224     0.0174       0.0178      
    
    ============================================================
    EXAMPLE 10: Quality Assessment with Different PCA Dimensions
    ============================================================
    Testing different PCA dimensions on Fashion-MNIST...
    
    PCA with 5 components:


    [I 2025-10-17 17:14:09,882] Trial 0 finished with value: 0.007395913168342295 and parameters: {'sigma': 0.8820313607703371, 'lambd': 0.06253790197703084}. Best is trial 0 with value: 0.007395913168342295.
    [I 2025-10-17 17:14:10,087] Trial 1 finished with value: 0.007749204063147156 and parameters: {'sigma': 0.8620350071972787, 'lambd': 2.6800483822530356e-05}. Best is trial 0 with value: 0.007395913168342295.
    [I 2025-10-17 17:14:10,263] Trial 2 finished with value: 0.007252565910786242 and parameters: {'sigma': 0.023846568931949773, 'lambd': 1.1465202899474279e-05}. Best is trial 2 with value: 0.007252565910786242.
    [I 2025-10-17 17:14:10,479] Trial 3 finished with value: 0.0058625509097887 and parameters: {'sigma': 3.195064544953249, 'lambd': 0.0001615568821195399}. Best is trial 3 with value: 0.0058625509097887.
    [I 2025-10-17 17:14:10,694] Trial 4 finished with value: 0.004217487037452004 and parameters: {'sigma': 2.6778072212636874, 'lambd': 0.12490791044712994}. Best is trial 4 with value: 0.004217487037452004.
    [I 2025-10-17 17:14:10,905] Trial 5 finished with value: 0.007861045792178228 and parameters: {'sigma': 0.10528291591651531, 'lambd': 0.0006171244286537342}. Best is trial 4 with value: 0.004217487037452004.
    [I 2025-10-17 17:14:11,120] Trial 6 finished with value: 0.00796508328670374 and parameters: {'sigma': 0.6633119026632547, 'lambd': 0.0014583629803455777}. Best is trial 4 with value: 0.004217487037452004.
    [I 2025-10-17 17:14:11,333] Trial 7 finished with value: 0.007542610276465331 and parameters: {'sigma': 0.025909318891564753, 'lambd': 0.02733199066719973}. Best is trial 4 with value: 0.004217487037452004.
    [I 2025-10-17 17:14:11,481] Trial 8 finished with value: 0.007763520772336841 and parameters: {'sigma': 0.1028858964151467, 'lambd': 0.0003548116813234751}. Best is trial 4 with value: 0.004217487037452004.
    [I 2025-10-17 17:14:11,697] Trial 9 finished with value: 0.007931592513185898 and parameters: {'sigma': 0.024647590405361568, 'lambd': 0.0007808648127586882}. Best is trial 4 with value: 0.004217487037452004.
    [I 2025-10-17 17:14:11,881] Trial 10 finished with value: 0.0029512509871078597 and parameters: {'sigma': 9.305207442855584, 'lambd': 0.8421297720085298}. Best is trial 10 with value: 0.0029512509871078597.
    [I 2025-10-17 17:14:12,095] Trial 11 finished with value: 0.0031186270927277687 and parameters: {'sigma': 8.777256904578405, 'lambd': 0.8276065382899112}. Best is trial 10 with value: 0.0029512509871078597.
    [I 2025-10-17 17:14:12,267] Trial 12 finished with value: 0.003011164650953569 and parameters: {'sigma': 9.2655229393007, 'lambd': 0.9670013542894392}. Best is trial 10 with value: 0.0029512509871078597.
    [I 2025-10-17 17:14:12,519] Trial 13 finished with value: 0.002876934337141066 and parameters: {'sigma': 9.363685776775727, 'lambd': 0.9010782687757858}. Best is trial 13 with value: 0.002876934337141066.
    [I 2025-10-17 17:14:12,756] Trial 14 finished with value: 0.004816321382929627 and parameters: {'sigma': 3.236875355914408, 'lambd': 0.007506616800751803}. Best is trial 13 with value: 0.002876934337141066.
    [I 2025-10-17 17:14:13,099] Trial 15 finished with value: 0.0033731107384530126 and parameters: {'sigma': 4.8663451120678465, 'lambd': 0.23544181344574067}. Best is trial 13 with value: 0.002876934337141066.
    [I 2025-10-17 17:14:13,384] Trial 16 finished with value: 0.006208648202061439 and parameters: {'sigma': 1.7299896001541488, 'lambd': 0.01379284749860457}. Best is trial 13 with value: 0.002876934337141066.
    [I 2025-10-17 17:14:13,757] Trial 17 finished with value: 0.008024009312408636 and parameters: {'sigma': 0.21725067636737497, 'lambd': 0.2687278008549822}. Best is trial 13 with value: 0.002876934337141066.
    [I 2025-10-17 17:14:14,106] Trial 18 finished with value: 0.0033401188386670104 and parameters: {'sigma': 9.661895902611315, 'lambd': 0.0042217753524644306}. Best is trial 13 with value: 0.002876934337141066.
    [I 2025-10-17 17:14:14,407] Trial 19 finished with value: 0.006188176782675978 and parameters: {'sigma': 1.628450682232627, 'lambd': 0.05001303664821358}. Best is trial 13 with value: 0.002876934337141066.


      Using RFF with 150 components
      Explained variance: 0.618
      Best MMD: 0.0029
      Current MMD: 0.0009
    
    PCA with 10 components:


    [I 2025-10-17 17:14:16,063] A new study created in memory with name: no-name-fc08bedc-8ba8-48b5-a38a-b87882b2fa71
    [I 2025-10-17 17:14:16,206] Trial 0 finished with value: 0.004597442403194464 and parameters: {'sigma': 0.23959507950146675, 'lambd': 0.0006931100400958058}. Best is trial 0 with value: 0.004597442403194464.
    [I 2025-10-17 17:14:16,360] Trial 1 finished with value: 0.004652955557853788 and parameters: {'sigma': 0.010329310029977284, 'lambd': 0.0002887026648741285}. Best is trial 0 with value: 0.004597442403194464.
    [I 2025-10-17 17:14:16,531] Trial 2 finished with value: 0.004438723315533959 and parameters: {'sigma': 2.212946010752523, 'lambd': 0.00016669801196031274}. Best is trial 2 with value: 0.004438723315533959.
    [I 2025-10-17 17:14:16,709] Trial 3 finished with value: 0.004378271170153425 and parameters: {'sigma': 0.268336989394594, 'lambd': 0.5441705173492666}. Best is trial 3 with value: 0.004378271170153425.
    [I 2025-10-17 17:14:16,914] Trial 4 finished with value: 0.004483456608362379 and parameters: {'sigma': 0.3123600562038798, 'lambd': 0.0046303341397805455}. Best is trial 3 with value: 0.004378271170153425.
    [I 2025-10-17 17:14:17,091] Trial 5 finished with value: 0.004254761414422766 and parameters: {'sigma': 2.52047145232276, 'lambd': 0.19476342506380415}. Best is trial 5 with value: 0.004254761414422766.
    [I 2025-10-17 17:14:17,269] Trial 6 finished with value: 0.004486804966114798 and parameters: {'sigma': 0.23901513612253464, 'lambd': 0.167456860816161}. Best is trial 5 with value: 0.004254761414422766.
    [I 2025-10-17 17:14:17,458] Trial 7 finished with value: 0.004603441313264692 and parameters: {'sigma': 0.08417006970026696, 'lambd': 3.239954531397513e-05}. Best is trial 5 with value: 0.004254761414422766.
    [I 2025-10-17 17:14:17,634] Trial 8 finished with value: 0.004659364957668848 and parameters: {'sigma': 0.2174689292316603, 'lambd': 0.0003949806077984321}. Best is trial 5 with value: 0.004254761414422766.
    [I 2025-10-17 17:14:17,852] Trial 9 finished with value: 0.004643056589199697 and parameters: {'sigma': 0.6271770651953724, 'lambd': 0.025213654037282863}. Best is trial 5 with value: 0.004254761414422766.
    [I 2025-10-17 17:14:18,005] Trial 10 finished with value: 0.0032252088116024235 and parameters: {'sigma': 9.462757567619906, 'lambd': 0.033666766574878264}. Best is trial 10 with value: 0.0032252088116024235.
    [I 2025-10-17 17:14:18,163] Trial 11 finished with value: 0.0032986095730336666 and parameters: {'sigma': 8.64000391983197, 'lambd': 0.03583283987925894}. Best is trial 10 with value: 0.0032252088116024235.
    [I 2025-10-17 17:14:18,421] Trial 12 finished with value: 0.0034537074116932783 and parameters: {'sigma': 8.9128629585144, 'lambd': 0.01201121087106977}. Best is trial 10 with value: 0.0032252088116024235.
    [I 2025-10-17 17:14:18,672] Trial 13 finished with value: 0.0032685534732815455 and parameters: {'sigma': 8.914381481379705, 'lambd': 0.05039857372290846}. Best is trial 10 with value: 0.0032252088116024235.
    [I 2025-10-17 17:14:18,811] Trial 14 finished with value: 0.004213604587147527 and parameters: {'sigma': 2.7478206095023796, 'lambd': 0.07117303526804912}. Best is trial 10 with value: 0.0032252088116024235.
    [I 2025-10-17 17:14:19,044] Trial 15 finished with value: 0.004587269732908544 and parameters: {'sigma': 1.2553293086331168, 'lambd': 0.0037601216362659626}. Best is trial 10 with value: 0.0032252088116024235.
    [I 2025-10-17 17:14:19,275] Trial 16 finished with value: 0.0031441487082925464 and parameters: {'sigma': 4.867057106103801, 'lambd': 0.6951790361819624}. Best is trial 16 with value: 0.0031441487082925464.
    [I 2025-10-17 17:14:19,516] Trial 17 finished with value: 0.004467220477651881 and parameters: {'sigma': 0.037311307038347834, 'lambd': 0.7945345215021298}. Best is trial 16 with value: 0.0031441487082925464.
    [I 2025-10-17 17:14:19,732] Trial 18 finished with value: 0.003298987179630824 and parameters: {'sigma': 4.670700610744778, 'lambd': 0.2314728912900029}. Best is trial 16 with value: 0.0031441487082925464.
    [I 2025-10-17 17:14:19,937] Trial 19 finished with value: 0.004671499210053675 and parameters: {'sigma': 0.9065174997335304, 'lambd': 0.010535679370338748}. Best is trial 16 with value: 0.0031441487082925464.


      Using RFF with 150 components
      Explained variance: 0.719
      Best MMD: 0.0031
      Current MMD: 0.0016
    
    PCA with 15 components:


    [I 2025-10-17 17:14:21,366] A new study created in memory with name: no-name-96b9463f-7856-4ada-bdf2-3d5cd31102c4
    [I 2025-10-17 17:14:21,599] Trial 0 finished with value: 0.003262283712436921 and parameters: {'sigma': 0.04194886858261683, 'lambd': 2.8996611541204417e-05}. Best is trial 0 with value: 0.003262283712436921.
    [I 2025-10-17 17:14:21,736] Trial 1 finished with value: 0.0030744225374605607 and parameters: {'sigma': 2.658143221420064, 'lambd': 0.5325799117944284}. Best is trial 1 with value: 0.0030744225374605607.
    [I 2025-10-17 17:14:21,908] Trial 2 finished with value: 0.0032388127299949484 and parameters: {'sigma': 0.39200164097099094, 'lambd': 0.01828599731003998}. Best is trial 1 with value: 0.0030744225374605607.
    [I 2025-10-17 17:14:22,143] Trial 3 finished with value: 0.0032649555411272714 and parameters: {'sigma': 0.023800645626366728, 'lambd': 0.00023898833968547144}. Best is trial 1 with value: 0.0030744225374605607.
    [I 2025-10-17 17:14:22,325] Trial 4 finished with value: 0.003212042290103858 and parameters: {'sigma': 5.136763400313113, 'lambd': 0.002437689503381915}. Best is trial 1 with value: 0.0030744225374605607.
    [I 2025-10-17 17:14:22,569] Trial 5 finished with value: 0.0032552951435582582 and parameters: {'sigma': 0.34073132962108593, 'lambd': 0.00017663937070822012}. Best is trial 1 with value: 0.0030744225374605607.
    [I 2025-10-17 17:14:22,810] Trial 6 finished with value: 0.003245886325487281 and parameters: {'sigma': 0.11593393537037143, 'lambd': 0.00010872705026073396}. Best is trial 1 with value: 0.0030744225374605607.
    [I 2025-10-17 17:14:22,966] Trial 7 finished with value: 0.003190789828918072 and parameters: {'sigma': 0.024969229956203234, 'lambd': 0.6662281348556616}. Best is trial 1 with value: 0.0030744225374605607.
    [I 2025-10-17 17:14:23,141] Trial 8 finished with value: 0.0032598378365531503 and parameters: {'sigma': 0.030528353713654595, 'lambd': 0.008800485666801049}. Best is trial 1 with value: 0.0030744225374605607.
    [I 2025-10-17 17:14:23,392] Trial 9 finished with value: 0.0032267242747699787 and parameters: {'sigma': 0.2730541663935448, 'lambd': 8.960336127490481e-05}. Best is trial 1 with value: 0.0030744225374605607.
    [I 2025-10-17 17:14:23,613] Trial 10 finished with value: 0.0026609638626353977 and parameters: {'sigma': 5.762702513486472, 'lambd': 0.8051733424863858}. Best is trial 10 with value: 0.0026609638626353977.
    [I 2025-10-17 17:14:23,767] Trial 11 finished with value: 0.002631854586463701 and parameters: {'sigma': 8.169357232510995, 'lambd': 0.757765469826795}. Best is trial 11 with value: 0.002631854586463701.
    [I 2025-10-17 17:14:24,010] Trial 12 finished with value: 0.0027780034826694914 and parameters: {'sigma': 8.918759312115615, 'lambd': 0.099292361578671}. Best is trial 11 with value: 0.002631854586463701.
    [I 2025-10-17 17:14:24,167] Trial 13 finished with value: 0.0032480629034076967 and parameters: {'sigma': 1.6285122961168212, 'lambd': 0.08550251898978017}. Best is trial 11 with value: 0.002631854586463701.
    [I 2025-10-17 17:14:24,386] Trial 14 finished with value: 0.0032871198235389174 and parameters: {'sigma': 1.2217759932990329, 'lambd': 0.13678777433347997}. Best is trial 11 with value: 0.002631854586463701.
    [I 2025-10-17 17:14:24,616] Trial 15 finished with value: 0.0027231142522288953 and parameters: {'sigma': 8.470417204455702, 'lambd': 0.816513255505266}. Best is trial 11 with value: 0.002631854586463701.
    [I 2025-10-17 17:14:24,840] Trial 16 finished with value: 0.0032417396015410028 and parameters: {'sigma': 2.807721040446678, 'lambd': 0.0013707845174766925}. Best is trial 11 with value: 0.002631854586463701.
    [I 2025-10-17 17:14:25,008] Trial 17 finished with value: 0.003269505077665167 and parameters: {'sigma': 0.7418091674875817, 'lambd': 0.020988565322234008}. Best is trial 11 with value: 0.002631854586463701.
    [I 2025-10-17 17:14:25,211] Trial 18 finished with value: 0.0030869373225207147 and parameters: {'sigma': 3.8070709578046102, 'lambd': 0.13848784102302195}. Best is trial 11 with value: 0.002631854586463701.
    [I 2025-10-17 17:14:25,386] Trial 19 finished with value: 0.003242618513917286 and parameters: {'sigma': 0.7894412554812702, 'lambd': 0.25333377310285465}. Best is trial 11 with value: 0.002631854586463701.


      Using RFF with 150 components
      Explained variance: 0.760
      Best MMD: 0.0026
      Current MMD: 0.0012
    
    PCA with 20 components:


    [I 2025-10-17 17:14:27,118] A new study created in memory with name: no-name-78a78fc0-b93d-4777-b90f-37c9117b3ea8
    [I 2025-10-17 17:14:27,473] Trial 0 finished with value: 0.0028507917571380425 and parameters: {'sigma': 0.01701389258172007, 'lambd': 0.754646569848943}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:27,785] Trial 1 finished with value: 0.0029048926954787866 and parameters: {'sigma': 0.3481559074036468, 'lambd': 0.001040095810717479}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:28,184] Trial 2 finished with value: 0.0028940520291561383 and parameters: {'sigma': 0.040012140597401005, 'lambd': 0.010981390966425301}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:28,460] Trial 3 finished with value: 0.0028975069844195344 and parameters: {'sigma': 0.013275452331229294, 'lambd': 0.00012939116404386925}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:28,778] Trial 4 finished with value: 0.0028991139681298415 and parameters: {'sigma': 0.013381214463375363, 'lambd': 0.0012355145417439757}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:29,031] Trial 5 finished with value: 0.0029042217564676205 and parameters: {'sigma': 1.596893729969738, 'lambd': 0.34927966926351545}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:29,384] Trial 6 finished with value: 0.0028898311322900798 and parameters: {'sigma': 0.05100658911843875, 'lambd': 0.012655006692945238}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:29,627] Trial 7 finished with value: 0.002906325512808648 and parameters: {'sigma': 2.1515525697051063, 'lambd': 5.805310343511277e-05}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:29,853] Trial 8 finished with value: 0.002861021381940424 and parameters: {'sigma': 0.5631149033592096, 'lambd': 0.928548282279497}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:30,082] Trial 9 finished with value: 0.0029012119351557466 and parameters: {'sigma': 0.11327482404508064, 'lambd': 0.03724589041036686}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:30,332] Trial 10 finished with value: 0.00289831007516994 and parameters: {'sigma': 0.13346263679148615, 'lambd': 0.14306901222507976}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:30,569] Trial 11 finished with value: 0.002880034958577477 and parameters: {'sigma': 0.61689835874623, 'lambd': 0.8756050986563503}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:30,827] Trial 12 finished with value: 0.002903262909530305 and parameters: {'sigma': 0.9076691767726783, 'lambd': 0.15870515907398186}. Best is trial 0 with value: 0.0028507917571380425.
    [I 2025-10-17 17:14:31,066] Trial 13 finished with value: 0.0026422984355517314 and parameters: {'sigma': 9.69010864712242, 'lambd': 0.9513408894117135}. Best is trial 13 with value: 0.0026422984355517314.
    [I 2025-10-17 17:14:31,269] Trial 14 finished with value: 0.0029033401630111054 and parameters: {'sigma': 0.14577631933105503, 'lambd': 0.07282894400266589}. Best is trial 13 with value: 0.0026422984355517314.
    [I 2025-10-17 17:14:31,455] Trial 15 finished with value: 0.0028656799956336522 and parameters: {'sigma': 4.27714315099672, 'lambd': 0.020888406413039337}. Best is trial 13 with value: 0.0026422984355517314.
    [I 2025-10-17 17:14:31,615] Trial 16 finished with value: 0.0028932148688457105 and parameters: {'sigma': 8.464274193702044, 'lambd': 1.709117145572373e-05}. Best is trial 13 with value: 0.0026422984355517314.
    [I 2025-10-17 17:14:31,876] Trial 17 finished with value: 0.002845913520536925 and parameters: {'sigma': 8.844837434317297, 'lambd': 0.003391826517101554}. Best is trial 13 with value: 0.0026422984355517314.
    [I 2025-10-17 17:14:32,080] Trial 18 finished with value: 0.002873564744579124 and parameters: {'sigma': 5.576485608488262, 'lambd': 0.0022365463091387964}. Best is trial 13 with value: 0.0026422984355517314.
    [I 2025-10-17 17:14:32,302] Trial 19 finished with value: 0.0028987436288253976 and parameters: {'sigma': 2.929312329284539, 'lambd': 0.00022810309912462944}. Best is trial 13 with value: 0.0026422984355517314.


      Using RFF with 150 components
      Explained variance: 0.786
      Best MMD: 0.0026
      Current MMD: 0.0012
    
    PCA Dimension Comparison:
    --------------------------------------------------
    Components   Explained Var  Best MMD   Current MMD  Energy Dist 
    --------------------------------------------------
    5            0.618          0.0029     0.0009       0.0062      
    10           0.719          0.0031     0.0016       0.0122      
    15           0.760          0.0026     0.0012       0.0107      
    20           0.786          0.0026     0.0012       0.0077      
    
    ============================================================
    ALL EXAMPLES COMPLETED SUCCESSFULLY!
    ============================================================




  <div id="df-ef5661b4-0a38-4228-8432-a3878fa74887" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AC.PA.Close</th>
      <th>AI.PA.Close</th>
      <th>AIR.PA.Close</th>
      <th>ATO.PA.Close</th>
      <th>BNP.PA.Close</th>
      <th>CAP.PA.Close</th>
      <th>CS.PA.Close</th>
      <th>ENGI.PA.Close</th>
      <th>GLE.PA.Close</th>
      <th>KER.PA.Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-04-24</th>
      <td>0.028883</td>
      <td>-0.023468</td>
      <td>-0.003077</td>
      <td>0.026478</td>
      <td>-0.004714</td>
      <td>0.005854</td>
      <td>-0.019373</td>
      <td>0.004361</td>
      <td>-0.004342</td>
      <td>-0.071147</td>
    </tr>
    <tr>
      <th>2024-04-25</th>
      <td>-0.010734</td>
      <td>-0.013631</td>
      <td>-0.023073</td>
      <td>-0.051293</td>
      <td>0.009259</td>
      <td>-0.022130</td>
      <td>-0.013226</td>
      <td>-0.000311</td>
      <td>-0.007345</td>
      <td>0.003520</td>
    </tr>
    <tr>
      <th>2024-04-26</th>
      <td>0.013813</td>
      <td>0.008460</td>
      <td>-0.009253</td>
      <td>0.009214</td>
      <td>-0.019946</td>
      <td>0.009896</td>
      <td>0.001478</td>
      <td>0.001243</td>
      <td>0.009518</td>
      <td>0.032021</td>
    </tr>
    <tr>
      <th>2024-04-29</th>
      <td>-0.011896</td>
      <td>-0.004221</td>
      <td>-0.007413</td>
      <td>0.175485</td>
      <td>0.010540</td>
      <td>0.011748</td>
      <td>0.009994</td>
      <td>0.009274</td>
      <td>0.009819</td>
      <td>-0.006829</td>
    </tr>
    <tr>
      <th>2024-04-30</th>
      <td>-0.011313</td>
      <td>-0.002824</td>
      <td>-0.007856</td>
      <td>-0.116465</td>
      <td>-0.001626</td>
      <td>-0.037684</td>
      <td>-0.052541</td>
      <td>0.001230</td>
      <td>-0.007651</td>
      <td>-0.016976</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ef5661b4-0a38-4228-8432-a3878fa74887')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ef5661b4-0a38-4228-8432-a3878fa74887 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ef5661b4-0a38-4228-8432-a3878fa74887');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-bbcee3d8-404c-4383-83e1-ab076c8c9bde">
      <button class="colab-df-quickchart" onclick="quickchart('df-bbcee3d8-404c-4383-83e1-ab076c8c9bde')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-bbcee3d8-404c-4383-83e1-ab076c8c9bde button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





  <div id="df-ec770696-7ff4-41a0-9030-6c5cf5d87808" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AC.PA.Close</th>
      <th>AI.PA.Close</th>
      <th>AIR.PA.Close</th>
      <th>ATO.PA.Close</th>
      <th>BNP.PA.Close</th>
      <th>CAP.PA.Close</th>
      <th>CS.PA.Close</th>
      <th>ENGI.PA.Close</th>
      <th>GLE.PA.Close</th>
      <th>KER.PA.Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2025-08-20</th>
      <td>-0.011993</td>
      <td>-0.006674</td>
      <td>0.002460</td>
      <td>-0.019722</td>
      <td>-0.000911</td>
      <td>-0.021182</td>
      <td>-0.002761</td>
      <td>-0.006199</td>
      <td>0.007999</td>
      <td>-0.025209</td>
    </tr>
    <tr>
      <th>2025-08-21</th>
      <td>-0.009031</td>
      <td>0.000454</td>
      <td>0.016286</td>
      <td>-0.053609</td>
      <td>0.002859</td>
      <td>-0.008683</td>
      <td>-0.000251</td>
      <td>-0.004249</td>
      <td>0.009063</td>
      <td>0.002833</td>
    </tr>
    <tr>
      <th>2025-08-22</th>
      <td>-0.021231</td>
      <td>-0.007973</td>
      <td>-0.015727</td>
      <td>-0.033652</td>
      <td>-0.014511</td>
      <td>-0.019288</td>
      <td>-0.014687</td>
      <td>-0.014870</td>
      <td>-0.013245</td>
      <td>0.037579</td>
    </tr>
    <tr>
      <th>2025-08-25</th>
      <td>-0.003909</td>
      <td>0.004791</td>
      <td>0.030454</td>
      <td>-0.062625</td>
      <td>0.003287</td>
      <td>0.030027</td>
      <td>0.003311</td>
      <td>0.012312</td>
      <td>0.006077</td>
      <td>-0.009052</td>
    </tr>
    <tr>
      <th>2025-08-26</th>
      <td>0.004884</td>
      <td>0.002614</td>
      <td>0.011627</td>
      <td>0.053213</td>
      <td>0.017822</td>
      <td>0.003691</td>
      <td>0.012131</td>
      <td>0.003976</td>
      <td>0.023575</td>
      <td>-0.016202</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ec770696-7ff4-41a0-9030-6c5cf5d87808')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ec770696-7ff4-41a0-9030-6c5cf5d87808 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ec770696-7ff4-41a0-9030-6c5cf5d87808');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-60e72bdb-9bbb-409e-93a4-321814467876">
      <button class="colab-df-quickchart" onclick="quickchart('df-60e72bdb-9bbb-409e-93a4-321814467876')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-60e72bdb-9bbb-409e-93a4-321814467876 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



    [I 2025-10-17 17:14:33,260] A new study created in memory with name: no-name-d37f9ab0-e289-4822-895f-385925469b7e
    [I 2025-10-17 17:14:33,294] Trial 0 finished with value: 0.0007943400151759761 and parameters: {'sigma': 5.625779360906917, 'lambd': 0.0131744015527339}. Best is trial 0 with value: 0.0007943400151759761.
    [I 2025-10-17 17:14:33,329] Trial 1 finished with value: 0.0027987220636565002 and parameters: {'sigma': 0.7161417827371748, 'lambd': 8.644655305256373e-05}. Best is trial 0 with value: 0.0007943400151759761.
    [I 2025-10-17 17:14:33,363] Trial 2 finished with value: 0.0013932879284475064 and parameters: {'sigma': 0.4555931270421259, 'lambd': 9.340513011515643e-05}. Best is trial 0 with value: 0.0007943400151759761.
    [I 2025-10-17 17:14:33,398] Trial 3 finished with value: 0.0020673461587394915 and parameters: {'sigma': 5.64540023244463, 'lambd': 7.334907642778781e-05}. Best is trial 0 with value: 0.0007943400151759761.
    [I 2025-10-17 17:14:33,431] Trial 4 finished with value: 0.0021189663398144543 and parameters: {'sigma': 1.233043503059137, 'lambd': 0.11812225321262308}. Best is trial 0 with value: 0.0007943400151759761.
    [I 2025-10-17 17:14:33,469] Trial 5 finished with value: 0.0036006030161266356 and parameters: {'sigma': 0.0446876580552351, 'lambd': 0.014736710283921805}. Best is trial 0 with value: 0.0007943400151759761.
    [I 2025-10-17 17:14:33,506] Trial 6 finished with value: 0.0027647592501125473 and parameters: {'sigma': 1.254798196940215, 'lambd': 0.0007680149945928696}. Best is trial 0 with value: 0.0007943400151759761.
    [I 2025-10-17 17:14:33,543] Trial 7 finished with value: 0.003968947465641026 and parameters: {'sigma': 0.5008369636523282, 'lambd': 0.0010256434648317635}. Best is trial 0 with value: 0.0007943400151759761.
    [I 2025-10-17 17:14:33,584] Trial 8 finished with value: 0.002038373063199206 and parameters: {'sigma': 0.044855098704406265, 'lambd': 7.241927826511569e-05}. Best is trial 0 with value: 0.0007943400151759761.
    [I 2025-10-17 17:14:33,622] Trial 9 finished with value: 0.0017935939915683097 and parameters: {'sigma': 2.0154327653679247, 'lambd': 0.0003715739836055347}. Best is trial 0 with value: 0.0007943400151759761.
    [I 2025-10-17 17:14:33,647] Trial 10 finished with value: 0.0006797890226231118 and parameters: {'sigma': 8.144102791916099, 'lambd': 0.5282235280573779}. Best is trial 10 with value: 0.0006797890226231118.
    [I 2025-10-17 17:14:33,672] Trial 11 finished with value: 0.00026594387919520734 and parameters: {'sigma': 9.356466445560258, 'lambd': 0.43103189078543414}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:33,694] Trial 12 finished with value: 0.0007044341871582649 and parameters: {'sigma': 8.87294381873665, 'lambd': 0.6811668708461237}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:33,722] Trial 13 finished with value: 0.0007416978185643686 and parameters: {'sigma': 0.014743516696234237, 'lambd': 0.9434721198892767}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:33,779] Trial 14 finished with value: 0.001864877152997657 and parameters: {'sigma': 2.941469825948283, 'lambd': 0.10762976122047574}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:33,821] Trial 15 finished with value: 0.003182176975237372 and parameters: {'sigma': 0.16111960468586803, 'lambd': 0.11807961736592867}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:33,846] Trial 16 finished with value: 0.0021474959328093846 and parameters: {'sigma': 9.954135386873839, 'lambd': 1.2676794558336279e-05}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:33,868] Trial 17 finished with value: 0.0024022754105550437 and parameters: {'sigma': 0.15973867307373435, 'lambd': 0.021329041365881582}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:33,914] Trial 18 finished with value: 0.0007947120267768337 and parameters: {'sigma': 3.327448691197009, 'lambd': 0.30570223410128056}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:33,949] Trial 19 finished with value: 0.0017886755983256286 and parameters: {'sigma': 4.1301230617510685, 'lambd': 0.04390126032480344}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:33,987] Trial 20 finished with value: 0.003474285817205436 and parameters: {'sigma': 1.3324133149688133, 'lambd': 0.00619771708779163}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:34,014] Trial 21 finished with value: 0.0005130545452709523 and parameters: {'sigma': 9.875383834339798, 'lambd': 0.727751285700473}. Best is trial 11 with value: 0.00026594387919520734.
    [I 2025-10-17 17:14:34,046] Trial 22 finished with value: 0.00020958678078697446 and parameters: {'sigma': 9.294395566333224, 'lambd': 0.3488628555837875}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,072] Trial 23 finished with value: 0.0011136555294379846 and parameters: {'sigma': 3.169481450868823, 'lambd': 0.2672943210184916}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,099] Trial 24 finished with value: 0.0012802216956599999 and parameters: {'sigma': 5.771465918932082, 'lambd': 0.24634610180575106}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,124] Trial 25 finished with value: 0.0018012690951030308 and parameters: {'sigma': 2.0616214192182576, 'lambd': 0.06839475122875349}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,151] Trial 26 finished with value: 0.000700089366824308 and parameters: {'sigma': 9.143424193954292, 'lambd': 0.37046700862133985}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,179] Trial 27 finished with value: 0.0011682158814763088 and parameters: {'sigma': 0.20875190377905847, 'lambd': 0.9833796031870473}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,206] Trial 28 finished with value: 0.0020034144219209793 and parameters: {'sigma': 5.000320119099681, 'lambd': 0.03306131483114492}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,237] Trial 29 finished with value: 0.0017010871411755613 and parameters: {'sigma': 5.9376388706747845, 'lambd': 0.008946998417558423}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,278] Trial 30 finished with value: 0.0018079320276713773 and parameters: {'sigma': 2.220963542002852, 'lambd': 0.003476403669251675}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,308] Trial 31 finished with value: 0.0005260759296796813 and parameters: {'sigma': 9.894906802186128, 'lambd': 0.3209017968530076}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,332] Trial 32 finished with value: 0.000442124073658956 and parameters: {'sigma': 6.517102913879885, 'lambd': 0.16759356159663055}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,359] Trial 33 finished with value: 0.0005617168847249498 and parameters: {'sigma': 5.831921436221848, 'lambd': 0.16388594015967792}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,388] Trial 34 finished with value: 0.0010410004685130758 and parameters: {'sigma': 4.267361966935676, 'lambd': 0.0717990111487866}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,423] Trial 35 finished with value: 0.0015249838606530286 and parameters: {'sigma': 0.743404589070603, 'lambd': 0.1718073360438499}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,457] Trial 36 finished with value: 0.000991324183079012 and parameters: {'sigma': 6.613062306658863, 'lambd': 0.5370278093766302}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,492] Trial 37 finished with value: 0.002015164956569171 and parameters: {'sigma': 3.918875704033748, 'lambd': 0.03966137831592432}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,520] Trial 38 finished with value: 0.0017635295350848512 and parameters: {'sigma': 2.6768442020515435, 'lambd': 0.07738131344642103}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,542] Trial 39 finished with value: 0.0009168877164063005 and parameters: {'sigma': 1.6253181121871743, 'lambd': 0.57049543742156}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,567] Trial 40 finished with value: 0.0023658872647471796 and parameters: {'sigma': 0.8965462221956919, 'lambd': 0.16617167432405108}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,602] Trial 41 finished with value: 0.0005561293378784082 and parameters: {'sigma': 9.909178880944706, 'lambd': 0.3678123002098977}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,647] Trial 42 finished with value: 0.0003001045696182647 and parameters: {'sigma': 6.6681661490512285, 'lambd': 0.24298534437030353}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,704] Trial 43 finished with value: 0.0006213847035683173 and parameters: {'sigma': 6.199460964030569, 'lambd': 0.19375066548511452}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,740] Trial 44 finished with value: 0.0007069991328423342 and parameters: {'sigma': 6.53712779704426, 'lambd': 0.9708892987037245}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,766] Trial 45 finished with value: 0.00327840922837197 and parameters: {'sigma': 0.0763304679999898, 'lambd': 0.5131482700172046}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,796] Trial 46 finished with value: 0.0016639185372400167 and parameters: {'sigma': 0.38672046269397425, 'lambd': 0.09006819632033962}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,840] Trial 47 finished with value: 0.001809793576592611 and parameters: {'sigma': 4.366003373192074, 'lambd': 0.0016215965537156294}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,898] Trial 48 finished with value: 0.0006248699887387854 and parameters: {'sigma': 7.453742811814838, 'lambd': 0.051375779966374185}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:34,956] Trial 49 finished with value: 0.0022116703936509996 and parameters: {'sigma': 0.015049254779191673, 'lambd': 0.023597922252300382}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:35,009] Trial 50 finished with value: 0.0034033147925425222 and parameters: {'sigma': 2.428834761015442, 'lambd': 0.00027764138945784365}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:35,072] Trial 51 finished with value: 0.000656937500455701 and parameters: {'sigma': 7.768734519721114, 'lambd': 0.4041697458893845}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:35,127] Trial 52 finished with value: 0.0007408603494183374 and parameters: {'sigma': 9.470112942957746, 'lambd': 0.6961635796254613}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:35,185] Trial 53 finished with value: 0.0006000158470103045 and parameters: {'sigma': 3.6111782152498817, 'lambd': 0.25500472154410875}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:35,239] Trial 54 finished with value: 0.0003780449475951908 and parameters: {'sigma': 4.802231291345405, 'lambd': 0.11350566738117857}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:35,293] Trial 55 finished with value: 0.0005624337991014805 and parameters: {'sigma': 4.74878570370497, 'lambd': 0.13537290552169404}. Best is trial 22 with value: 0.00020958678078697446.
    [I 2025-10-17 17:14:35,347] Trial 56 finished with value: 0.0001980350426795674 and parameters: {'sigma': 7.30540702408865, 'lambd': 0.1176205750643718}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,399] Trial 57 finished with value: 0.0017771006707709702 and parameters: {'sigma': 7.510239189539025, 'lambd': 0.014201631316041985}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,453] Trial 58 finished with value: 0.0011793788655372062 and parameters: {'sigma': 3.023367256690396, 'lambd': 0.10276267457787497}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,509] Trial 59 finished with value: 0.001129901717932702 and parameters: {'sigma': 1.661148766365103, 'lambd': 0.026374914117341327}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,561] Trial 60 finished with value: 0.0011080229394437868 and parameters: {'sigma': 5.568306061426028, 'lambd': 0.05742862884204015}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,616] Trial 61 finished with value: 0.0005128538363763369 and parameters: {'sigma': 7.909095005977163, 'lambd': 0.22898802947953062}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,677] Trial 62 finished with value: 0.0009364884297076159 and parameters: {'sigma': 7.491501519339944, 'lambd': 0.2171041478139598}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,727] Trial 63 finished with value: 0.0006078810193519413 and parameters: {'sigma': 4.8781005371426405, 'lambd': 0.40542345523827006}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,781] Trial 64 finished with value: 0.0010032574099840819 and parameters: {'sigma': 3.53587556438744, 'lambd': 0.11317945429956296}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,831] Trial 65 finished with value: 0.0013528120441843594 and parameters: {'sigma': 7.916320264468055, 'lambd': 4.411420843695402e-05}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,887] Trial 66 finished with value: 0.0008812232655741603 and parameters: {'sigma': 5.114259148578681, 'lambd': 0.22815809103599266}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:35,942] Trial 67 finished with value: 0.0004683798727702637 and parameters: {'sigma': 6.76932238249006, 'lambd': 0.1456603378273577}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:36,001] Trial 68 finished with value: 0.0015802841635983444 and parameters: {'sigma': 3.8512510769564363, 'lambd': 0.14297460219368116}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:36,060] Trial 69 finished with value: 0.002129518671783126 and parameters: {'sigma': 2.7210492185944744, 'lambd': 0.03538814859390222}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:36,119] Trial 70 finished with value: 0.0007796333619327633 and parameters: {'sigma': 6.281394384082659, 'lambd': 0.08005430035572933}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:36,174] Trial 71 finished with value: 0.0018047485645007466 and parameters: {'sigma': 7.704475109216896, 'lambd': 0.2988865601598843}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:36,239] Trial 72 finished with value: 0.00024266149857243846 and parameters: {'sigma': 4.88270831307013, 'lambd': 0.47429903045247124}. Best is trial 56 with value: 0.0001980350426795674.
    [I 2025-10-17 17:14:36,275] Trial 73 finished with value: 0.00011409673923967745 and parameters: {'sigma': 4.837745189193317, 'lambd': 0.6775383372602185}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,330] Trial 74 finished with value: 0.0005037726949221888 and parameters: {'sigma': 4.453040443853297, 'lambd': 0.7738274167238229}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,400] Trial 75 finished with value: 0.000991258750055346 and parameters: {'sigma': 5.320238074405867, 'lambd': 0.6075360503901304}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,457] Trial 76 finished with value: 0.0009691436440586454 and parameters: {'sigma': 1.8861921282714436, 'lambd': 0.4280927277783616}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,512] Trial 77 finished with value: 0.0005551761919810883 and parameters: {'sigma': 3.7257876033022606, 'lambd': 0.31528217016459364}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,558] Trial 78 finished with value: 0.0010219034260874338 and parameters: {'sigma': 3.144156874070921, 'lambd': 0.8097033488072441}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,609] Trial 79 finished with value: 0.0019405639987730705 and parameters: {'sigma': 1.1224444747064137, 'lambd': 0.5282553807335726}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,657] Trial 80 finished with value: 0.004197175649950591 and parameters: {'sigma': 0.07578792489830381, 'lambd': 0.17950577627158576}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,705] Trial 81 finished with value: 0.00040697847517368047 and parameters: {'sigma': 6.049642564493944, 'lambd': 0.1391689029105095}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,736] Trial 82 finished with value: 0.0002516208478098303 and parameters: {'sigma': 8.765581889257906, 'lambd': 0.29598232251054285}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,761] Trial 83 finished with value: 0.0010171069768680763 and parameters: {'sigma': 9.400667782038585, 'lambd': 0.31090804848866666}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,801] Trial 84 finished with value: 0.0006423890659776887 and parameters: {'sigma': 5.069109040418955, 'lambd': 0.47803275514460886}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,866] Trial 85 finished with value: 0.00032626857972584133 and parameters: {'sigma': 8.773685603723248, 'lambd': 0.9736936506267027}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,920] Trial 86 finished with value: 0.00022383024125338657 and parameters: {'sigma': 8.58486362903349, 'lambd': 0.9649315596108536}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,959] Trial 87 finished with value: 0.0005726078297922932 and parameters: {'sigma': 8.197713800231636, 'lambd': 0.8575849062247036}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:36,986] Trial 88 finished with value: 0.00039194053528457395 and parameters: {'sigma': 9.779854609540456, 'lambd': 0.6502597991572513}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,013] Trial 89 finished with value: 0.0007796001210169656 and parameters: {'sigma': 6.850199669948823, 'lambd': 0.4270459061543165}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,066] Trial 90 finished with value: 0.0008878254510173988 and parameters: {'sigma': 5.839759753152064, 'lambd': 0.978019234773362}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,140] Trial 91 finished with value: 0.0005135102415951831 and parameters: {'sigma': 8.733044455804215, 'lambd': 0.6417137384528625}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,170] Trial 92 finished with value: 0.0002379975245896393 and parameters: {'sigma': 4.226143667154324, 'lambd': 0.3636979360220931}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,190] Trial 93 finished with value: 0.0007803907005912158 and parameters: {'sigma': 4.1236930851286875, 'lambd': 0.3060197111218387}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,212] Trial 94 finished with value: 0.0001464904667827227 and parameters: {'sigma': 8.534225255359505, 'lambd': 0.4727968794248274}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,235] Trial 95 finished with value: 0.0003836771045013787 and parameters: {'sigma': 6.833505117546917, 'lambd': 0.37608788182699243}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,266] Trial 96 finished with value: 0.0028722293894809514 and parameters: {'sigma': 0.01800739842107343, 'lambd': 0.5548671172540407}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,320] Trial 97 finished with value: 0.0013401362756435553 and parameters: {'sigma': 0.23975120921678886, 'lambd': 0.2719137533919699}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,372] Trial 98 finished with value: 0.00021530273537373468 and parameters: {'sigma': 5.586687613009044, 'lambd': 0.20441657749138106}. Best is trial 73 with value: 0.00011409673923967745.
    [I 2025-10-17 17:14:37,422] Trial 99 finished with value: 0.0003498357100084615 and parameters: {'sigma': 5.315870065247185, 'lambd': 0.46267392690947157}. Best is trial 73 with value: 0.00011409673923967745.


      Using RFF with 50 components
    Bivariate Results:
    Best sigma: 4.838, lambda: 0.678, MMD: 0.000


    /usr/local/lib/python3.12/dist-packages/synthe/distro_simulator.py:840: UserWarning: p-value capped: true value larger than 0.25. Consider specifying `method` (e.g. `method=stats.PermutationMethod()`.)
      ad_result = stats.anderson_ksamp([Y_orig[:, i], Y_sim[:, i]])
    /usr/local/lib/python3.12/dist-packages/synthe/distro_simulator.py:840: UserWarning: p-value floored: true value smaller than 0.001. Consider specifying `method` (e.g. `method=stats.PermutationMethod()`.)
      ad_result = stats.anderson_ksamp([Y_orig[:, i], Y_sim[:, i]])



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_75.png){:class="img-responsive"}
    


    
    ============================================================
    COMPREHENSIVE STATISTICAL TEST RESULTS
    ============================================================
    
    Dimension 1:
      Kolmogorov-Smirnov Test:
        Statistic: 0.071429
        p-value: 0.334143
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: -0.517025
        Significance level: 0.250
        Interpretation: Distributions similar
    
    Dimension 2:
      Kolmogorov-Smirnov Test:
        Statistic: 0.057143
        p-value: 0.617909
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: -0.681447
        Significance level: 0.250
        Interpretation: Distributions similar
    
    Dimension 3:
      Kolmogorov-Smirnov Test:
        Statistic: 0.051429
        p-value: 0.744448
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: -0.166155
        Significance level: 0.250
        Interpretation: Distributions similar
    
    Dimension 4:
      Kolmogorov-Smirnov Test:
        Statistic: 0.182857
        p-value: 0.000016
        Significance: SIGNIFICANT
      Anderson-Darling Test:
        Statistic: 15.807250
        Significance level: 0.001
        Interpretation: Distributions differ
    
    Dimension 5:
      Kolmogorov-Smirnov Test:
        Statistic: 0.051429
        p-value: 0.744448
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: -0.888281
        Significance level: 0.250
        Interpretation: Distributions similar
    
    Dimension 6:
      Kolmogorov-Smirnov Test:
        Statistic: 0.071429
        p-value: 0.334143
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: 0.115670
        Significance level: 0.250
        Interpretation: Distributions similar
    
    Dimension 7:
      Kolmogorov-Smirnov Test:
        Statistic: 0.057143
        p-value: 0.617909
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: -0.406425
        Significance level: 0.250
        Interpretation: Distributions similar
    
    Dimension 8:
      Kolmogorov-Smirnov Test:
        Statistic: 0.074286
        p-value: 0.289300
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: 0.644059
        Significance level: 0.179
        Interpretation: Distributions similar
    
    Dimension 9:
      Kolmogorov-Smirnov Test:
        Statistic: 0.054286
        p-value: 0.681633
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: -0.512941
        Significance level: 0.250
        Interpretation: Distributions similar
    
    Dimension 10:
      Kolmogorov-Smirnov Test:
        Statistic: 0.071429
        p-value: 0.334143
        Significance: Not Significant
      Anderson-Darling Test:
        Statistic: 0.105073
        Significance level: 0.250
        Interpretation: Distributions similar



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_77.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2025-10-17/2025-10-17-P-Y-GAN-like_3_78.png){:class="img-responsive"}
    





    {'ks_results': [(np.float64(0.07142857142857142),
       np.float64(0.33414277179281887)),
      (np.float64(0.05714285714285714), np.float64(0.6179086766712222)),
      (np.float64(0.05142857142857143), np.float64(0.7444477161091746)),
      (np.float64(0.18285714285714286), np.float64(1.5747515517642453e-05)),
      (np.float64(0.05142857142857143), np.float64(0.7444477161091746)),
      (np.float64(0.07142857142857142), np.float64(0.33414277179281887)),
      (np.float64(0.05714285714285714), np.float64(0.6179086766712222)),
      (np.float64(0.07428571428571429), np.float64(0.2893000501430068)),
      (np.float64(0.054285714285714284), np.float64(0.6816329570848737)),
      (np.float64(0.07142857142857142), np.float64(0.33414277179281887))],
     'ad_results': [(np.float64(-0.5170247542790974), np.float64(0.25)),
      (np.float64(-0.6814466337397485), np.float64(0.25)),
      (np.float64(-0.16615496895697046), np.float64(0.25)),
      (np.float64(15.807250144266789), np.float64(0.001)),
      (np.float64(-0.8882811421178678), np.float64(0.25)),
      (np.float64(0.11566983431582328), np.float64(0.25)),
      (np.float64(-0.4064249682682882), np.float64(0.25)),
      (np.float64(0.6440585815762728), 0.17914164508840966),
      (np.float64(-0.5129406111560781), np.float64(0.25)),
      (np.float64(0.10507310724372729), np.float64(0.25))],
     'dimensions': 10}


<a target="_blank" href="https://colab.research.google.com/github/Techtonique/synthe/blob/main/examples/2025_10_17_P_Y_GAN_like.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
