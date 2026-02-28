# MNIST Exploratory Data Analysis

## Overview

This notebook performs a comprehensive exploratory data analysis (EDA) of the MNIST handwritten digit dataset. The goal is to deeply understand the structure, distribution, and characteristics of the data *before* any modeling work begins. Good EDA answers questions like:

- How balanced are the classes? Are some digits over/under-represented?
- What does the pixel intensity landscape look like? How sparse are the images?
- Are there natural clusters in the data? How separable are the digit classes?
- Which principal components capture the most variance? What do "eigendigits" look like?
- Are there outliers or atypical samples that could hurt training?

These insights directly inform modeling decisions: architecture choice, preprocessing, augmentation, and loss function design.

## Dataset Background

**MNIST** (Modified National Institute of Standards and Technology) is a dataset of 70,000 grayscale handwritten digit images (60,000 train / 10,000 test), each 28x28 pixels with values in [0, 255]. It was introduced by LeCun et al. (1998) as a benchmark for handwritten digit recognition and has since become the canonical "hello world" dataset for image classification.

Each image is a single centered digit (0-9) written by Census Bureau employees and high school students. The images were size-normalized and centered in the 28x28 frame from the original NIST dataset.

**Key properties:**
- 10 classes (digits 0-9), roughly balanced
- Single-channel grayscale, 28x28 pixels = 784 features when flattened
- High sparsity (most pixels are background/zero)
- Relatively "easy" by modern standards (~99.7% achievable), but still useful for rapid prototyping

## EDA Techniques: Image Data

The notebook applies the following techniques, each chosen because it reveals something specific about image data that generic tabular EDA would miss.

### 1. Sample Visualization Grid

**What:** Display a grid of random samples organized by class (10 rows x 10 columns).

**Why:** The single most important EDA step for image data. Directly viewing samples reveals writing style variation, image quality, centering issues, and potential labeling errors. This is something you *cannot* do with tabular summary statistics alone.

**Reference:** Standard practice in computer vision; see any introductory ML textbook or course (e.g., Stanford CS231n).

### 2. Class Distribution Analysis

**What:** Bar charts showing sample counts per digit for both train and test splits, with min/max ratio to quantify imbalance.

**Why:** Class imbalance directly affects model training. Even small imbalances (e.g., 2:1 ratio) can bias gradient-based optimizers toward majority classes. MNIST is roughly balanced, but verifying this is essential before assuming uniform priors.

**Reference:** He & Garcia (2009), "Learning from Imbalanced Data," *IEEE TKDE*.

### 3. Pixel Intensity Analysis

**What:** Overall histogram of pixel values across the dataset, per-class KDE overlays, and sparsity computation (% of zero-valued pixels).

**Why:** Pixel intensity distributions reveal the input's statistical profile. MNIST images are highly sparse (mostly black background), which matters for:
- Normalization strategy (mean/std vs. min/max)
- Activation function choice (ReLU naturally handles sparse inputs)
- Weight initialization (sparse inputs → different effective fan-in)

Per-class KDE differences show whether some digits use more "ink" than others (e.g., 1 vs. 8).

**Reference:** Tukey (1977), *Exploratory Data Analysis*, Addison-Wesley. Foundational work establishing EDA as a discipline.

### 4. Mean and Standard Deviation Images

**What:** Per-class and global mean/std images computed by averaging pixel values across all samples of each class.

**Why:** Mean images reveal the "prototype" or average shape of each digit. Std images reveal *where* the variation happens -- high std regions show where writing styles diverge most (e.g., the loop of a 2, the angle of a 7). This informs:
- Data augmentation strategy (augment high-variance regions)
- Whether mean subtraction is useful as preprocessing
- Understanding inter-class vs. intra-class variation

**Reference:** Standard technique; see Goodfellow et al. (2016), *Deep Learning*, Chapter 12 (Methodology).

### 5. Pixel Correlation Analysis

**What:** Correlation matrix of pixel intensities across the 784-dimensional flattened representation, displayed as a heatmap.

**Why:** Pixel correlations reveal spatial structure. Neighboring pixels are highly correlated (images are spatially smooth), which is:
- Why CNNs work (they exploit local spatial correlations)
- Why PCA is effective (correlated features → compressible)
- Why treating pixels as independent features (as in naive Bayes) loses information

The heatmap is subsampled (every 4th pixel) for computational tractability.

**Reference:** Related to the spatial correlation assumptions underlying convolutional architectures. See LeCun et al. (1998).

### 6. PCA Analysis

**What:** Fit Principal Component Analysis on flattened training images, plot cumulative explained variance, and visualize the top principal components as 28x28 "eigendigit" images.

**Why:** PCA reveals the intrinsic dimensionality of the data. Key findings:
- How many components are needed for 90/95/99% variance (typically ~50/~90/~330 for MNIST)
- Eigendigits show the dominant modes of variation (stroke angle, thickness, curvature)
- Justifies dimensionality reduction as a preprocessing step for methods like t-SNE

**Reference:** Pearson (1901), "On Lines and Planes of Closest Fit"; Jolliffe (2002), *Principal Component Analysis*, Springer. For PCA on MNIST specifically, see scikit-learn documentation examples.

### 7. t-SNE Projection

**What:** Apply t-Distributed Stochastic Neighbor Embedding to project data from 784 dimensions to 2D, with PCA to 50 dims as a standard speedup preprocessing step. Colored scatter plot by digit label.

**Why:** t-SNE preserves local neighborhood structure, making it excellent for visualizing cluster quality. Well-separated clusters in t-SNE suggest that a classifier should be able to distinguish classes. Overlapping regions indicate classes that are inherently confusable (e.g., 4 vs. 9, 3 vs. 5).

**Important caveats:**
- t-SNE is non-parametric and stochastic; different runs yield different layouts
- Distances between clusters are *not* meaningful (only within-cluster structure is)
- Perplexity parameter affects the apparent cluster sizes

**Reference:** van der Maaten & Hinton (2008), "Visualizing Data using t-SNE," *JMLR* 9:2579-2605.

### 8. UMAP Projection

**What:** Apply Uniform Manifold Approximation and Projection to 2D on the same subset, with side-by-side comparison to t-SNE.

**Why:** UMAP is a more recent alternative to t-SNE that better preserves global structure (inter-cluster distances are more meaningful) and runs significantly faster. Comparing both methods gives a more robust picture of cluster structure than either alone.

**Reference:** McInnes, Healy & Melville (2018), "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction," *arXiv:1802.03426*.

### 9. Outlier Detection via Distance-to-Centroid

**What:** Compute per-class centroids in pixel space, measure each sample's Euclidean distance to its class centroid, display the most atypical samples (farthest from centroid), and show box plots of distances per class.

**Why:** Outlier detection in image datasets catches:
- Labeling errors (a "1" that's actually a "7")
- Unusual writing styles that may confuse classifiers
- Corrupted or low-quality samples

Distance-to-centroid is simple but effective for MNIST because the pixel space is relatively low-dimensional and classes are roughly Gaussian-distributed.

**Reference:** Related to Mahalanobis distance-based outlier detection. See Aggarwal (2017), *Outlier Analysis*, Springer, Chapter 3.

## EDA Techniques: Tabular/Non-Image Data (Contrast)

For non-image (tabular) datasets, standard EDA techniques differ significantly:

| Technique | Tabular EDA | Image EDA Equivalent |
|-----------|-------------|---------------------|
| **Summary statistics** (mean, median, std, min, max) | Per-feature `.describe()` | Mean/std images (Section 5) |
| **Missing value analysis** | `df.isnull().sum()` | N/A (images have fixed pixel grids) |
| **Feature correlation heatmaps** | Pearson/Spearman between columns | Pixel correlation matrix (Section 6) |
| **Distribution plots** (histograms, KDE, box plots) | Per-feature distributions | Pixel intensity distributions (Section 4) |
| **Outlier detection** | IQR method, z-score, isolation forest | Distance-to-centroid (Section 10) |
| **Pairwise scatter plots** | `sns.pairplot()` on features | t-SNE / UMAP projections (Sections 8-9) |
| **Target distribution** | `value_counts()` bar chart | Class distribution analysis (Section 3) |

**Key differences:**
- Tabular data has *named, semantically meaningful features* (age, income, etc.). Image pixels are *anonymous* and only meaningful in spatial context.
- Tabular EDA relies heavily on missing value analysis; images from standard datasets rarely have missing values.
- Tabular correlation heatmaps are directly interpretable (feature A correlates with feature B). Pixel correlations primarily reflect spatial proximity.
- Dimensionality reduction (PCA, t-SNE, UMAP) is essential for image EDA because 784 features can't be inspected individually, but tabular datasets with <50 features can often be explored directly.

## Dependencies

```
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
umap-learn
```

## How to Run

1. Ensure you have the dependencies installed:
   ```bash
   pip install torch torchvision numpy matplotlib seaborn scikit-learn umap-learn
   ```

2. Ensure the MNIST data is available at `<project_root>/data/MNIST/`. The notebook will auto-download it via torchvision if not present.

3. Open and run the notebook:
   ```bash
   cd exploration/mnist_eda/
   jupyter notebook mnist_eda.ipynb
   ```

4. The notebook runs end-to-end in sequence. Some cells (t-SNE, UMAP) may take 1-2 minutes on CPU.

## References

1. **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P.** (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324. http://yann.lecun.com/exdb/mnist/

2. **van der Maaten, L., & Hinton, G.** (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605. https://jmlr.org/papers/v9/vandermaaten08a.html

3. **McInnes, L., Healy, J., & Melville, J.** (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*. https://arxiv.org/abs/1802.03426

4. **Tukey, J. W.** (1977). *Exploratory Data Analysis*. Addison-Wesley.

5. **Pearson, K.** (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559-572.

6. **Jolliffe, I. T.** (2002). *Principal Component Analysis* (2nd ed.). Springer.

7. **He, H., & Garcia, E. A.** (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

8. **Aggarwal, C. C.** (2017). *Outlier Analysis* (2nd ed.). Springer.

9. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/

10. **scikit-learn documentation** -- PCA, t-SNE: https://scikit-learn.org/stable/modules/decomposition.html

11. **UMAP documentation**: https://umap-learn.readthedocs.io/

12. **matplotlib documentation**: https://matplotlib.org/stable/

13. **seaborn documentation**: https://seaborn.pydata.org/

14. **Kaggle: MNIST Classification EDA**: https://www.kaggle.com/code/dejavu23/mnist-classification-eda

15. **Kaggle: MNIST -- Preprocessing & Classifiers**: https://www.kaggle.com/code/gpreda/mnist-preprocessing-classifiers
