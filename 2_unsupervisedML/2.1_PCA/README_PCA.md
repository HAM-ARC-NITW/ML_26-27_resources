# Principal Component Analysis (PCA)

## Overview
Principal Component Analysis (PCA) is an unsupervised dimensionality-reduction technique used to represent high-dimensional data using fewer features while preserving as much variance as possible.

## Key Concepts
- **Principal Components:** New features formed as linear combinations of the original features.
- **PC1:** The direction that captures the greatest variance in the data.
- **PC2 and later components:** Capture the greatest remaining variance while being orthogonal to earlier components.
- **Explained Variance:** Measures how much of the dataset's variance is captured by each principal component.

## How PCA Works
1. **Prepare the data:** Standardize features when they use different units or scales.
2. **Calculate covariance:** Examine how features vary together.
3. **Find principal components:** Use eigenvectors to identify component directions and eigenvalues to measure the variance associated with them.
4. **Select components:** Keep the number of components needed for the desired explained variance.
5. **Transform the data:** Project the original data onto the selected components to obtain a lower-dimensional representation.

## Advantages
- Reduces the number of features.
- Can make downstream processing more efficient.
- Helps visualize high-dimensional data.
- Represents correlated features using orthogonal components.
- Can support data compression and exploratory analysis.

## Limitations
- Some information may be lost when dimensions are removed.
- Results can be sensitive to outliers and feature scaling.
- Principal components can be harder to interpret than original features.
- Standard PCA captures linear structure and does not use target labels to select components.

## Applications
- Data visualization
- Image processing and compression
- Machine-learning preprocessing
- Exploratory data analysis
- High-dimensional dataset analysis

## Notes
PCA is an unsupervised technique. The number of components should be chosen according to the dataset and the goal of the analysis. For machine-learning workflows, fit preprocessing and PCA on the training data only, then apply the fitted transformations to validation or test data to avoid data leakage.

## References
- Principal Component Analysis (PCA) concepts and workflow.
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/decomposition.html#pca
