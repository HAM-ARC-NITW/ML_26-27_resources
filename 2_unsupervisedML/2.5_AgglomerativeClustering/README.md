# Agglomerative Clustering
Agglomerative Clustering is a bottom-up, unsupervised hierarchical clustering method. 

It begins with every data point as its own cluster and iteratively merges the closest pairs until a single root cluster remains.

![clustering](<assets/Screenshot 2026-09-27 at 12.40.38 PM.png>)

## Why Use It?
### What problem does it solve compared with flat clustering?

- No need to choose k in advance
- Reveals nested structure.
Clusters within clusters: taxonomies, gene families, product categories.
- Works with any dissimilarity: Euclidean, cosine, correlation, Jaccard, custom
- No random initialisation; the dendrogram shows every single merge.

## Algorithm
### 1. Initialize
Treat each of the n
data points as its own
cluster.

### 2. Compute
Build the n × n matrix
of pairwise distances
between points.

### 3. Merge
Find the closest pair
of clusters and merge
them into one.

### 4. Update
Recompute distances
from the new cluster
to all others using the
linkage rule.

### 5. Repeat
Continue until one
cluster remains;
record each merge
height.

## Further info
For more theoretical details: [Agglomerative_Clustering documentation](docs/Agglomerative_Clustering.pdf)

For code implementation: [agglomerative clustering implementation](code/main.ipynb)
