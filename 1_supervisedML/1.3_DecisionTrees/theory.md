# Decision Trees
---
*HAM ARC ML Sessions, 2026–27.*
*Group: [Adeeba](https://github.com/shaikadeebatamjeed-hash), [Karthik](https://github.com/KarthikNagendran).*

if you prefer powerpoint presentations, [here](DecisionTree_HAMArc.pdf) ya go

<table width="100%" style="background-color:#F0F4F8; border:none;">
<tr><td style="padding:16px;">

**<span style="color:#293C70;">AGENDA / TABLE OF CONTENTS</span>**

- 01. Introduction to Decision Trees (Nodes, Structure, Examples)
- 02. Purity & Entropy (Mathematical Foundations & Decision Criteria)
- 03. Information Gain (Split Optimization, Continuous Variables & Encoding)
- 04. Tree Ensembles & Bootstrapping
- 05. Generating Tree Samples (Bagging, Random Forests & XGBoost)
- 06. Comparison: Decision Trees vs. Neural Networks

</td></tr>
</table>

## <span style="color:#5064A0;">01</span> <span style="color:#293C70;">Introduction to Decision Trees</span>

A Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It structures data splits in a tree-like flowchart format.

- Key Components of a Decision Tree:
- **Root Node:** The topmost node that represents the entire dataset and gets split first based on the most dominant feature (e.g., Feature 1).
- **Decision Nodes:** Sub-nodes that split into further branches based on feature values (e.g., Feature 1, Feature 2, Feature 3).
- **Leaf Nodes (Terminal Nodes):** The final output nodes that do not split further and contain the class labels or final predictions.

**Example Dataset (Cat Classification):**

| **Ear shape (x1)** | **Face shape (x2)** | **Whiskers (x3)** | **Cat (Y)** |
|---|---|---|---|
| Pointy | Round | Present | Yes |
| Floppy | Not Round | Present | Yes |
| Floppy | Round | Present | No |
| Floppy | Round | Absent | No |

**Tree Representation Options:**
*Is this the only possible tree for a given dataset?* NO! Multiple valid decision trees can be constructed depending on which feature is selected as the root or internal split nodes (e.g., splitting first on Ear Shape vs. Whiskers vs. Face Shape).

**Important Note:** Some decision trees might not correctly classify all training data or might overfit. Therefore, we must systematically select features at each node such that the resulting tree accurately classifies the data. To achieve this, we evaluate **Purity, Entropy,** and **Information Gain.**

## <span style="color:#5064A0;">02</span> <span style="color:#293C70;">Purity & Entropy</span>

**1. What is Purity?**
Purity measures how homogeneous or uniform the samples are within a single node.

> **<span style="color:#293C70;">Formula:</span> Purity = (Number of samples in the majority class) / (Total samples)**

$$\text{Purity} = \frac{\text{Number of samples in the majority class}}{\text{Total samples}}$$

| **Pass** | **Fail** | **Purity** |
|---|---|---|
| 10 | 0 | 10 / 10 = 1.0 (100%) |
| 4 | 6 | 6 / 10 = 0.6 (60%) |

**Benefits of Purity:**
- Helps choose the root node: The feature that maximizes purity is preferred.
- Helps decide when to stop splitting: Splitting stops when a node reaches 100% purity or a maximum predefined depth is reached (to prevent overfitting).

**2. What is Entropy?**
Entropy is a mathematical metric used to quantify the level of impurity, disorder, or randomness at a given node.

- **Maximum Impurity (H = 1):** If a node has a perfectly even 50/50 split of classes (e.g., 50% Cat, 50% Not Cat), entropy is 1. This represents maximum uncertainty.
- **Maximum Purity (H = 0):** If a node contains only a single class (100% Cat or 0% Cat), entropy drops to 0. This represents total certainty.

> **<span style="color:#B43232;">Rule: Lower Entropy = Higher Purity | Higher Entropy = Lower Purity</span>**

**Entropy Function Formula:**

$$H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)$$

where $p_0 = 1 - p_1$, so:

$$H(p_1) = -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1)$$

Note: By mathematical convention, $\log_2(0)$ is defined as $0$ to avoid undefined limits.

## <span style="color:#5064A0;">03</span> <span style="color:#293C70;">Information Gain</span>

Information Gain (IG) measures how much a particular feature split reduces entropy (improves purity) compared to the parent node.

> **<span style="color:#293C70;">Information Gain = H(p_root) − [ w_left · H(p_left) + w_right · H(p_right) ]</span>**

$$IG = H(p_{root}) - \left[ w_{left} \cdot H(p_{left}) + w_{right} \cdot H(p_{right}) \right]$$

**Example Calculation:**
For a node split where $w_{left} = 4/7\ (0.57)$ and $w_{right} = 1/3\ (0.33)$:

$$IG = H(0.5) - \left[ (0.57) \cdot H(0.57) + (0.33) \cdot H(0.33) \right]$$

$$IG = 1 - \left[ 0.57 \cdot (0.9858) + 0.33 \cdot (0.9149) \right] = 1 - 0.9646 = 0.0354$$

**Benefits & Stopping Criteria:**
- Root Node Selection: The feature yielding the highest Information Gain is selected as the split node.
- Stopping Rules: Splitting stops when Information Gain falls below a defined threshold, when sample size per node becomes too small, when 100% purity (H=0) is reached, or when maximum tree depth is met.

**One-Hot Encoding:**
When categorical feature values have no natural numerical order (e.g., Ear Shape = Pointy, Floppy), they are converted into binary vector representations (0s and 1s) to allow distance and gain calculations.

**Splitting on Continuous Variables:**
For continuous numerical features (e.g., Weight = 8.8, 9.2, 11.4, 15.8 kg):
1. Sort the feature values and test prospective split thresholds (e.g., Weight ≤ 9, Weight ≤ 12).
2. Calculate Information Gain for each candidate threshold value.
3. Select the threshold that maximizes IG (e.g., IG at threshold 9 = 0.61 vs IG at 12 = 0.27; hence split on Weight ≤ 9).

## <span style="color:#5064A0;">04</span> <span style="color:#293C70;">Tree Ensembles & Bootstrapping</span>

**Tree Ensemble Concept:**
A single decision tree can be highly sensitive to slight changes in the training data—adding even a single extra dataset point can drastically alter the root node and tree structure. To solve this instability, we build multiple decision trees and combine their predictions into a **Tree Ensemble (Forest)**. Predictions are made via majority voting (for classification) or averaging (for regression).

**Bootstrapping (Sampling with Replacement):**
To train diverse trees from a single training dataset, we use Bootstrapping:
- Imagine putting all training examples into a bag.
- Randomly draw one example, record it, and put it BACK into the bag.
- Repeat this $m$ times to construct a new dataset subset of size $m$.
- Train a decision tree on this subset. Repeat $B$ times to generate $B$ distinct trees.

## <span style="color:#5064A0;">05</span> <span style="color:#293C70;">Generating Tree Samples</span>

**1. Bagged Decision Trees & Random Forest**
Given a training set of size $m$, the bootstrapping process is repeated for $b = 1$ to $B$ times (typically $B = 64\text{–}128$, ideally $< 100$).
Trees created purely via bootstrap sampling are called **Bagged Decision Trees.**

- **The Random Forest Extension:** In standard bagging, if one feature is overwhelmingly strong, almost all trees pick it as the root node, making the trees highly correlated. To fix this, Random Forests force each split to select from a **random subset of $k$ features** (where $k < n$ total features, typically $k = \sqrt{n}$). This decorrelates the trees and ensures high diversity.

**2. XGBoost (Extreme Gradient Boosting)**
Unlike Bagging (where trees are built independently in parallel), Boosting builds trees **sequentially**:
- For $b = 1$ to $B$: Construct a new dataset of size $m$ using sampling with replacement.
- Instead of equal probability ($1/m$), increase the sampling probability for examples that were misclassified by previous trees.
- Each new tree sequentially focuses on fixing the errors (residuals) left behind by prior trees, iteratively minimizing total prediction error.

**Why Use XGBoost?**

- Open-source, highly scalable implementation of boosted trees.
- Fast and computationally efficient parallel execution.
- Built-in regularizations (L1/L2) that prevent overfitting.
- Smart default criteria for splitting and early stopping.
- Highly competitive algorithm across Kaggle & real-world tabular data benchmarks.

## <span style="color:#5064A0;">06</span> <span style="color:#293C70;">Decision Trees vs. Neural Networks</span>

**Comparison Summary Table:**

| **Feature / Property** | **Decision Trees & Ensembles** | **Neural Networks** |
|---|---|---|
| **Data Type** | Best for Tabular (Structured) data (CSVs, Excel) | Best for Unstructured data (Images, Audio, Text, Video) |
| **Training Speed** | Fast to train and evaluate | Slow (requires high compute / GPUs) |
| **Interpretability** | High for small trees (clear decision rules) | Low / Black-box model |
| **Cost / Resources** | Low cost, non-expensive | High computational & hardware cost |
| **Modularity / Pipeline** | Harder to combine end-to-end | Easy to chain, string together & use Transfer Learning |