# Classification of multivariate signals using the Hilbert transform and Riemannian geometry
This repository accompanies the paper: "*Classification of multivariate signals using the Hilbert transform and Riemannian geometry*" [1]. For a quick breakdown of the approach, see the [Explanation of Methods](#explaination-of-methods) below. If you find this code useful in your research, please cite our work: [Insert Citation]


# Abstract:
We describe two Hilbert transform-based methods for augmenting covariance matrices from multivariate signals. The two methods are shown to be isometric under the Riemannian affine-invariant metric on the manifold of symmetric/Hermitian positive definite matrices. The augmented representations distinguish cases that standard covariances clearly cannot and, when paired with a Riemannian minimum distance to mean classifier, improve classification of both synthetic data and real EEG data without introducing any extra hyperparameters. This novel combination of methods also significantly improves tangent space classifier accuracy on the same dataset, outperforming state-of-the-art classifiers with similarly sized parameter grids. We also examine how the augmented covariances interact with the minimum distance to mean classifier and show how multivariate cross-covariance functions behave under the Hilbert transform.



# Usage:

### 1. Setup:
Install the required packages (see `requirements.txt`):

```bash
pip install -r requirements.txt
```


### 2. Reproduce paper the results:
To replicate the results and figures presented in the paper, run the following commands in your terminal:

**Example 4.2 Results**
```bash
python ex2.py
python ex2_plot.py
```

**EEG Results**
```bash
python eeg_classification.py
python eeg_plot.py
python eeg_analysis.py
```


### 3. Example usage in scikit-learn pipeline:
The estimators and transformers are designed to be compatible with `scikit-learn` Pipelines. The pipelines expect labeled multivariate time-sereis data of shape `(n_data, n_channels, n_times)`.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.classification import MDM

# Imports from this repo
from estimation import AnalyticCovariances
from tangentspace import TangentSpaceSub, TangentSpaceHPD

# --- Classifier pipeline examples ---

# 1. Analytic Covariance (ACOV) + MDM
clf_acov_mdm = Pipeline([
    ('acov', AnalyticCovariances()),
    ('clf', MDM())
])

# 2. Hilbert Augmented Covariance (HACOV) + MDM
clf_hacov_mdm = Pipeline([
    ('hacov', AnalyticCovariances(real_output=True)),
    ('clf', MDM())
])

# 3. ACOV + Tangent Space + Logistic Regression
clf_acov_tsh_lr = Pipeline([
    ('acov', AnalyticCovariances()),
    ('ts', TangentSpaceHPD()),
    ('lr', LogisticRegression())
])

# 4. HACOV + Subspace Tangent Space mapping
clf_hacov_tssub_lr = Pipeline([
    ('hacov', AnalyticCovariances(real_output=True)),
    ('ts', TangentSpaceSub()),
    ('lr', LogisticRegression())
])
```


# Explaination of Methods:

For a more detailed walkthrough see the full paper [INSERT LATER] [1].

## Standard Method: Classification of Covariance Matrices (SPD):
This methodology was popularized for EEG classification by Barachant et al. (2012) in the paper, "Multiclass Brain–Computer Interface Classification by Riemannian Geometry" [2].

In this framework, data is represented by covariance matrices of size $n \times n$. For $N$ samples of a zero mean multivariate time series data $\mathbf{x}(t) \in \mathbb{R}^n$, the covariance estimated as:

$$
\hat{X} = \frac{1}{N-1}\sum_{t=0}^N \mathbf{x}(t)\mathbf{x}^T(t) \in \mathbb{R}^{n \times n}
$$

#### The Geometry of SPD Matrices

Covariance matrices belong to the set of **Symmetric Positive Definite (SPD)** matrices, which form a differentiable manifold. In any point of the manifold, the manifold can be locally approximated by a Euclidean space, known as the **tangent space** at that point. For a point in the SPD manifold, the tangent space is the set of symmetric matrices. 

Using standard Euclidean distances in this space often leads to undesirable effects, such as  *determinant swelling* and geodesic extrapolations extending outside the manifold. Instead, we can equip the manifold with a Riemannian metric that will lead to an alternative notion of distance. While several metrics exist, a popular choice is the **Affine Invariant Riemannian (AIR) metric**. The AIR inner product (Riemannian metric) of two tangent space vectors (symmetric matrices) $V,W$ at a point $X$ on the manifold is:

$$
\langle V,W \rangle_{X} = \langle X^{-1/2}VX^{-1/2}, X^{-1/2}WX^{-1/2} \rangle_{F} = \mathrm{tr}(X^{-1}VX^{-1}W).
$$

#### The AIR Distance

The distance induced by the AIR metric represents the length of the shortest path (geodesic) between two covariance matrices $X$ and $Y$:

$$
d_{\text{AIR}}(X,Y) = \left \Vert \log \left( X^{-1/2}YX^{-1/2} \right) \right \Vert_{F} = \left[\sum_{i=1}^n \ln^2(\lambda_{i}) \right]^{1/2}
$$

where $\lambda_i$ are the eigenvalues of $X^{-1/2}YX^{-1/2}$.
#### The Geometric (Fréchet) Mean
For a set of points (SPD matrices) on this manifold, an "average" or geometric mean can be defined as the Fréchet mean ($M$), which is the element that minimizes the sum of squared distances to all elements in the set:

$$
M = \arg\min_X \sum_{i=1}^N d_{\text{AIR}}^2(X_i,X).
$$

#### Classification Strategies

For a dataset with labeled covariance matrices, two common classification strategies are:
1. **Minimum Distance to Mean (MDM) classifier:** First, the Fréchet mean is calculated for each class. A new data point is assigned to the class whose mean is closest under the AIR distance. This method operates entirely on the manifold.
2. **Tangent Space (TS) classifier:** Here, data (SPD matrices) are projected to a tangent space. Since the tangent space is a vector space, standard machine learning can be applied (after a few additional processing steps, see below).

#### Tangent Space Classification Explained

In order to apply standard classifiers (which often assume data from a Euclidean vector space), the following steps are used:

1. **Find reference point:** The Fréchet mean ($M$) of the training set is computed to serve as the reference point for the tangent space projection.
2. **Tangent space projection:** All matrices are mapped into the tangent space at $M$ using the **Logarithmic map**. The resulting projected data are symmetric matrices.
3. **Whiten:** To ensure the standard Euclidean/Frobenius inner product matches the Riemannian metric, the tangent vectors are whitened using the map $\phi_{M}(V) = M^{-1/2}VM^{-1/2}.$ This "straightens" the geometry so that the Frobenius/Euclidean inner product between whitened matrices is equivalent to the AIR inner product for tangent vectors (symmetric matrices).
4. **Vectorize:** The upper triangular elements of the whitened matrices are then extracted and stacked ($\mathrm{vech}$). The off-diagonal elements are scaled $\sqrt{2}$ giving the final data representation as:

$$
V^{\text{vec}} = \mathrm{vech}(\sqrt{2}\phi_{M}(V) + (1-\sqrt{2})\mathrm{diag}(\phi_{M}(V))).
$$

The final representation ensures that the standard dot product between such vectors matches the Riemannian metric as: $\langle V^{\text{vec}}, W^{\text{vec}} \rangle = \langle V,W \rangle_{M}$. Now, any standard classifier, such as **Logistic Regression (LR)** or **Support Vector Machines (SVM)**, can now be applied directly.


## Paper Method: Classification of Analytic Covariance Matrices (HPD matrices)

While standard covariance matrices capture relationships between signal amplitudes, they cannot distinguish lead-lag relations in phase information. To address this, we use the analytic signal to construct Hermitian Positive Definite (HPD) matrices.

#### The Analytic Signal and Covariance matrices

For zero mean multivariate time series data $\mathbf{x}(t) \in \mathbb{R}^n$,  the analytic signal using the Hilbert transform (link to Wikipedia) is constructed as:

$$
\mathbf{x}_a(t) = \mathbf{x}(t) + i\,\widehat{\mathbf{x}}(t)
$$

The corresponding complex covariance matrix, or **Analytic Covariance** (ACOV) matrix, is defined as:

$$
X_{a} := \mathbb{E}[\mathbf{x}_{a}\mathbf{x}_{a}^{*}]  = C_{\mathbf{x}\mathbf{x}} + iC_{\widehat{\mathbf{x}}\mathbf{x}} - iC_{\mathbf{x}\widehat{\mathbf{x}}} + C_{\widehat{\mathbf{x}}\widehat{\mathbf{x}}}
    = 2C_{\mathbf{x}\mathbf{x}} + 2iC_{\widehat{\mathbf{x}}\mathbf{x}},
$$

where $C_{\mathbf{x}\mathbf{y}}$ is the cross covariance between two signals $\mathbf{x}(t)$ and $\mathbf{y}(t)$. The resulting matrix $X_a$, is an HPD matrix. From $N$ samples of multivariate time series data $\mathbf{x}(t) \in \mathbb{R}^n$, the analytic covariance can be estimated as:

$$
\hat{X}_a = \frac{1}{N-1}\sum_{t=0}^N \mathbf{x}_{a}(t)\mathbf{x}_{a}^*(t) \in \mathbb{C}^{n \times n}
$$

#### Geometry and Classification

The set of HPD matrices also form a differentiable manifold. Tangent vectors are now Hermitian matrices. The AIR metric and AIR distance for HPD matrices are analogous to the SPD cases (but matrices being HPD/Hermitian). Naturally, the Fréchet mean is found in same way as in the SPD case, but now using the HPD AIR-distance. Clearly an MDM classifier based on ACOV matrices can be constructed in the same way as in the standard SPD case.

#### Complex Vectorization

When projecting HPD matrices to the tangent space using the Logarithmic map, the result is now a Hermitian matrix rather than a symmetric matrix. As standard machine learning algorithms typically require real-valued vectors, the vectorization step is designed to map the complex-valued Hermitian matrices into a real column vector while preserving the inner product from the HPD AIR metric. For this we use the same whitening step as before followed by stacking the upper diagonal real and imaginary parts individually. We apply the same scaling logic as the SPD case i.e., scaling off-diagonals by $\sqrt{2}$ as:

$$
V_{a}^\text{col}
=\begin{bmatrix}
    \mathrm{vech}(\sqrt{2}\mathrm{Re}(\phi_{M_{a}}({V_{a}})) + (1-\sqrt{2})\mathrm{diag}(\mathrm{Re}(\phi_{M_{a}}({V_{a}})))) \\
    \mathrm{vech}_{0}(\sqrt{2}\mathrm{Im}(\phi_{M_{a}}({V_{a}})))
\end{bmatrix},
$$

where we use $\mathrm{vech}$ to denote vectorization of the upper triangular part including the diagonal, while $\mathrm{vech}_{0}$ that vectorizes the strictly upper triangular part (excluding the diagonal) is used for the imaginary part (as a Hermitian matrix has a zero-diagonal). The final vector representation ensures that the standard dot product between such vectors matches the Riemannian metric as: 

$$
\langle V_{a},W_{a} \rangle_{M_{a}} = (V_{a}^\text{col})^T W_{a}^\text{col}
$$

Using this representation, we retain an approximation of geometry of the HPD Riemannian manifold while data is mapped to a real-valued vector space. We are ready to apply any standard ML algorithm.


# References:

[1] [Insert reference to paper here.]

[2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass Brain–Computer Interface Classification by Riemannian Geometry," in IEEE Transactions on Biomedical Engineering, vol. 59, no. 4, pp. 920-928, April 2012, doi: 10.1109/TBME.2011.2172210.
