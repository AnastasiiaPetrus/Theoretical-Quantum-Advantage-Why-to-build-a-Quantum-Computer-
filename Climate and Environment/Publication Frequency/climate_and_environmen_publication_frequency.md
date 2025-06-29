# Supplementary Table S2. Standardized and Aggregated Algorithm Frequency (fpub)

## Overview

To ensure reproducibility and comparability of results on algorithm usage in climate and environmental research, we systematically aggregated data from three independent online repositories of peer-reviewed articles. Each source provided a list of algorithms and methods as cited in recent research, but with considerable variability in nomenclature, abbreviations, and variants.

**Standardization Process:**
- All extracted algorithm names were harmonized through manual review, with synonyms and abbreviations unified under a single standard label.
- Algorithmic variants and closely related models were clustered together (e.g., all Random Forest modifications are grouped as "Random Forest," with variants listed in parentheses).
- For each standardized method, we counted the number of times (**Total mentions**) the algorithm or method was referenced across all sources.
- For every algorithm, we calculated the normalized publication frequency (**fpub**) as:
    - **fpub(A) = total mentions of A / Σ total mentions of A'**
    - Here, **Total mentions = 687** (i.e., the sum of all *mentions* in the table).
    - The *fpub* value represents the share of total mentions and enables unbiased comparison of algorithm importance.
- The resulting tables provide a deduplicated, standardized reference list with both *Total mentions* and *fpub* for each algorithm.

---

## Table Legend

- **Algorithm & Method**: Cluster of related algorithms or methods.
- **Total mentions**: The number of times the method was mentioned in the aggregated, deduplicated literature survey (absolute count, 2010–2024).
- **fpub**: The normalized publication frequency, i.e. \( \frac{\text{Total mentions}}{687} \), rounded to three decimal places.

---

## Total mentions: **687**

---

## TOP-10 Algorithms and Methods by Frequency

| Algorithm & Method | Total mentions | fpub |
|------------------------------------------------------|---------------|-------|
| Distributed Random Forest (DRF, RF, RFR, VSURF, MIDAS-QRF, FM-QRF, Factor MIDAS-QRF) | 31 | 0.045 |
| Convolutional Neural Network (CNN, 3D-CNN, Deterministic CNN, EfficientNet, GVSAO-CNN, LSTM-CNN, Mask R-CNN, Multi-Head 1D CNN, SRCNN, CNN-BiLSTM, DeepLabV3+, PSPNet, U-Net, Attention U-Net, ResNet-34, Xception-65) | 22 | 0.032 |
| Linear Regression Methods (LR, LSLR, MLR, Multivariate Linear Regression, Linear Quantile Mixed Model, Segmented Regression, Power-Law Regression, Quantile Regression) | 22 | 0.032 |
| Gradient Boosting Methods (GB, GBM, GBR, ET, XRT, XGBoost, CatBoost, LightGBM) | 21 | 0.031 |
| Support Vector Machine (SVM, SVC, SVR, IPSO‑SVM, PSO‑SVM, SVM with RBF Kernel) | 19 | 0.028 |
| Long Short-Term Memory (LSTM, CNN-LSTM, Transformer-LSTM, GAN-LSTM, LSTM-AE, TimesNet, Informer) | 16 | 0.023 |
| Principal Component Analysis (PCA) | 11 | 0.016 |
| Logistic Regression Models (Logistic Regression, Multilevel Logistic Regression, Multinomial Logit Model) | 10 | 0.015 |
| Markov Chain Monte Carlo (MCMC, Metropolis-Hastings, Gibbs Sampling) | 10 | 0.015 |
| Transformer (MaskFormer, DETR, iTransformer, BEiT, DINOv2, ViT, Transformer Encoder, CNN-Transformer) | 10 | 0.015 |

---

## Supplementary Table: Full Standardized Algorithm Frequency

| Algorithm & Method | Total mentions | fpub |
|--------------------|---------------|-------|
| Distributed Random Forest (DRF, RF, RFR, VSURF, MIDAS-QRF, FM-QRF, Factor MIDAS-QRF) | 31 | 0.045 |
| Convolutional Neural Network (CNN, 3D-CNN, Deterministic CNN, EfficientNet, GVSAO-CNN, LSTM-CNN, Mask R-CNN, Multi-Head 1D CNN, SRCNN, CNN-BiLSTM, DeepLabV3+, PSPNet, U-Net, Attention U-Net, ResNet-34, Xception-65) | 22 | 0.032 |
| Linear Regression Methods (LR, LSLR, MLR, Multivariate Linear Regression, Linear Quantile Mixed Model, Segmented Regression, Power-Law Regression, Quantile Regression) | 22 | 0.032 |
| Gradient Boosting Methods (GB, GBM, GBR, ET, XRT, XGBoost, CatBoost, LightGBM) | 21 | 0.031 |
| Support Vector Machine (SVM, SVC, SVR, IPSO‑SVM, PSO‑SVM, SVM with RBF Kernel) | 19 | 0.028 |
| Long Short-Term Memory (LSTM, CNN-LSTM, Transformer-LSTM, GAN-LSTM, LSTM-AE, TimesNet, Informer) | 16 | 0.023 |
| Principal Component Analysis (PCA) | 11 | 0.016 |
| Logistic Regression Models (Logistic Regression, Multilevel Logistic Regression, Multinomial Logit Model) | 10 | 0.015 |
| Markov Chain Monte Carlo (MCMC, Metropolis-Hastings, Gibbs Sampling) | 10 | 0.015 |
| Transformer (MaskFormer, DETR, iTransformer, BEiT, DINOv2, ViT, Transformer Encoder, CNN-Transformer) | 10 | 0.015 |
| K-Means Clustering (K-Means, DTW-Based K-Means, K-Means Co-Clustering) | 9 | 0.013 |
| Artificial Neural Network (ANN, FNN, GRNN, GA-ANN, PSO-ANN) | 7 | 0.010 |
| Bayesian Statistical Methods (BART, BNP-PC, Bayesian Age-Depth Modeling, Bayesian Mixed-Effects Modeling, Bayesian Inversion, Bayesian Update for Occupancy Mapping, Bayesian Optimization Algorithm) | 7 | 0.010 |
| Reinforcement Learning (RL, MARL, REINFORCE, IPPO, PPO, TRPO, Actor-Critic, A2C, DDPG, SAC, TD3, DPG, TQC) | 7 | 0.010 |
| Decision Tree Methods (CART, DT, RT, M5P) | 6 | 0.009 |
| Independent Proximal Policy Optimization (IPPO, PPO, TRPO, REINFORCE) | 6 | 0.009 |
| Multi-Layer Perceptron (MLP) | 6 | 0.009 |
| Neural Network (NN) | 6 | 0.009 |
| Partial Least Squares (PLS-DA, PLSR) | 6 | 0.009 |
| SHapley Additive Explanations (SHAP) | 6 | 0.009 |
| Fuzzy Methods (Fuzzy C-Means, IVFS, Spatial Fuzzy Kappa, Fuzzy Overlay Analysis) | 5 | 0.007 |
| Gaussian Methods (Gaussian Graphical Models, Gaussian Processes, GPR, GRBF) | 5 | 0.007 |
| Generative Adversarial Network (GAN, ESRGAN, SRGAN) | 5 | 0.007 |
| Grid Search (GridSearchCV, Random GridSearchCV) | 5 | 0.007 |
| K-Nearest Neighbors (KNN, ItemNN, UserNN) | 5 | 0.007 |
| Adam Optimizer | 4 | 0.006 |
| Levenberg–Marquardt (LM) Optimization Algorithm | 4 | 0.006 |
| Monte Carlo Simulation | 4 | 0.006 |
| Savitzky–Golay Filter | 4 | 0.006 |
| Actor-Critic (Actor-Critic, A2C, DDPG, DPG, SAC, TD3) | 3 | 0.004 |
| AutoRegressive Integrated Moving Average (ARIMA, SARIMA, SARIMAX) | 3 | 0.004 |
| Autoencoder (AE, VAE) | 3 | 0.004 |
| DeepLabV3+ | 3 | 0.004 |
| Ensemble Kalman Filter (EnKF) | 3 | 0.004 |
| Graph Neural Networks (GCN, GNN, Local Learnable GNN-Model) | 3 | 0.004 |
| Hierarchical Clustering (Ward’s Hierarchical Clustering, Hierarchical Frequency-Based Location Estimation) | 3 | 0.004 |
| K-Fold Cross-Validation (KFCV) | 3 | 0.004 |
| Kernel Density Estimation (KDE) | 3 | 0.004 |
| Ridge Regression | 3 | 0.004 |
| Analysis of Variance (ANOVA) | 2 | 0.003 |
| Bayesian Neural Network (BNN) | 2 | 0.003 |
| Bi-Spectral Method | 2 | 0.003 |
| Boruta Algorithm | 2 | 0.003 |
| Copulas for Multivariate Hazard Modeling | 2 | 0.003 |
| Deep Neural Network (DNN) | 2 | 0.003 |
| Diffusion Models (DDIM, DDPM, LDM) | 2 | 0.003 |
| Dynamic Time Warping (DTW, TWDTW) | 2 | 0.003 |
| Empirical Orthogonal Function (EOF) | 2 | 0.003 |
| Fast Fourier Transform (FFT) | 2 | 0.003 |
| Feature Pyramid Network (FPN) | 2 | 0.003 |
| Gated Recurrent Unit (GRU) | 2 | 0.003 |
| Generalized Additive Models (GAM, GAMLSS) | 2 | 0.003 |
| Generalized Linear Model (GLM) | 2 | 0.003 |
| Grey Wolf Optimizer (GWO, IGWO) | 2 | 0.003 |
| Kalman Filter (KF) | 2 | 0.003 |
| Limited-memory Broyden–Fletcher–Goldfarb–Shanno Optimization Algorithm (L-BFGS, L-BFGS-B) | 2 | 0.003 |
| Lagrangian Particle Trajectory Tracking | 2 | 0.003 |
| Latent Dirichlet Allocation (LDA) | 2 | 0.003 |
| Least Absolute Shrinkage and Selection Operator (LASSO) | 2 | 0.003 |
| Linear Interpolation | 2 | 0.003 |
| Linear Programming | 2 | 0.003 |
| Morphological Thinning | 2 | 0.003 |
| Partial Regression Analysis | 2 | 0.003 |
| Physically Informed Neural Network (PINN) | 2 | 0.003 |
| Random Grid Search (GridSearchCV, Random GridSearchCV) | 2 | 0.003 |
| Random Subspace Method (RSS) | 2 | 0.003 |
| Response Surface Methodology (RSM) | 2 | 0.003 |
| Seasonal Autoregressive Integrated Moving Average (SARIMA) | 2 | 0.003 |
| Self-Attention Mechanism (Multihead Self-Attention, Self-Attention) | 2 | 0.003 |
| Singular Value Decomposition (SVD, SVD++) | 2 | 0.003 |
| Spectral Clustering | 2 | 0.003 |
| Transfer Learning | 2 | 0.003 |
| Welch Method | 2 | 0.003 |
| 4D-Var (Four-Dimensional Variational Data Assimilation) | 1 | 0.001 |
| Accumulated Latitudinal Dispersal Calculation | 1 | 0.001 |
| AdaBoost | 1 | 0.001 |
| Adams-Bashforth Two-Step Time Integration | 1 | 0.001 |
| Adaptive Combiner | 1 | 0.001 |
| Adaptive Multiple Importance Sampling (AMIS) | 1 | 0.001 |
| Adaptive Neuro-Fuzzy Inference System (ANFIS) | 1 | 0.001 |
| Additive Regression (AR) | 1 | 0.001 |
| Adjoint Method | 1 | 0.001 |
| Advection–Diffusion Modeling | 1 | 0.001 |
| Adversarial Loss | 1 | 0.001 |
| AI-Supported Object Detection Pipeline (Head-Tail Segmentation & Classification) | 1 | 0.001 |
| Alternating Direction Method of Multipliers (ADMM) | 1 | 0.001 |
| Analysis of Compositions of Microbiomes with Bias Correction (ANCOM-BC2) | 1 | 0.001 |
| AR(1) Model | 1 | 0.001 |
| Atrous Spatial Pyramid Pooling (ASPP) | 1 | 0.001 |
| Transformer (Transformer Encoder, CNN-Transformer, DETR, iTransformer, ViT, BEiT, DINOv2) | 1 | 0.001 |
| Augmented Dickey-Fuller Test (ADF) | 1 | 0.001 |
| Augmented Random Search (ARS) | 1 | 0.001 |
| Backward Selection Using Akaike Information Criterion (AIC) | 1 | 0.001 |
| Backward Trajectory Analysis | 1 | 0.001 |
| Bagging (Bootstrap Aggregating) | 1 | 0.001 |
| Best Subset Regression Analysis | 1 | 0.001 |
| Bicubic Interpolation | 1 | 0.001 |
| Bidirectional Gated Recurrent Units (Bi-GRU) | 1 | 0.001 |
| Binary Dynamic Programming | 1 | 0.001 |
| Binomial Likelihood Modeling | 1 | 0.001 |
| Boolean Matching | 1 | 0.001 |
| Bootstrap Resampling | 1 | 0.001 |
| Breadth-First Search (BFS) | 1 | 0.001 |
| Breaks for Additive Season and Trend (BFAST) | 1 | 0.001 |
| Canonical Correlation Analysis (CCA) | 1 | 0.001 |
| CaViaR (SAV, AD, AS, IG) | 1 | 0.001 |
| Central Composite Design (CCD) | 1 | 0.001 |
| Centroid-Based Instance Segmentation | 1 | 0.001 |
| Change Vector Analysis (CVA) | 1 | 0.001 |
| CHARMm Algorithm | 1 | 0.001 |
| Climate Match Algorithm | 1 | 0.001 |
| Climate Niche Envelope Approach | 1 | 0.001 |
| Clustal Omega | 1 | 0.001 |
| Cluster Analysis | 1 | 0.001 |
| Clustering Algorithm for Macro- and Meso-Segment Detection | 1 | 0.001 |
| Cobweb Model | 1 | 0.001 |
| Column-and-Constraint Generation Algorithm (C&CGA) | 1 | 0.001 |
| Comparative Molecular Field Analysis (CoMFA) | 1 | 0.001 |
| Compartmental Dynamical Thermodynamics / TMN Modeling | 1 | 0.001 |
| Conditional Autoregressive (CAR) Model | 1 | 0.001 |
| Conditional Logistic Regression | 1 | 0.001 |
| Consistency Distillation | 1 | 0.001 |
| Consistency Models (CM) | 1 | 0.001 |
| Constrained Delaunay Triangulation (CDT) | 1 | 0.001 |
| Constraint-Driven Optimization | 1 | 0.001 |
| Contrastive Learning (InfoNCE Loss) | 1 | 0.001 |
| Correction Method for Population Weighting | 1 | 0.001 |
| DADA2 (High-Resolution Sample Inference) | 1 | 0.001 |
| Deep Ensembles | 1 | 0.001 |
| Delaunay Triangulation (via JIGSAW) | 1 | 0.001 |
| Denoising Diffusion Implicit Models (DDIM) | 1 | 0.001 |
| Denoising Diffusion Probabilistic Models (DDPM) | 1 | 0.001 |
| Density Functional Theory (DFT) Calculations | 1 | 0.001 |
| Density-Based Spatial Clustering of Applications with Noise (DBSCAN) | 1 | 0.001 |
| Desirability Function Approach | 1 | 0.001 |
| Differential Flatness Transform | 1 | 0.001 |
| Dirichlet Process Mixture Model (DPM) | 1 | 0.001 |
| DiRienzo–Zurbenko Algorithm (DZA) | 1 | 0.001 |
| Discrete Element Method (DEM) Simulation | 1 | 0.001 |
| Dispersal–Extinction–Cladogenesis (DEC) Model | 1 | 0.001 |
| Distance-Based Redundancy Analysis (dbRDA) | 1 | 0.001 |
| DistilBERT | 1 | 0.001 |
| Distorted Born Approximation (DBA) | 1 | 0.001 |
| DLinear | 1 | 0.001 |
| DNA Sequence Alignment | 1 | 0.001 |
| DOMAIN Algorithm | 1 | 0.001 |
| Dropout Regularization | 1 | 0.001 |
| Dynamic Dewpoint Isotherm (DDI) Method | 1 | 0.001 |
| Dynamic MIDAS-QRF | 1 | 0.001 |
| Dynamic Programming | 1 | 0.001 |
| Edge-Greedy Search for Causal Graph Construction | 1 | 0.001 |
| eDMD with Dictionary Learning (eDMD-DL) | 1 | 0.001 |
| ElasticNet | 1 | 0.001 |
| Empirical Mode Decomposition (EMD) | 1 | 0.001 |
| End-Member Modeling Algorithm (EMMA) | 1 | 0.001 |
| Energy and Mass Conservation Equations | 1 | 0.001 |
| Ensemble Averaging | 1 | 0.001 |
| Entropy Weight Method | 1 | 0.001 |
| Enzymatic Catalysis Modelling | 1 | 0.001 |
| Equal-Area Quadratic Spline | 1 | 0.001 |
| Ewald Summation | 1 | 0.001 |
| Exergy Analysis | 1 | 0.001 |
| Expectation-Maximization (EM) Algorithm | 1 | 0.001 |
| Experience Replay | 1 | 0.001 |
| Exponential Smoothing | 1 | 0.001 |
| Extended Dynamic Mode Decomposition (eDMD) | 1 | 0.001 |
| Extended Fourier Amplitude Sensitivity Testing (EFAST) | 1 | 0.001 |
| Extreme Learning Machine (ELM) | 1 | 0.001 |
| Extremely Randomized Trees (XRT) | 1 | 0.001 |
| Fama–French Three-Factor and Five-Factor Model Regressions | 1 | 0.001 |
| Fast LiDAR-Inertial Odometry 2 (FAST-LIO2) | 1 | 0.001 |
| Data Fusion Methods (Data-Level Fusion, Decision-Level Fusion, Feature-Level Fusion) | 1 | 0.001 |
| Finite Differences Simulator | 1 | 0.001 |
| Finite Volume Method (FVM) | 1 | 0.001 |
| Fully Connected Neural Network (FNN) | 1 | 0.001 |
| Galerkin Method | 1 | 0.001 |
| Gamma Regression | 1 | 0.001 |
| Generalized Autoregressive Conditional Heteroskedasticity / GARCH-Mixed Data Sampling (GARCH / GARCH-MIDAS) | 1 | 0.001 |
| Gene Expression Programming (GEP) | 1 | 0.001 |
| Generalized Estimating Equations (GEE) | 1 | 0.001 |
| Generalized Likelihood Uncertainty Estimation (GLUE) | 1 | 0.001 |
| Generalized Three-Cornered Hat (TCH) Method | 1 | 0.001 |
| Genetic Algorithm (GA, NSGA-II, RAGA) | 1 | 0.001 |
| Geodetector Model (GDM) | 1 | 0.001 |
| GF‑SG Fusion Algorithm | 1 | 0.001 |
| GIS-Based Spatial Multi-Criteria Evaluation | 1 | 0.001 |
| Graphical LASSO (gLASSO) | 1 | 0.001 |
| Gravity Search Optimization Algorithm (GVSAO) | 1 | 0.001 |
| Grid Interpolation (Linear) | 1 | 0.001 |
| Group-Based Trajectory Modeling | 1 | 0.001 |
| Harmonic Tidal Analysis | 1 | 0.001 |
| Harmony Search | 1 | 0.001 |
| Heat Transfer Algorithm | 1 | 0.001 |
| Hidden Markov Model (HMM) | 1 | 0.001 |
| Hilbert Spectral Analysis | 1 | 0.001 |
| Hobday-Based Anomaly Detection Algorithm | 1 | 0.001 |
| Homology Modeling | 1 | 0.001 |
| Improved 1-D DAS Method | 1 | 0.001 |
| Improved 2-D DAS Method | 1 | 0.001 |
| Incremental kd-Tree (ikd-Tree) | 1 | 0.001 |
| Indicator Species Analysis (IndVal.g) | 1 | 0.001 |
| Initial-Condition-Dependent Finite-Time Stabilizing Controller | 1 | 0.001 |
| Integral Equation Method (IEM) | 1 | 0.001 |
| Interferometric Coherence Techniques | 1 | 0.001 |
| Interquartile Range (IQR)-Based Outlier Detection | 1 | 0.001 |
| Inverse Modelling (Parameter Optimization) | 1 | 0.001 |
| Isotonic Regression | 1 | 0.001 |
| Item Nearest Neighbor (ItemNN) | 1 | 0.001 |
| Iterative Optimization | 1 | 0.001 |
| Jenks Natural Breaks Classification | 1 | 0.001 |
| K-Shape Clustering | 1 | 0.001 |
| Kaplan–Meier Survival Analysis | 1 | 0.001 |
| KDTree-Based Spatial Grouping | 1 | 0.001 |
| Kernelized Hypothesis Testing (HSIC) | 1 | 0.001 |
| Kolmogorov–Smirnov Test (KS Test) | 1 | 0.001 |
| Kolmogorov–Zurbenko (KZ) Filter | 1 | 0.001 |
| Kolmogorov–Zurbenko (KZ) Periodogram | 1 | 0.001 |
| Kriging (Spatial Interpolation) | 1 | 0.001 |
| L-Curve Method | 1 | 0.001 |
| L2 Regularization | 1 | 0.001 |
| Laplace Approximation | 1 | 0.001 |
| LAVD-Based Eddy Detection Algorithm | 1 | 0.001 |
| Least Squares Inversion with Smoothness-Constrained Regularization | 1 | 0.001 |
| Leave-One-Year-Out Cross-Validation | 1 | 0.001 |
| Linear Fitting | 1 | 0.001 |
| Linear Mixed Effect Model (REML) | 1 | 0.001 |
| Linear Mixed Models (LMM) | 1 | 0.001 |
| Linear Quantile Mixed Model (LQMM) | 1 | 0.001 |
| Linearized Multi-Block ADMM with Regularization | 1 | 0.001 |
| Local Outlier Factor (LOF) | 1 | 0.001 |
| Log-Odds Update Scheme | 1 | 0.001 |
| Lossless Compression Algorithm (ZIP) | 1 | 0.001 |
| Lossy Compression Algorithm (ZFP) | 1 | 0.001 |
| Locally Weighted Scatterplot Smoothing (LOWESS) | 1 | 0.001 |
| LSTM Autoencoder (LSTM-AE) | 1 | 0.001 |
| Manual Thresholding | 1 | 0.001 |
| Marker-Assisted Selection | 1 | 0.001 |
| Markov Decision Process (MDP) | 1 | 0.001 |
| Mass Balance Modeling | 1 | 0.001 |
| Mass Curve Technique (MCT) | 1 | 0.001 |
| Match Climates Regional Algorithm | 1 | 0.001 |
| Maximum Likelihood Classification (MLC) | 1 | 0.001 |
| Maximum Power Point Tracking (MPPT) Control | 1 | 0.001 |
| Maximum-Likelihood Phylogenetic Analysis | 1 | 0.001 |
| MIDAS Quantile Random Forest (MIDAS-QRF) | 1 | 0.001 |
| Migration Event Detection Algorithm | 1 | 0.001 |
| Minimum Bounding Rectangle | 1 | 0.001 |
| MIR Method | 1 | 0.001 |
| Mixed Data Sampling (MIDAS) | 1 | 0.001 |
| Mixed-Effects Modeling (LME4) | 1 | 0.001 |
| Mixed-Finite Element Scheme | 1 | 0.001 |
| Mixed-Integer Linear Programming (MILP) | 1 | 0.001 |
| Mixture Density Networks (MDN) | 1 | 0.001 |
| Model Predictive Control (MPC) | 1 | 0.001 |
| Model-Agnostic Meta-Learning (MAML) | 1 | 0.001 |
| Modified Response Matrix Method | 1 | 0.001 |
| Moffat Uncertainty Analysis Method | 1 | 0.001 |
| Molecular Docking | 1 | 0.001 |
| Molecular Dynamics Simulation | 1 | 0.001 |
| Monte Carlo Dropout (MC-Dropout) | 1 | 0.001 |
| Monte Carlo Integration | 1 | 0.001 |
| Moving Average Filter | 1 | 0.001 |
| Multi-Agent Reinforcement Learning (MARL) | 1 | 0.001 |
| Multi-Criteria Performance Evaluation | 1 | 0.001 |
| Multi-Label Machine Learning | 1 | 0.001 |
| Multi-Objective Evolutionary Algorithm (MOEA) | 1 | 0.001 |
| Multi-Objective Optimization | 1 | 0.001 |
| Multi-Task Linear Regression (MTLR) | 1 | 0.001 |
| Multihead Self-Attention Mechanism | 1 | 0.001 |
| Multilevel Logistic Regression (MLLogR) | 1 | 0.001 |
| Multilinear Regression (MLR) | 1 | 0.001 |
| Multinomial Logit Model | 1 | 0.001 |
| Multiscale Geographically Weighted Regression (MGWR) | 1 | 0.001 |
| Multivariate Linear Regression | 1 | 0.001 |
| Multivariate Quantiles and Multiple-Output Regression Quantiles | 1 | 0.001 |
| Multivariate Regression | 1 | 0.001 |
| Naive Bayes | 1 | 0.001 |
| Negative Binomial Mixed-Effects Models | 1 | 0.001 |
| Neighbor-Joining Method | 1 | 0.001 |
| Network-Based Path Filtering | 1 | 0.001 |
| Neural Network Operators (Kantorovich) | 1 | 0.001 |
| Neural Network Operators (Classical) | 1 | 0.001 |
| Neural Network Classifier (NN Classifier) | 1 | 0.001 |
| Non-Dominated Sorting Genetic Algorithm II (NSGA-II) | 1 | 0.001 |
| Non-Metric Multidimensional Scaling (NMDS) | 1 | 0.001 |
| Nonlinear Regression (nlsLM Package) | 1 | 0.001 |
| Nonparanormal Transformation | 1 | 0.001 |
| Normalization of Raster Data | 1 | 0.001 |
| Optical Flow Method | 1 | 0.001 |
| Ordinary Differential Equation (ODE) System Modelling | 1 | 0.001 |
| Parallel Factor Analysis (PARAFAC) Modeling | 1 | 0.001 |
| Partial Correlation Analysis | 1 | 0.001 |
| Pruned Exact Linear Time Algorithm (PELT) | 1 | 0.001 |
| Percentile-Based Extreme Precipitation Analysis | 1 | 0.001 |
| Permutation Feature Importance | 1 | 0.001 |
| PID Control | 1 | 0.001 |
| Poisson Regression | 1 | 0.001 |
| Polynomial Basis Functions | 1 | 0.001 |
| Power Signal Feedback (PSF) Method | 1 | 0.001 |
| Power-Law Regression | 1 | 0.001 |
| Power-Law Regression (Plage–Sunspot Coverage Relation) | 1 | 0.001 |
| Pressure-Implicit with Splitting of Operators (PISO) Algorithm | 1 | 0.001 |
| Probabilistic Cellular Automata (PCA) | 1 | 0.001 |
| Probabilistic Coral Reef Optimization (PCRO-SL) | 1 | 0.001 |
| Probability Density Function (PDF) Analysis | 1 | 0.001 |
| Process-Based Modeling (Custom Respiration Models) | 1 | 0.001 |
| Projection Pursuit Model (PPM) | 1 | 0.001 |
| Pruning of Edges | 1 | 0.001 |
| Pseudo-Absence Sampling | 1 | 0.001 |
| Pseudo-Labeling Algorithm | 1 | 0.001 |
| Python-Based Data Processing | 1 | 0.001 |
| Python-Based Logistic Curve Fitting (4-Parameter, Grid-Based) | 1 | 0.001 |
| Quadratic Classifier | 1 | 0.001 |
| Quadratic Model | 1 | 0.001 |
| Quadratic Programming (QP) Optimization | 1 | 0.001 |
| Quadratic Regression | 1 | 0.001 |
| Quantile Mapping | 1 | 0.001 |
| Quantile Regression Forest (QRF) | 1 | 0.001 |
| Quantile-Based Outlier Removal | 1 | 0.001 |
| Quartic Model | 1 | 0.001 |
| Radial Basis Functions (RBF) | 1 | 0.001 |
| Rank-Based Regression Analysis | 1 | 0.001 |
| Ray-Casting Algorithm | 1 | 0.001 |
| Real-Coded Accelerated Genetic Algorithm (RAGA) | 1 | 0.001 |
| Recurrent Neural Network (RNN) | 1 | 0.001 |
| Reduced-Order Dual Decomposition | 1 | 0.001 |
| Regression Analysis with Control Variables | 1 | 0.001 |
| Relay-Based Switching | 1 | 0.001 |
| Reranking Algorithm | 1 | 0.001 |
| Residual Learning | 1 | 0.001 |
| River Profile Analysis Using Knickpoint Detection | 1 | 0.001 |
| Robustness Checks with Dummy Variable Regression | 1 | 0.001 |
| Robocentric Occupancy Grid Mapping (ROG-Map) | 1 | 0.001 |
| Rule-Based Control Logic | 1 | 0.001 |
| Runge-Kutta Method | 1 | 0.001 |
| Runoff Process Vectorization (RPV) | 1 | 0.001 |
| Safe Flight Corridor (SFC) Generation | 1 | 0.001 |
| Seasonal Autoregressive Integrated Moving Average with eXogenous Variables (SARIMAX) | 1 | 0.001 |
| Seasonal-Trend Decomposition Using LOESS (STL) | 1 | 0.001 |
| Self-Organizing Maps (SOM) | 1 | 0.001 |
| Semi-Analytical Inversion Model (iSAM) | 1 | 0.001 |
| Semi-Implicit Method for Pressure-Linked Equations (SIMPLE) | 1 | 0.001 |
| Signed Distance Transform (SDT) | 1 | 0.001 |
| Simplex Lattice Design | 1 | 0.001 |
| Simulation-Based Rule Curve Optimization | 1 | 0.001 |
| Simultaneous Multiplicative Algebraic Reconstruction Technique (SMART) | 1 | 0.001 |
| Sinusoidal Representation Networks (SIREN) | 1 | 0.001 |
| Skeletonisation Algorithm | 1 | 0.001 |
| Slope–Area Analysis | 1 | 0.001 |
| Simultaneous Multiplicative Algebraic Reconstruction Technique with Tikhonov Regularization (SMART with Tikhonov Reg) | 1 | 0.001 |
| SPACETIME Algorithm | 1 | 0.001 |
| Sparse Linear Method (SLIM) | 1 | 0.001 |
| Spatially-Constrained Clustering (ClustGeo) | 1 | 0.001 |
| Sparse InversE Covariance Estimation (SPIEC-EASI) | 1 | 0.001 |
| Spill Adjustment Optimization via Brent’s Method | 1 | 0.001 |
| Spline Quantile Regression | 1 | 0.001 |
| Split Learning (SL) | 1 | 0.001 |
| Stacked Ensembles (SE) | 1 | 0.001 |
| Statistical Threshold Analysis for TC Genesis | 1 | 0.001 |
| Stochastic and Deterministic ODEs | 1 | 0.001 |
| Swath Profiling | 1 | 0.001 |
| TC Detection via OWZP Method | 1 | 0.001 |
| Temporal Downscaling Algorithm Using Proxy Hydrological Data | 1 | 0.001 |
| Text Analysis Using Keyword-Based Sentence Classification | 1 | 0.001 |
| Time Series Cross-Correlation Analysis | 1 | 0.001 |
| Time-Nonhomogeneous Continuous-Time Markov Chain (CTMC) | 1 | 0.001 |
| Time-Series Temporal Lag Selection | 1 | 0.001 |
| Tracking-Driven Classification Algorithm | 1 | 0.001 |
| Trust Region Methods | 1 | 0.001 |
| Twin Delayed DDPG (TD3) | 1 | 0.001 |
| Two-Stage Robust Optimization | 1 | 0.001 |
| Urban Building Energy Modeling (UBEM) | 1 | 0.001 |
| Variational Inference (VI) | 1 | 0.001 |
| Vector Autoregression (VAR) | 1 | 0.001 |
| Virtual Screening | 1 | 0.001 |
| ViSIR Hybrid Architecture | 1 | 0.001 |
| Voigt Profile Fitting | 1 | 0.001 |
| Volume of Fluid (VOF) Method | 1 | 0.001 |
| Voronoi-Based Skeleton Extraction | 1 | 0.001 |
| Watershed Segmentation | 1 | 0.001 |
| Weighted Sampling for Class Imbalance | 1 | 0.001 |
| Word2Vec | 1 | 0.001 |

---

*For all rows, fpub is calculated as `Total mentions / 687` (rounded to 3 decimal places for readability).*

---
