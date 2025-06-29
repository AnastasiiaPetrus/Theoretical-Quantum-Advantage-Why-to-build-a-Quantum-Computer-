# Supplementary Table S1. Practical Use of Classical Algorithms in Climate and Environmental Sciences

## Overview

This supplementary material systematically maps classical algorithms to their real-world adoption in climate and environmental science and engineering. Each algorithm or method is assessed using the **fuse** (Field Use Evidence Score) criterion, which is set to `1` if the method is actively applied in practical (operational, industrial, or regulatory) workflows according to recent reports, guidelines, or published case studies; otherwise, fuse is set to `0`.  
The evaluation is based on an extensive review of international and national agency documents (e.g., US EPA, NOAA, EEA, Chinese MEE), recent peer-reviewed literature, and technical best-practice publications.

## Table Legend

- **Algorithm & Method**: Cluster of related algorithms or models (may include variants and common abbreviations).
- **fuse**:  
  - `1`: Demonstrated and established practical use in climate/environmental science (including operational monitoring, regulatory reporting, or major industrial projects, 2020–2025).  
  - `0`: No significant evidence of practical application in the above domains.

---

## Example: TOP-10 Practical Algorithms

| Algorithm & Method                                                         | fuse | Brief Example/Reference                                                                                   |
|---------------------------------------------------------------------------|------|-----------------------------------------------------------------------------------------------------------|
| Random Forest (RF, DRF, VSURF, QRF, FM-QRF, MIDAS-QRF)                    | 1    | Used for land use, air quality, drought, ecology ([EPA 2024][7], [Cutler et al. 2007][3])                 |
| Convolutional Neural Networks (CNN and variants)                          | 1    | For satellite image recognition, ice, agriculture ([Sinha et al. 2024][8])                                |
| Linear Regression (LR, OLS, Multilin R and other variants)                | 1    | Classic baseline for environmental trend analysis, criteria, and models ([Wilks 2011][5], [EPA 2018][6])  |
| Gradient Boosting (GB, XGBoost, LightGBM, CatBoost, GBR and other variants)| 1    | Increasingly used for weather, pollution, satellite data ([EPA T.E.S.T.][7])                              |
| Support Vector Machine (SVM, SVM-RBF and other variants)                  | 1    | Used in case studies, less common in operational pipelines ([EPA T.E.S.T.][7])                            |
| Long Short-Term Memory (LSTM and variants)                                | 1    | For time series (rainfall, temperature, hydrology) ([EPA 2023][9])                                        |
| Principal Component Analysis (PCA)                                        | 1    | Dimensionality reduction, trend detection ([EPA ROE][14])                                                 |
| Logistic Regression (LogReg and variants)                                 | 1    | Used in event classification (drought, risk, pollution) ([EPA PH 2020][10])                               |
| Markov Chain Monte Carlo (MCMC, Metropolis-Hastings, Gibbs Sampling)      | 1    | For uncertainty, scenario, risk analysis ([FOCUS 2001][13])                                               |
| Transformer (MaskFormer, DETR, and other variants)                        | 1    | ...                                  |

#### References (see section below for details)

1. [Random forest regression models in ecology: Accounting for messy biological data and producing predictions with uncertainty](https://meetings.pices.int/publications/Presentations/PICES-2024/S3-Akselrud.pdf)
2. [Predicting Fishing Effort and Catch Using Semantic Trajectories and Machine Learning](https://link.springer.com/chapter/10.1007/978-3-030-38081-6_7)
3. Cutler, D. R. et al. (2007). Random forests for classification in ecology. *Ecology*, 88(11), 2783–2792.
4. [Site-Specific Water Quality Data Better Defines Aluminum Aquatic Toxicity](https://www.scsengineers.com/site-specific-water-quality-data-better-defines-aluminum-aquatic-toxicity/?utm_source=chatgpt.com)
5. Wilks, D. S. (2011). *Statistical Methods in the Atmospheric Sciences*.
6. [AQUATIC LIFE AMBIENT WATER QUALITY CRITERIA FOR ALUMINUM - 2018](https://ris.dls.virginia.gov/uploads/9VAC25/dibr/Final%20Aquatic%20Life%20Ambient%20Water%20Quality%20Criteria%20for%20Aluminum%202018-20220104142527.pdf?utm_source=chatgpt.com)
7. [Overview of T.E.S.T. (Toxicity Estimation Software Tool)](https://www.epa.gov/system/files/documents/2024-06/test_508.pdf?utm_source=chatgpt.com)
8. [A Review of CNN Applications in Smart Agriculture Using Multimodal Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC11768470/?utm_source=chatgpt.com)
9. [Ammonia Emissions Enhancements with Deep Learning CTM (Forward-Backward) and Remote-sensing Observations](https://www.epa.gov/system/files/documents/2023-11/240pm_baek.pdf?utm_source=chatgpt.com)
10. [Development of Logistic Regression Models for Portland Harbor](https://semspub.epa.gov/work/10/100013730.pdf?utm_source=chatgpt.com)
11. ...
12. ...
13. [LANDSCAPE AND MITIGATION FACTORS IN AQUATIC ECOLOGICAL RISK ASSESSMENT: Monte Carlo assessments](https://esdac.jrc.ec.europa.eu/ESDB_Archive/eusoils_docs/other/FOCUS_Vol1.pdf?utm_source=chatgpt.com)
14. [EPA's Report on the Environment](https://cfpub.epa.gov/roe/technical-documentation_pdf.cfm?i=28&utm_source=chatgpt.com)


## Supplementary Table: Full fuse Mapping
| Algorithm & Method | fuse |
|------------------------------------------------------|------|
| Distributed Random Forest (DRF, RF, RFR, VSURF, MIDAS-QRF, FM-QRF, Factor MIDAS-QRF) | 1 |
| Convolutional Neural Network (CNN, 3D-CNN, Deterministic CNN, EfficientNet, GVSAO-CNN, LSTM-CNN, Mask R-CNN, Multi-Head 1D CNN, SRCNN, CNN-BiLSTM, DeepLabV3+, PSPNet, U-Net, Attention U-Net, ResNet-34, Xception-65) | 1 |
| Linear Regression Methods (LR, LSLR, MLR, Multivariate Linear Regression, Linear Quantile Mixed Model, Segmented Regression, Power-Law Regression, Quantile Regression) | 1 |
| Gradient Boosting Methods (GB, GBM, GBR, ET, XRT, XGBoost, CatBoost, LightGBM) | 1 |
| Support Vector Machine (SVM, SVC, SVR, IPSO‑SVM, PSO‑SVM, SVM with RBF Kernel) | 1 |
| Long Short-Term Memory (LSTM, CNN-LSTM, Transformer-LSTM, GAN-LSTM, LSTM-AE, TimesNet, Informer) | 1 |
| Principal Component Analysis (PCA) | 1 |
| Logistic Regression Models (Logistic Regression, Multilevel Logistic Regression, Multinomial Logit Model) | 1 |
| Markov Chain Monte Carlo (MCMC, Metropolis-Hastings, Gibbs Sampling) | 1 |
| Transformer (MaskFormer, DETR, iTransformer, BEiT, DINOv2, ViT, Transformer Encoder, CNN-Transformer) | 1 |
| K-Means Clustering (K-Means, DTW-Based K-Means, K-Means Co-Clustering) | 1 |
| Artificial Neural Network (ANN, FNN, GRNN, GA-ANN, PSO-ANN) | 1 |
| Bayesian Statistical Methods (BART, BNP-PC, Bayesian Age-Depth Modeling, Bayesian Mixed-Effects Modeling, Bayesian Inversion, Bayesian Update for Occupancy Mapping, Bayesian Optimization Algorithm) | 1 |
| Reinforcement Learning (RL, MARL, REINFORCE, IPPO, PPO, TRPO, Actor-Critic, A2C, DDPG, SAC, TD3, DPG, TQC) | 1 |
| Decision Tree Methods (CART, DT, RT, M5P) | 1 |
| Independent Proximal Policy Optimization (IPPO, PPO, TRPO, REINFORCE) | 1 |
| Multi-Layer Perceptron (MLP) | 1 |
| Neural Network (NN) | 1 |
| Partial Least Squares (PLS-DA, PLSR) | 1 |
| SHapley Additive Explanations (SHAP) | 1 |
| Fuzzy Methods (Fuzzy C-Means, IVFS, Spatial Fuzzy Kappa, Fuzzy Overlay Analysis) | 1 |
| Gaussian Methods (Gaussian Graphical Models, Gaussian Processes, GPR, GRBF) | 1 |
| Generative Adversarial Network (GAN, ESRGAN, SRGAN) | 1 |
| Grid Search (GridSearchCV, Random GridSearchCV) | 1 |
| K-Nearest Neighbors (KNN, ItemNN, UserNN) | 1 |
| Adam Optimizer | 1 |
| Levenberg–Marquardt (LM) Optimization Algorithm | 1 |
| Monte Carlo Simulation | 1 |
| Savitzky–Golay Filter | 1 |
| Actor-Critic (Actor-Critic, A2C, DDPG, DPG, SAC, TD3) | 1 |
| AutoRegressive Integrated Moving Average (ARIMA, SARIMA, SARIMAX) | 1 |
| Autoencoder (AE, VAE) | 1 |
| DeepLabV3+ | 1 |
| Ensemble Kalman Filter (EnKF) | 1 |
| Graph Neural Networks (GCN, GNN, Local Learnable GNN-Model) | 1 |
| Hierarchical Clustering (Ward’s Hierarchical Clustering, Hierarchical Frequency-Based Location Estimation) | 1 |
| K-Fold Cross-Validation (KFCV) | 1 |
| Kernel Density Estimation (KDE) | 1 |
| Ridge Regression | 1 |
| Analysis of Variance (ANOVA) | 1 |
| Bayesian Neural Network (BNN) | 1 |
| Bi-Spectral Method | 1 |
| Boruta Algorithm | 1 |
| Copulas for Multivariate Hazard Modeling | 1 |
| Deep Neural Network (DNN) | 1 |
| Diffusion Models (DDIM, DDPM, LDM) | 1 |
| Dynamic Time Warping (DTW, TWDTW) | 1 |
| Empirical Orthogonal Function (EOF) | 1 |
| Fast Fourier Transform (FFT) | 1 |
| Feature Pyramid Network (FPN) | 1 |
| Gated Recurrent Unit (GRU) | 1 |
| Generalized Additive Models (GAM, GAMLSS) | 1 |
| Generalized Linear Model (GLM) | 1 |
| Grey Wolf Optimizer (GWO, IGWO) | 1 |
| Kalman Filter (KF) | 1 |
| Limited-memory Broyden–Fletcher–Goldfarb–Shanno Optimization Algorithm (L-BFGS, L-BFGS-B) | 1 |
| Lagrangian Particle Trajectory Tracking | 1 |
| Latent Dirichlet Allocation (LDA) | 1 |
| Least Absolute Shrinkage and Selection Operator (LASSO) | 1 |
| Linear Interpolation | 1 |
| Linear Programming | 1 |
| Morphological Thinning | 0 |
| Partial Regression Analysis | 1 |
| Physically Informed Neural Network (PINN) | 1 |
| Random Grid Search (GridSearchCV, Random GridSearchCV) | 1 |
| Random Subspace Method (RSS) | 1 |
| Response Surface Methodology (RSM) | 1 |
| Seasonal Autoregressive Integrated Moving Average (SARIMA) | 1 |
| Self-Attention Mechanism (Multihead Self-Attention, Self-Attention) | 1 |
| Singular Value Decomposition (SVD, SVD++) | 1 |
| Spectral Clustering | 1 |
| Transfer Learning | 1 |
| Welch Method | 1 |
| 4D-Var (Four-Dimensional Variational Data Assimilation) | 1 |
| Accumulated Latitudinal Dispersal Calculation | 0 |
| AdaBoost | 1 |
| Adams-Bashforth Two-Step Time Integration | 1 |
| Adaptive Combiner | 0 |
| Adaptive Multiple Importance Sampling (AMIS) | 0 |
| Adaptive Neuro-Fuzzy Inference System (ANFIS) | 1 |
| Additive Regression (AR) | 1 |
| Adjoint Method | 1 |
| Advection–Diffusion Modeling | 1 |
| Adversarial Loss | 0 |
| AI-Supported Object Detection Pipeline (Head-Tail Segmentation & Classification) | 0 |
| Alternating Direction Method of Multipliers (ADMM) | 1 |
| Analysis of Compositions of Microbiomes with Bias Correction (ANCOM-BC2) | 0 |
| AR(1) Model | 1 |
| Atrous Spatial Pyramid Pooling (ASPP) | 0 |
| Augmented Dickey-Fuller Test (ADF) | 1 |
| Augmented Random Search (ARS) | 0 |
| Backward Selection Using Akaike Information Criterion (AIC) | 1 |
| Backward Trajectory Analysis | 1 |
| Bagging (Bootstrap Aggregating) | 1 |
| Best Subset Regression Analysis | 0 |
| Bicubic Interpolation | 1 |
| Bidirectional Gated Recurrent Units (Bi-GRU) | 1 |
| Binary Dynamic Programming | 0 |
| Binomial Likelihood Modeling | 0 |
| Boolean Matching | 0 |
| Bootstrap Resampling | 1 |
| Breadth-First Search (BFS) | 0 |
| Breaks for Additive Season and Trend (BFAST) | 1 |
| Canonical Correlation Analysis (CCA) | 1 |
| CaViaR (SAV, AD, AS, IG) | 0 |
| Central Composite Design (CCD) | 1 |
| Centroid-Based Instance Segmentation | 0 |
| Change Vector Analysis (CVA) | 1 |
| CHARMm Algorithm | 0 |
| Climate Match Algorithm | 0 |
| Climate Niche Envelope Approach | 1 |
| Clustal Omega | 0 |
| Cluster Analysis | 1 |
| Clustering Algorithm for Macro- and Meso-Segment Detection | 0 |
| Cobweb Model | 0 |
| Column-and-Constraint Generation Algorithm (C&CGA) | 0 |
| Comparative Molecular Field Analysis (CoMFA) | 0 |
| Compartmental Dynamical Thermodynamics / TMN Modeling | 0 |
| Conditional Autoregressive (CAR) Model | 0 |
| Conditional Logistic Regression | 1 |
| Consistency Distillation | 0 |
| Consistency Models (CM) | 0 |
| Constrained Delaunay Triangulation (CDT) | 0 |
| Constraint-Driven Optimization | 0 |
| Contrastive Learning (InfoNCE Loss) | 0 |
| Correction Method for Population Weighting | 1 |
| DADA2 (High-Resolution Sample Inference) | 0 |
| Deep Ensembles | 1 |
| Delaunay Triangulation (via JIGSAW) | 1 |
| Denoising Diffusion Implicit Models (DDIM) | 1 |
| Denoising Diffusion Probabilistic Models (DDPM) | 1 |
| Density Functional Theory (DFT) Calculations | 0 |
| Density-Based Spatial Clustering of Applications with Noise (DBSCAN) | 1 |
| Desirability Function Approach | 0 |
| Differential Flatness Transform | 0 |
| Dirichlet Process Mixture Model (DPM) | 0 |
| DiRienzo–Zurbenko Algorithm (DZA) | 0 |
| Discrete Element Method (DEM) Simulation | 1 |
| Dispersal–Extinction–Cladogenesis (DEC) Model | 0 |
| Distance-Based Redundancy Analysis (dbRDA) | 0 |
| DistilBERT | 0 |
| Distorted Born Approximation (DBA) | 0 |
| DLinear | 0 |
| DNA Sequence Alignment | 0 |
| DOMAIN Algorithm | 0 |
| Dropout Regularization | 0 |
| Dynamic Dewpoint Isotherm (DDI) Method | 0 |
| Dynamic MIDAS-QRF | 0 |
| Dynamic Programming | 1 |
| Edge-Greedy Search for Causal Graph Construction | 0 |
| eDMD with Dictionary Learning (eDMD-DL) | 0 |
| ElasticNet | 1 |
| Empirical Mode Decomposition (EMD) | 1 |
| End-Member Modeling Algorithm (EMMA) | 1 |
| Energy and Mass Conservation Equations | 1 |
| Ensemble Averaging | 1 |
| Entropy Weight Method | 0 |
| Enzymatic Catalysis Modelling | 1 |
| Equal-Area Quadratic Spline | 1 |
| Ewald Summation | 0 |
| Exergy Analysis | 1 |
| Expectation-Maximization (EM) Algorithm | 1 |
| Experience Replay | 0 |
| Exponential Smoothing | 1 |
| Extended Dynamic Mode Decomposition (eDMD) | 0 |
| Extended Fourier Amplitude Sensitivity Testing (EFAST) | 0 |
| Extreme Learning Machine (ELM) | 1 |
| Extremely Randomized Trees (XRT) | 1 |
| Fama–French Three-Factor and Five-Factor Model Regressions | 0 |
| Fast LiDAR-Inertial Odometry 2 (FAST-LIO2) | 0 |
| Data Fusion Methods (Data-Level Fusion, Decision-Level Fusion, Feature-Level Fusion) | 1 |
| Finite Differences Simulator | 1 |
| Finite Volume Method (FVM) | 1 |
| Fully Connected Neural Network (FNN) | 1 |
| Galerkin Method | 1 |
| Gamma Regression | 1 |
| Generalized Autoregressive Conditional Heteroskedasticity / GARCH-Mixed Data Sampling (GARCH / GARCH-MIDAS) | 0 |
| Gene Expression Programming (GEP) | 1 |
| Generalized Estimating Equations (GEE) | 1 |
| Generalized Likelihood Uncertainty Estimation (GLUE) | 1 |
| Generalized Three-Cornered Hat (TCH) Method | 0 |
| Genetic Algorithm (GA, NSGA-II, RAGA) | 1 |
| Geodetector Model (GDM) | 1 |
| GF‑SG Fusion Algorithm | 0 |
| GIS-Based Spatial Multi-Criteria Evaluation | 1 |
| Graphical LASSO (gLASSO) | 1 |
| Gravity Search Optimization Algorithm (GVSAO) | 0 |
| Grid Interpolation (Linear) | 1 |
| Group-Based Trajectory Modeling | 0 |
| Harmonic Tidal Analysis | 1 |
| Harmony Search | 1 |
| Heat Transfer Algorithm | 1 |
| Hidden Markov Model (HMM) | 1 |
| Hilbert Spectral Analysis | 1 |
| Hobday-Based Anomaly Detection Algorithm | 1 |
| Homology Modeling | 0 |
| Improved 1-D DAS Method | 0 |
| Improved 2-D DAS Method | 0 |
| Incremental kd-Tree (ikd-Tree) | 0 |
| Indicator Species Analysis (IndVal.g) | 1 |
| Initial-Condition-Dependent Finite-Time Stabilizing Controller | 0 |
| Integral Equation Method (IEM) | 1 |
| Interferometric Coherence Techniques | 1 |
| Interquartile Range (IQR)-Based Outlier Detection | 1 |
| Inverse Modelling (Parameter Optimization) | 1 |
| Isotonic Regression | 1 |
| Item Nearest Neighbor (ItemNN) | 1 |
| Iterative Optimization | 1 |
| Jenks Natural Breaks Classification | 1 |
| K-Shape Clustering | 1 |
| Kaplan–Meier Survival Analysis | 1 |
| KDTree-Based Spatial Grouping | 1 |
| Kernelized Hypothesis Testing (HSIC) | 0 |
| Kolmogorov–Smirnov Test (KS Test) | 1 |
| Kolmogorov–Zurbenko (KZ) Filter | 1 |
| Kolmogorov–Zurbenko (KZ) Periodogram | 0 |
| Kriging (Spatial Interpolation) | 1 |
| L-Curve Method | 1 |
| L2 Regularization | 1 |
| Laplace Approximation | 1 |
| LAVD-Based Eddy Detection Algorithm | 1 |
| Least Squares Inversion with Smoothness-Constrained Regularization | 1 |
| Leave-One-Year-Out Cross-Validation | 1 |
| Linear Fitting | 1 |
| Linear Mixed Effect Model (REML) | 1 |
| Linear Mixed Models (LMM) | 1 |
| Linear Quantile Mixed Model (LQMM) | 1 |
| Linearized Multi-Block ADMM with Regularization | 0 |
| Local Outlier Factor (LOF) | 1 |
| Log-Odds Update Scheme | 1 |
| Lossless Compression Algorithm (ZIP) | 1 |
| Lossy Compression Algorithm (ZFP) | 1 |
| Locally Weighted Scatterplot Smoothing (LOWESS) | 1 |
| LSTM Autoencoder (LSTM-AE) | 1 |
| Manual Thresholding | 1 |
| Marker-Assisted Selection | 0 |
| Markov Decision Process (MDP) | 0 |
| Mass Balance Modeling | 1 |
| Mass Curve Technique (MCT) | 1 |
| Match Climates Regional Algorithm | 0 |
| Maximum Likelihood Classification (MLC) | 1 |
| Maximum Power Point Tracking (MPPT) Control | 0 |
| Maximum-Likelihood Phylogenetic Analysis | 0 |
| MIDAS Quantile Random Forest (MIDAS-QRF) | 1 |
| Migration Event Detection Algorithm | 0 |
| Minimum Bounding Rectangle | 1 |
| MIR Method | 0 |
| Mixed Data Sampling (MIDAS) | 1 |
| Mixed-Effects Modeling (LME4) | 1 |
| Mixed-Finite Element Scheme | 1 |
| Mixed-Integer Linear Programming (MILP) | 1 |
| Mixture Density Networks (MDN) | 0 |
| Model Predictive Control (MPC) | 0 |
| Model-Agnostic Meta-Learning (MAML) | 0 |
| Modified Response Matrix Method | 0 |
| Moffat Uncertainty Analysis Method | 0 |
| Molecular Docking | 0 |
| Molecular Dynamics Simulation | 0 |
| Monte Carlo Dropout (MC-Dropout) | 1 |
| Monte Carlo Integration | 1 |
| Moving Average Filter | 1 |
| Multi-Agent Reinforcement Learning (MARL) | 0 |
| Multi-Criteria Performance Evaluation | 1 |
| Multi-Label Machine Learning | 1 |
| Multi-Objective Evolutionary Algorithm (MOEA) | 1 |
| Multi-Objective Optimization | 1 |
| Multi-Task Linear Regression (MTLR) | 0 |
| Multihead Self-Attention Mechanism | 1 |
| Multilevel Logistic Regression (MLLogR) | 1 |
| Multilinear Regression (MLR) | 1 |
| Multinomial Logit Model | 1 |
| Multiscale Geographically Weighted Regression (MGWR) | 1 |
| Multivariate Linear Regression | 1 |
| Multivariate Quantiles and Multiple-Output Regression Quantiles | 1 |
| Multivariate Regression | 1 |
| Naive Bayes | 1 |
| Negative Binomial Mixed-Effects Models | 1 |
| Neighbor-Joining Method | 0 |
| Network-Based Path Filtering | 0 |
| Neural Network Operators (Kantorovich) | 0 |
| Neural Network Operators (Classical) | 0 |
| Neural Network Classifier (NN Classifier) | 1 |
| Non-Dominated Sorting Genetic Algorithm II (NSGA-II) | 1 |
| Non-Metric Multidimensional Scaling (NMDS) | 1 |
| Nonlinear Regression (nlsLM Package) | 1 |
| Nonparanormal Transformation | 0 |
| Normalization of Raster Data | 1 |
| Optical Flow Method | 1 |
| Ordinary Differential Equation (ODE) System Modelling | 1 |
| Parallel Factor Analysis (PARAFAC) Modeling | 1 |
| Partial Correlation Analysis | 1 |
| Pruned Exact Linear Time Algorithm (PELT) | 1 |
| Percentile-Based Extreme Precipitation Analysis | 1 |
| Permutation Feature Importance | 1 |
| PID Control | 0 |
| Poisson Regression | 1 |
| Polynomial Basis Functions | 1 |
| Power Signal Feedback (PSF) Method | 0 |
| Power-Law Regression | 1 |
| Power-Law Regression (Plage–Sunspot Coverage Relation) | 0 |
| Pressure-Implicit with Splitting of Operators (PISO) Algorithm | 1 |
| Probabilistic Cellular Automata (PCA) | 1 |
| Probabilistic Coral Reef Optimization (PCRO-SL) | 0 |
| Probability Density Function (PDF) Analysis | 1 |
| Process-Based Modeling (Custom Respiration Models) | 1 |
| Projection Pursuit Model (PPM) | 0 |
| Pruning of Edges | 0 |
| Pseudo-Absence Sampling | 1 |
| Pseudo-Labeling Algorithm | 0 |
| Python-Based Data Processing | 1 |
| Python-Based Logistic Curve Fitting (4-Parameter, Grid-Based) | 1 |
| Quadratic Classifier | 1 |
| Quadratic Model | 1 |
| Quadratic Programming (QP) Optimization | 1 |
| Quadratic Regression | 1 |
| Quantile Mapping | 1 |
| Quantile Regression Forest (QRF) | 1 |
| Quantile-Based Outlier Removal | 1 |
| Quartic Model | 1 |
| Radial Basis Functions (RBF) | 1 |
| Rank-Based Regression Analysis | 1 |
| Ray-Casting Algorithm | 0 |
| Real-Coded Accelerated Genetic Algorithm (RAGA) | 1 |
| Recurrent Neural Network (RNN) | 1 |
| Reduced-Order Dual Decomposition | 0 |
| Regression Analysis with Control Variables | 1 |
| Relay-Based Switching | 0 |
| Reranking Algorithm | 0 |
| Residual Learning | 1 |
| River Profile Analysis Using Knickpoint Detection | 1 |
| Robustness Checks with Dummy Variable Regression | 1 |
| Robocentric Occupancy Grid Mapping (ROG-Map) | 0 |
| Rule-Based Control Logic | 1 |
| Runge-Kutta Method | 1 |
| Runoff Process Vectorization (RPV) | 0 |
| Safe Flight Corridor (SFC) Generation | 0 |
| Seasonal Autoregressive Integrated Moving Average with eXogenous Variables (SARIMAX) | 1 |
| Seasonal-Trend Decomposition Using LOESS (STL) | 1 |
| Self-Organizing Maps (SOM) | 1 |
| Semi-Analytical Inversion Model (iSAM) | 0 |
| Semi-Implicit Method for Pressure-Linked Equations (SIMPLE) | 1 |
| Signed Distance Transform (SDT) | 0 |
| Simplex Lattice Design | 1 |
| Simulation-Based Rule Curve Optimization | 1 |
| Simultaneous Multiplicative Algebraic Reconstruction Technique (SMART) | 0 |
| Sinusoidal Representation Networks (SIREN) | 0 |
| Skeletonisation Algorithm | 0 |
| Slope–Area Analysis | 1 |
| Simultaneous Multiplicative Algebraic Reconstruction Technique with Tikhonov Regularization (SMART with Tikhonov Reg) | 0 |
| SPACETIME Algorithm | 0 |
| Sparse Linear Method (SLIM) | 0 |
| Spatially-Constrained Clustering (ClustGeo) | 1 |
| Sparse InversE Covariance Estimation (SPIEC-EASI) | 0 |
| Spill Adjustment Optimization via Brent’s Method | 0 |
| Spline Quantile Regression | 1 |
| Split Learning (SL) | 0 |
| Stacked Ensembles (SE) | 1 |
| Statistical Threshold Analysis for TC Genesis | 1 |
| Stochastic and Deterministic ODEs | 1 |
| Swath Profiling | 1 |
| TC Detection via OWZP Method | 0 |
| Temporal Downscaling Algorithm Using Proxy Hydrological Data | 1 |
| Text Analysis Using Keyword-Based Sentence Classification | 1 |
| Time Series Cross-Correlation Analysis | 1 |
| Time-Nonhomogeneous Continuous-Time Markov Chain (CTMC) | 0 |
| Time-Series Temporal Lag Selection | 1 |
| Tracking-Driven Classification Algorithm | 0 |
| Trust Region Methods | 1 |
| Twin Delayed DDPG (TD3) | 0 |
| Two-Stage Robust Optimization | 1 |
| Urban Building Energy Modeling (UBEM) | 1 |
| Variational Inference (VI) | 1 |
| Vector Autoregression (VAR) | 1 |
| Virtual Screening | 0 |
| ViSIR Hybrid Architecture | 0 |
| Voigt Profile Fitting | 1 |
| Volume of Fluid (VOF) Method | 1 |
| Voronoi-Based Skeleton Extraction | 0 |
| Watershed Segmentation | 1 |
| Weighted Sampling for Class Imbalance | 1 |
| Word2Vec | 1 |


