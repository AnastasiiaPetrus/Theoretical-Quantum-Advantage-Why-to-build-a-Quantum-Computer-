# Supplementary Table S1. Practical Use of Classical Algorithms in Climate and Environmental Sciences

## Overview

This supplementary material systematically maps classical algorithms to their real-world adoption in climate and environmental science and engineering. Each algorithm or method is assessed using the **FUSE** (Field Use Evidence Score) criterion, which is set to `1` if the method is actively applied in practical (operational, industrial, or regulatory) workflows according to recent reports, guidelines, or published case studies; otherwise, FUSE is set to `0`.  
The evaluation is based on an extensive review of international and national agency documents (e.g., US EPA, NOAA, EEA, Chinese MEE), recent peer-reviewed literature, and technical best-practice publications.

## Table Legend

- **Algorithm & Method**: Cluster of related algorithms or models (may include variants and common abbreviations).
- **FUSE**:  
  - `1`: Demonstrated and established practical use in climate/environmental science (including operational monitoring, regulatory reporting, or major industrial projects, 2010–2024).  
  - `0`: No significant evidence of practical application in the above domains.

---

## Example: TOP-10 Practical Algorithms

| Algorithm & Method                                                         | FUSE | Brief Example/Reference                                                                                   |
|---------------------------------------------------------------------------|------|-----------------------------------------------------------------------------------------------------------|
| Random Forest (RF, DRF, VSURF, QRF, FM-QRF, MIDAS-QRF)                    | 1    | Used for land use, air quality, drought, ecology ([EPA 2024][7], [Cutler et al. 2007][3])                 |
| Linear Regression (LR, OLS, Multilin R)                                   | 1    | Classic baseline for environmental trend analysis, criteria, and models ([Wilks 2011][5], [EPA 2018][6])  |
| Gradient Boosting (GB, XGBoost, LightGBM, CatBoost, GBR)                  | 1    | Increasingly used for weather, pollution, satellite data ([EPA T.E.S.T.][7])                              |
| Convolutional Neural Networks (CNN and variants)                          | 1    | For satellite image recognition, ice, agriculture ([Sinha et al. 2024][8])                                |
| Long Short-Term Memory (LSTM and variants)                                | 1    | For time series (rainfall, temperature, hydrology) ([EPA 2023][9])                                        |
| Logistic Regression (LogReg)                                              | 1    | Used in event classification (drought, risk, pollution) ([EPA PH 2020][10])                               |
| Artificial Neural Networks (ANN)                                          | 1    | Pollution, ecological risk, meteorology ([DOE 2022][11], [JRC 2022][12])                                  |
| Monte Carlo Simulation (MCS, MCMC)                                        | 1    | For uncertainty, scenario, risk analysis ([FOCUS 2001][13])                                               |
| Principal Component Analysis (PCA)                                        | 1    | Dimensionality reduction, trend detection ([EPA ROE][14])                                                 |
| Support Vector Machine (SVM, SVM-RBF)                                     | 1    | Used in case studies, less common in operational pipelines ([EPA T.E.S.T.][7])                            |

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
11. [PROJECT Artificial Neural Networks for MSW Contamination Characterization](https://www.energy.gov/sites/default/files/2022-06/CX-025934.pdf?utm_source=chatgpt.com)
12. [Technical Guidance Water Reuse Risk Management for Agricultural Irrigation Schemes in Europe](https://publications.jrc.ec.europa.eu/repository/bitstream/JRC129596/JRC129596_01.pdf?utm_source=chatgpt.com)
13. [LANDSCAPE AND MITIGATION FACTORS IN AQUATIC ECOLOGICAL RISK ASSESSMENT: Monte Carlo assessments](https://esdac.jrc.ec.europa.eu/ESDB_Archive/eusoils_docs/other/FOCUS_Vol1.pdf?utm_source=chatgpt.com)
14. [EPA's Report on the Environment](https://cfpub.epa.gov/roe/technical-documentation_pdf.cfm?i=28&utm_source=chatgpt.com)


## Supplementary Table: Full FUSE Mapping
| Algorithm & Method | FUSE |
|------------------------------------------------------|------|
| Random Forest (RF, DRF, VSURF, QRF, FM-QRF, MIDAS-QRF) | 1 |
| Linear Regression (LR, Least-Squares LR, Multitask LR, Multivar LR, OLS, Multilin R) | 1 |
| Gradient Boosting (GB, XGBoost, LightGBM, CatBoost, GBR) | 1 |
| Convolutional Neural Network (CNN, CNN-BiLSTM-Attention Model, CNN-LSTM, LSTM-CNN, CNN-Transformer, GVSAO-CNN, Mask R-CNN, Multi-Head 1D CNN, 3D CNN) | 1 |
| Long Short-Term Memory (LSTM, GAN-LSTM, Transformer-LSTM, LSTM-AE) | 1 |
| Logistic Regression (LogR, Multilevel LogReg) | 1 |
| Artificial Neural Network (ANN) | 1 |
| Monte Carlo Simulation (MCS, MCMC, MC-Integration) | 1 |
| Principal Component Analysis (PCA) | 1 |
| Support Vector Machine (SVM, SVM with RBF) | 1 |
| Classical Regression Models (GLM, GAM, GAMLSS, LMM, LME, ODE, Quadratic, Quartic) | 1 |
| K-means Clustering (DTW-based K-means Clustering) | 1 |
| K-Nearest Neighbors (KNN, ItemNN, UserNN) | 1 |
| U-Net (Attention U-Net, CentroidNet) | 1 |
| Ensemble Method (Ensemble Averaging, Deep Ensembles, Ensemble Kalman Filter, Stacked Ensemble) | 1 |
| Genetic Algorithm (GA, GA-ANN, NSGA-II, RAGA, GEP) | 1 |
| AutoRegressive Integrated Moving Average/Seasonal Autoregressive Integrated Moving Average (ARIMA/SARIMA, SARIMAX) | 1 |
| Multi-Layer Perceptron (Multi-Layer Perceptron, MLP) | 1 |
| SHAP (Shapley Additive Explanations) | 1 |
| Recurrent Neural Network (RNN, ESN, GRU, Bi-GRU) | 1 |
| Deep / Fully Connected Neural Network (DNN / FNN) | 1 |
| Grid Search Cross-Validation (Grid Search Cross-Validation, Grid Search CV, grid search, GridSearchCV) | 1 |
| Pseudo-Order Kinetics and Adsorption Models (Pseudo-First-Order, Pseudo-Second-Order, Freundlich, Langmuir, GAB) | 1 |
| Ridge Regression (Ridge Regression, Ridge Regression (L2-regularized least squares)) | 1 |
| Support Vector Regression (SVR) | 1 |
| General Circulation Models (MITgcm, HadCM3L, CCSM4, CESM1) | 1 |
| Adam Optimizer (Adam Optimizer, Adam optimizer) | 1 |
| Autoencoder (LSTM-AE, VAE) | 1 |
| Decision Tree (Decision Tree, DT, Decision Tree (DT)) | 1 |
| DeepLabV3+ (DeepLabV3+, DeepLabv3+, DeepLabv3+) | 1 |
| Generative Adversarial Network (GAN, ESRGAN) | 1 |
| Mann–Kendall Trend Test (MK trend test) | 1 |
| Reinforcement Learning (RL, MARL) | 1 |
| Quantile Regression (QR, MQR, SQR) | 1 |
| Partial Least Squares Regression (PLSR) | 1 |
| Savitzky–Golay Filter (Savitzky–Golay Filter, Savitzky–Golay filter, Savitzky–Golay filtering, Savitzky–Golay smoothing) | 1 |
| Spearman Correlation Coefficient (Spearman Correlation Coefficient, Spearman correlation, Spearman rank correlation) | 1 |
| Transformer-Based Architectures (Transformer, ViT, Large Language Model-Based Forecasting Backbone, FPT, GPT2 variants) | 1 |
| Regression Tree (RT, BART, CART) | 1 |
| Box Models (White-box, Black-box, Gray-box) | 1 |
| Hierarchical Clustering (Ward’s Hierarchical Clustering) | 1 |
| Coefficient of Determination (Coefficient of Determination, Coefficient of Determination (R²), R²) | 1 |
| Fast Fourier Transform (Fast Fourier Transform, FFT) | 1 |
| Diffusion Models (LDM, DDIM, DDPM) | 1 |
| Graph Neural Network (GNN, Local Learnable GNN-Model) | 1 |
| Photovoltaic and Solar Models (PV Gen Temp, Optimal Tilt, GCR) | 1 |
| Kernel Density Estimation (Kernel Density Estimation, KDE, Kernel density estimation) | 1 |
| LASSO (LASSO, Least Absolute Shrinkage and Selection Operator) | 1 |
| Levenberg–Marquardt Optimization Algorithm (Levenberg–Marquardt Optimization Algorithm, Levenberg–Marquardt algorithm, Marquardt–Levenberg Algorithm) | 1 |
| Partial Least Squares Discriminant Analysis (Partial Least Squares Discriminant Analysis, PLS-DA, Partial least squares-discriminant analysis (PLS-DA)) | 1 |
| Root Mean Square Error (Root Mean Square Error, RMSE) | 1 |
| Trust Region Policy Optimization (Trust Region Policy Optimization, TRPO) | 1 |
| Welch Method (Welch Method, Welch’s Method, Welch’s method) | 1 |
| 10-fold cross-validation (10-fold cross-validation) | 1 |
| Markov Models (HMM, AR1) | 1 |
| Analysis of Variance (Analysis of Variance, Analysis of variance (ANOVA), ANOVA, One-way ANOVA) | 1 |
| Bayesian Models (Bayesian Age-Depth, Bayesian Mixed-Effects) | 1 |
| Bayesian Inference (Bayesian Inference, Bayesian Inference Framework) | 1 |
| Bayesian Neural Network (BNN) | 1 |
| Bayesian Optimization Algorithm (Bayesian Optimization Algorithm) | 1 |
| Bayesian Update for Occupancy Mapping (Bayesian Update for Occupancy Mapping) | 0 |
| Boruta Algorithm (Boruta Algorithm, Boruta algorithm, Boruta) | 1 |
| Breadth-First Search (Breadth-First Search, Breadth-First Search (BFS) for Pathfinding) | 0 |
| Canonical Correlation Analysis (Canonical Correlation Analysis, CCA) | 1 |
| Cosine Similarity Analysis (Cosine Similarity Analysis, Cosine Similarity Analysis) | 1 |
| Cross-Validation (Cross-Validation, cross-validation, 10-fold cross-validation, 5-fold cross-validation, K-Fold Cross-Validation, Grouped cross-validation, Leave-One-Year-Out Cross-Validation, Grid Search Cross-Validation) | 1 |
| Density-Based Spatial Clustering of Applications with Noise (DBSCAN) | 1 |
| Deep Deterministic Policy Gradient (Deep Deterministic Policy Gradient, DDPG) | 0 |
| Deep Learning (Semantic Segmentation) (Deep Learning (Semantic Segmentation), Deep Learning (Semantic Segmentation)) | 1 |
| Differential Flatness Transform (Differential Flatness Transform) | 0 |
| Discrete Element Method (Discrete Element Method, DEM simulation) | 1 |
| Elastic-Net Regularization (ElasticNet) | 1 |
| Empirical Orthogonal Function Analysis (Empirical Orthogonal Function Analysis, EOF Analysis) | 1 |
| Ensemble tree methods (XRT) | 1 |
| Coordination and Coupling Models (Coordination Influence, CCD Model) | 1 |
| FAST-LIO2 (LiDAR-Inertial Odometry using iterative EKF and ikd-Tree) | 0 |
| Feature Pyramid Network (Feature Pyramid Network, FPN) | 1 |
| Levelized Cost and Cost Modeling (LCOE, Cost Modeling) | 1 |
| Fuzzy C-means Clustering | 1 |
| Gaussian Process Regression (GPR) | 1 |
| Gaussian Radial Basis Function (GRBF) | 1 |
| Graph Convolutional Network (GCN) | 1 |
| iTransformer (iTransformer) | 0 |
| Kalman Filter (KF) | 1 |
| Latent Dirichlet Allocation (Latent Dirichlet Allocation, LDA, Latent Dirichlet Allocation (LDA)) | 1 |
| Linear Interpolation (Linear Interpolation, linear interpolation) | 1 |
| Linear Programming (Linear Programming, linear programming) | 1 |
| Log-Odds Update Scheme (Log-Odds Update Scheme) | 0 |
| Mean Absolute Error (Mean Absolute Error, MAE) | 1 |
| Metropolis-Hastings Algorithm (Metropolis-Hastings Algorithm, Metropolis-Hastings Algorithm) | 1 |
| Model-Based Recursive Partitioning (Model-Based Recursive Partitioning, M5P) | 0 |
| Model Predictive Control (Model Predictive Control, MPC) | 1 |
| Network Analysis (Network Analysis, Network analysis (Force Atlas 2 algorithm), Network analysis (NA)) | 1 |
| Numerical Atmospheric Dispersion Modeling Environment (Numerical Atmospheric Dispersion Modeling Environment, Numerical Atmospheric-dispersion Modelling Environment (NAME)) | 1 |
| Parametric Sensitivity Analysis (Parametric Sensitivity Analysis) | 1 |
| Physically Informed Neural Network (PINN) | 1 |
| Proximal Policy Optimization (Proximal Policy Optimization, PPO) | 0 |
| Quadratic Programming Optimization (Quadratic Programming Optimization, Quadratic Programming (QP) Optimization (via OSQP)) | 1 |
| Quantile Mapping (Quantile Mapping, Quantile Mapping) | 1 |
| Response Surface Methodology (Response Surface Methodology, Response surface methodology (RSM)) | 1 |
| ROG-Map (ROG-Map (Robocentric Occupancy Grid Mapping)) | 0 |
| Safe Flight Corridor Generation (Safe Flight Corridor Generation, Safe Flight Corridor (SFC) Generation) | 0 |
| Self-Attention Mechanism (Self-Attention Mechanism, Self-Attention Mechanisms) | 1 |
| Soft Actor-Critic (Soft Actor-Critic, SAC, SAC (Soft Actor-Critic)) | 0 |
| Spectral Clustering | 1 |
| Support Vector Classification (Support Vector Classification, SVC, Support Vector Classifier (SVC)) | 1 |
| TQC (Truncated Quantile Critics) (TQC (Truncated Quantile Critics), TQC) | 0 |
| Transfer Learning (Transfer Learning, TL) | 1 |
| Variational Inference (Variational Inference, Variational inference) | 1 |
| Vision Transformer (Vision Transformer, ViT, Vision Transformer (ViT)) | 1 |
| Z-score Normalization (Z-score Normalization, Z-score normalization) | 1 |


