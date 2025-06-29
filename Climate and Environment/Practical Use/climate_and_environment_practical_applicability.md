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
| Algorithm & Method                                                                                                                                                            | fuse |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| Distributed Random Forest (DRF, RF, RFR, VSURF, MIDAS-QRF…)                                                                                                                 | 1    |
| Convolutional Neural Network (CNN, Mask R-CNN, U‑Net, ResNet‑34, EfficientNet…)                                                                                              | 1    |
| Linear Regression Methods (LR, MLR, Quantile Regression…)                                                                                                                   | 1    |
| Gradient Boosting (XGBoost, LightGBM, CatBoost…)                                                                                                                             | 1    |
| Support Vector Machine (SVM, SVR, IPSO‑SVM…)                                                                                                                                  | 1    |
| Long Short‑Term Memory (LSTM, LSTM‑AE, CNN‑LSTM…)                                                                                                                             | 1    |
| Principal Component Analysis (PCA)                                                                                                                                           | 1    |
| Logistic Regression (Logistic Regression, Multilevel Logit Model…)                                                                                                          | 1    |
| Markov Chain Monte Carlo (MCMC…)                                                                                                                                             | 1    |
| Transformer (ViT, DETR, MaskFormer и др.)                                                                                                                                     | 1    |
| K‑Means Clustering (K‑Means, DTW‑based…)                                                                                                                                      | 0    |
| Artificial Neural Network (ANN, GRNN, GA‑ANN…)                                                                                                                               | 1    |
| Bayesian Statistical Methods (BART, Bayesian Mixed‑Effects…)                                                                                                                 | 1    |
| Reinforcement Learning (PPO, SAC, MARL…)                                                                                                                                      | 0    |
| Decision Tree Methods (CART, M5P…)                                                                                                                                           | 1    |
| Multi‑Layer Perceptron (MLP), Neural Network (NN)                                                                                                                            | 1    |
| Partial Least Squares (PLS-DA, PLSR)                                                                                | 1    |
| SHapley Additive Explanations (SHAP)                                                                                | 1    |
| Fuzzy Methods (Fuzzy C‑Means, IVFS, Spatial Fuzzy Kappa, Fuzzy Overlay Analysis)                                  | 0    |
| Gaussian Methods (Gaussian Graphical Models, Gaussian Processes, GPR, GRBF)                                        | 1    |
| Generative Adversarial Network (GAN, ESRGAN, SRGAN)                                                                 | 0    |
| Grid Search (GridSearchCV, Random GridSearchCV)                                                                     | 1    |
| K‑Nearest Neighbors (KNN, ItemNN, UserNN)                                                                         | 1    |
| Adam Optimizer                                                                                                     | 1    |
| Levenberg–Marquardt Optimization Algorithm                                                                         | 0    |
| Monte Carlo Simulation                                                                                             | 1    |
| Savitzky–Golay Filter                                                                                               | 0    |
| Actor‑Critic (A2C, DDPG, SAC, TD3…)                                                                                 | 0    |
| AutoRegressive Integrated Moving Average (ARIMA, SARIMA, SARIMAX)                                                  | 1    |
| Autoencoder (AE, VAE)                                                                                               | 1    |
| DeepLabV3+                                                                                                          | 1    |
| Ensemble Kalman Filter (EnKF)                                                                                       | 1    |
| Graph Neural Networks (GCN, GNN…)                                                                                   | 1    |
| Hierarchical Clustering (Ward’s Hierarchical Clustering…)                                                          | 0    |
| K‑Fold Cross‑Validation (KFCV)                                                                                      | 1    |
| Kernel Density Estimation (KDE)                                                                                     | 1    |
| Ridge Regression                                                                                                    | 1    |
| Analysis of Variance (ANOVA)                                                                                        | 1    |
| Bayesian Neural Network (BNN)                                                                                       | 0    |
| Bi‑Spectral Method                                                                                                  | 0    |
| Boruta Algorithm                                                                                                    | 0    |
| Copulas for Multivariate Hazard Modeling                                                      | 1    |
| Deep Neural Network (DNN)                                                                     | 1    |
| Diffusion Models (DDIM, DDPM, LDM)                                                            | 0    |
| Dynamic Time Warping (DTW, TWDTW)                                                             | 0    |
| Empirical Orthogonal Function (EOF)                                                           | 1    |
| Fast Fourier Transform (FFT)                                                                  | 1    |
| Feature Pyramid Network (FPN)                                                                 | 0    |
| Gated Recurrent Unit (GRU)                                                                    | 1    |
| Generalized Additive Models (GAM, GAMLSS)                                                     | 1    |
| Generalized Linear Model (GLM)                                                                | 1    |
| Grey Wolf Optimizer (GWO, IGWO)                                                               | 0    |
| Kalman Filter (KF)                                                                            | 1    |
| L-BFGS Optimization Algorithm                                                                 | 0    |
| Lagrangian Particle Trajectory Tracking                                                       | 1    |
| Latent Dirichlet Allocation (LDA)                                                             | 0    |
| LASSO (Least Absolute Shrinkage and Selection Operator)                                       | 1    |
| Linear Interpolation                                                                          | 1    |
| Linear Programming                                                                            | 0    |
| Morphological Thinning                                                                        | 0    |
| Partial Regression Analysis                                                                   | 1    |
| Physically Informed Neural Network (PINN)                                                     | 0    |
| Random Grid Search (GridSearchCV, Random GridSearchCV)                                        | 1    |
| Random Subspace Method (RSS)                                                                  | 0    |
| Response Surface Methodology (RSM)                                                            | 0    |
| SARIMA (Seasonal ARIMA)                                                                      | 1    |
| Self-Attention Mechanism (Multihead Self‑Attention)                                           | 1    |
| Singular Value Decomposition (SVD, SVD++)                                                           | 1    |
| Spectral Clustering                                                                                 | 0    |
| Transfer Learning                                                                                    | 1    |
| Welch Method                                                                                        | 0    |
| Four‑Dimensional Variational Data Assimilation (4D‑Var)                                              | 1    |
| Accumulated Latitudinal Dispersal Calculation                                                       | 0    |
| AdaBoost                                                                                            | 1    |
| Adams‑Bashforth Two‑Step Time Integration                                                            | 0    |
| Adaptive Combiner                                                                                   | 0    |
| Adaptive Multiple Importance Sampling (AMIS)                                                         | 0    |
| ANFIS (Adaptive Neuro‑Fuzzy Inference System)                                                        | 0    |
| Additive Regression (AR)                                                                            | 1    |
| Adjoint Method                                                                                      | 0    |
| Advection–Diffusion Modeling                                                                        | 1    |
| Adversarial Loss                                                                                    | 0    |
| AI‑Supported Object Detection Pipeline (Head‑Tail Segmentation & Classification)                    | 0    |
| Alternating Direction Method of Multipliers (ADMM)                                                   | 0    |
| ANCOM‑BC2 (Microbiome composition bias correction)                                                 | 0    |
| AR(1) Model                                                                                          | 1    |
| Atrous Spatial Pyramid Pooling (ASPP)                                                                | 0    |
| Transformer (Encoder, CNN‑Transformer, DETR, iTransformer, ViT, BEiT, DINOv2)                        | 1    |
| Augmented Dickey‑Fuller Test (ADF)                                                                  | 1    |
| Augmented Random Search (ARS)                                                                       | 0    |
| Backward Selection with AIC                                                                         | 1    |
| Backward Trajectory Analysis                                                                        | 1    |
| Bagging (Bootstrap Aggregating)                                                                     | 1    |
| Best Subset Regression Analysis                                                                     | 1    |
| Bicubic Interpolation                                                                               | 1    |
| Bidirectional GRU (Bi‑GRU)                                                                          | 1    |
| Binary Dynamic Programming                                                                          | 0    |
| Binomial Likelihood Modeling                                                                       | 0    |
| Boolean Matching                                                                                   | 0    |
| Bootstrap Resampling                                                                                | 1    |
| Breadth‑First Search (BFS)                                                                         | 0    |
| BFAST (Breaks for Additive Season and Trend)                                                       | 1    |
| Canonical Correlation Analysis (CCA)                                                               | 1    |
| CaViaR (SAV, AD, AS, IG)                                                                            | 0    |
| Central Composite Design (CCD)                                                                     | 0    |
| Centroid‑Based Instance Segmentation                                                                | 0    |
| Change Vector Analysis (CVA)                                                                       | 1    |
| CHARMm Algorithm                                                                                   | 0    |
| Climate Match Algorithm                                                                            | 0    |
| Climate Niche Envelope Approach                                                                    | 1    |
| Clustal Omega                                                                                      | 0    |
| Cluster Analysis                                                                                   | 0    |
| Macro‑/Meso‑Segment Clustering                                                                     | 0    |
| Cobweb Model                                                                                       | 0    |
| Column‑and‑Constraint Generation Algorithm (C&CGA)                                                 | 0    |
| CoMFA                                                                                              | 0    |
| Compartmental Dynamical TMN Modeling                                                               | 0    |
| Conditional Autoregressive (CAR) Model                                                            | 1    |
| Conditional Logistic Regression                                                                     | 1    |
| Consistency Distillation                                                                            | 0    |
| Consistency Models (CM)                                                                            | 0    |
| Constrained Delaunay Triangulation (CDT)                                                           | 0    |
| Constraint‑Driven Optimization                                                                      | 0    |
| Contrastive Learning (InfoNCE Loss)                                                                | 0    |
| Population‑Weighting Correction Method                                                             | 0    |
| DADA2                                                                                              | 0    |
| Deep Ensembles                                                                                     | 1    |
| Delaunay Triangulation (via JIGSAW)                                                                       | 1    |
| Denoising Diffusion Implicit Models (DDIM)                                                               | 0    |
| Denoising Diffusion Probabilistic Models (DDPM)                                                           | 0    |
| Density Functional Theory (DFT) Calculations                                                              | 0    |
| DBSCAN (Density-Based Spatial Clustering of Applications with Noise)                                     | 1    |
| Desirability Function Approach                                                                            | 0    |
| Differential Flatness Transform                                                                           | 0    |
| Dirichlet Process Mixture Model (DPM)                                                                     | 0    |
| DiRienzo–Zurbenko Algorithm (DZA)                                                                         | 0    |
| Discrete Element Method (DEM) Simulation                                                                 | 0    |
| Dispersal–Extinction–Cladogenesis (DEC) Model                                                             | 0    |
| Distance-Based Redundancy Analysis (dbRDA)                                                                | 1    |
| DistilBERT                                                                                                | 0    |
| Distorted Born Approximation (DBA)                                                                        | 0    |
| DLinear                                                                                                   | 0    |
| DNA Sequence Alignment                                                                                   | 0    |
| DOMAIN Algorithm                                                                                         | 0    |
| Dropout Regularization                                                                                   | 1    |
| Dynamic Dewpoint Isotherm (DDI) Method                                                                   | 0    |
| Dynamic MIDAS‑QRF                                                                                         | 1    |
| Dynamic Programming                                                                                      | 1    |
| Edge‑Greedy Search for Causal Graph Construction                                                          | 0    |
| eDMD with Dictionary Learning (eDMD‑DL)                                                                   | 0    |
| ElasticNet                                                                                                | 1    |
| Empirical Mode Decomposition (EMD)                                                                        | 1    |
| End‑Member Modeling Algorithm (EMMA)                                                                      | 1    |
| Energy and Mass Conservation Equations                                                                    | 1    |
| Ensemble Averaging                                                                                        | 1    |
| Entropy Weight Method                                                                                     | 0    |
| Enzymatic Catalysis Modelling                                                                             | 0    |
| Equal‑Area Quadratic Spline                                                                                | 0    |
| Ewald Summation                                                                                          | 0    |
| Exergy Analysis                                                                                              | 1    |
| Expectation–Maximization (EM) Algorithm                                                                      | 1    |
| Experience Replay                                                                                            | 0    |
| Exponential Smoothing                                                                                        | 1    |
| Extended Dynamic Mode Decomposition (eDMD)                                                                   | 0    |
| Extended Fourier Amplitude Sensitivity Testing (EFAST)                                                       | 0    |
| Extreme Learning Machine (ELM)                                                                               | 0    |
| Extremely Randomized Trees (XRT)                                                                             | 1    |
| Fama–French Three‑Factor and Five‑Factor Model Regressions                                                   | 0    |
| FAST‑LIO2 (Fast LiDAR‑Inertial Odometry 2)                                                                   | 0    |
| Data Fusion Methods (Data‑Level, Decision‑Level, Feature‑Level Fusion)                                       | 1    |
| Finite Differences Simulator                                                                                 | 1    |
| Finite Volume Method (FVM)                                                                                   | 1    |
| Fully Connected Neural Network (FNN)                                                                         | 1    |
| Galerkin Method                                                                                              | 0    |
| Gamma Regression                                                                                             | 1    |
| GARCH / GARCH‑MIDAS                                                                                          | 0    |
| Gene Expression Programming (GEP)                                                                            | 0    |
| Generalized Estimating Equations (GEE)                                                                       | 1    |
| Generalized Likelihood Uncertainty Estimation (GLUE)                                                         | 1    |
| Generalized Three‑Cornered Hat (TCH) Method                                                                 | 1    |
| Genetic Algorithm (GA, NSGA‑II, RAGA)                                                                        | 1    |
| Geodetector Model (GDM)                                                                                      | 1    |
| GF‑SG Fusion Algorithm                                                                                        | 0    |
| GIS‑Based Spatial Multi‑Criteria Evaluation                                                                  | 1    |
| Graphical LASSO (gLASSO)                                                                                     | 1    |
| Gravity Search Optimization Algorithm (GVSAO)                                                                | 0    |
| Grid Interpolation (Linear)                                                                                  | 1    |
| Group‑Based Trajectory Modeling                                                                              | 0    |
| Harmonic Tidal Analysis                                                                                      | 1    |
| Harmony Search                                                                                               | 0    |
| Heat Transfer Algorithm                                                                            | 1    |
| Hidden Markov Model (HMM)                                                                         | 1    |
| Hilbert Spectral Analysis                                                                          | 0    |
| Hobday-Based Anomaly Detection Algorithm                                                          | 0    |
| Homology Modeling                                                                                 | 0    |
| Improved 1‑D DAS Method                                                                           | 0    |
| Improved 2‑D DAS Method                                                                           | 0    |
| Incremental kd‑Tree (ikd‑Tree)                                                                    | 0    |
| Indicator Species Analysis (IndVal.g)                                                             | 1    |
| Initial-Condition-Dependent Finite‑Time Stabilizing Controller                                   | 0    |
| Integral Equation Method (IEM)                                                                    | 0    |
| Interferometric Coherence Techniques                                                              | 1    |
| Interquartile Range (IQR)-Based Outlier Detection                                                  | 1    |
| Inverse Modelling (Parameter Optimization)                                                        | 1    |
| Isotonic Regression                                                                                | 0    |
| Item Nearest Neighbor (ItemNN)                                                                    | 1    |
| Iterative Optimization                                                                             | 1    |
| Jenks Natural Breaks Classification                                                                | 1    |
| K‑Shape Clustering                                                                                 | 0    |
| Kaplan–Meier Survival Analysis                                                                     | 1    |
| KDTree-Based Spatial Grouping                                                                      | 1    |
| Kernelized Hypothesis Testing (HSIC)                                                               | 0    |
| Kolmogorov–Smirnov Test (KS Test)                                                                  | 1    |
| Kolmogorov–Zurbenko (KZ) Filter                                                                   | 1    |
| Kolmogorov–Zurbenko (KZ) Periodogram                                                              | 1    |
| Kriging (Spatial Interpolation)                                                                   | 1    |
| L‑Curve Method                                                                                     | 0    |
| L2 Regularization                                                                                  | 1    |
| Laplace Approximation                                                                              | 0    |
| LAVD‑Based Eddy Detection Algorithm                                                               | 0    |
| Least Squares Inversion with Smoothness‑Constrained Regularization                               | 1    |
| Leave‑One‑Year‑Out Cross‑Validation                                                                | 1    |
| Linear Fitting                                                                                     | 1    |
| Linear Mixed Effect Model (REML)                                                                  | 1    |
| Linear Mixed Models (LMM)                                                                         | 1    |
| Linear Quantile Mixed Model (LQMM)                                                                | 1    |
| Linearized Multi‑Block ADMM with Regularization                                                   | 0    |
| Local Outlier Factor (LOF)                                                                        | 1    |
| Log‑Odds Update Scheme                                                                                 | 0    |
| Lossless Compression Algorithm (ZIP)                                                                   | 0    |
| Lossy Compression Algorithm (ZFP)                                                                      | 0    |
| LOWESS (Locally Weighted Scatterplot Smoothing)                                                        | 1    |
| LSTM Autoencoder (LSTM‑AE)                                                                            | 1    |
| Manual Thresholding                                                                                    | 1    |
| Marker‑Assisted Selection                                                                              | 0    |
| Markov Decision Process (MDP)                                                                          | 0    |
| Mass Balance Modeling                                                                                  | 1    |
| Mass Curve Technique (MCT)                                                                             | 1    |
| Match Climates Regional Algorithm                                                                      | 0    |
| Maximum Likelihood Classification (MLC)                                                                | 1    |
| Maximum Power Point Tracking (MPPT) Control                                                            | 0    |
| Maximum‑Likelihood Phylogenetic Analysis                                                               | 0    |
| MIDAS Quantile Random Forest (MIDAS‑QRF)                                                                | 1    |
| Migration Event Detection Algorithm                                                                    | 0    |
| Minimum Bounding Rectangle                                                                              | 0    |
| MIR Method                                                                                             | 0    |
| Mixed Data Sampling (MIDAS)                                                                            | 1    |
| Mixed‑Effects Modeling (LME4)                                                                          | 1    |
| Mixed‑Finite Element Scheme                                                                            | 0    |
| Mixed‑Integer Linear Programming (MILP)                                                                 | 0    |
| Mixture Density Networks (MDN)                                                                         | 1    |
| Model Predictive Control (MPC)                                                                         | 1    |
| Model‑Agnostic Meta‑Learning (MAML)                                                                    | 0    |
| Modified Response Matrix Method                                                                        | 0    |
| Moffat Uncertainty Analysis Method                                                                     | 0    |
| Molecular Docking                                                                                      | 0    |
| Molecular Dynamics Simulation                                                                          | 0    |
| Monte Carlo Dropout (MC‑Dropout)                                                                       | 1    |
| Monte Carlo Integration                                                                                | 1    |
| Moving Average Filter                                                                                  | 1    |
| Multi‑Agent Reinforcement Learning (MARL)                                                              | 0    |
| Multi‑Criteria Performance Evaluation                                                                  | 1    |
| Multi‑Label Machine Learning                                                                            | 0    |
| Multi‑Objective Evolutionary Algorithm (MOEA)                                                           | 0    |
| Multi‑Objective Optimization                                                                           | 1    |
| Multi-Task Linear Regression (MTLR)                                                                    | 1    |
| Multihead Self-Attention Mechanism                                                                     | 1    |
| Multilevel Logistic Regression (MLLogR)                                                                | 1    |
| Multilinear Regression (MLR)                                                                           | 1    |
| Multinomial Logit Model                                                                                 | 1    |
| Multiscale Geographically Weighted Regression (MGWR)                                                    | 1    |
| Multivariate Linear Regression                                                                          | 1    |
| Multivariate Quantiles / Multiple-Output Regression Quantiles                                          | 1    |
| Multivariate Regression                                                                                 | 1    |
| Naive Bayes                                                                                            | 1    |
| Negative Binomial Mixed‑Effects Models                                                                  | 1    |
| Neighbor‑Joining Method                                                                                 | 0    |
| Network‑Based Path Filtering                                                                            | 0    |
| Neural Network Operators (Kantorovich)                                                                  | 0    |
| Neural Network Operators (Classical)                                                                    | 0    |
| Neural Network Classifier (NN Classifier)                                                               | 1    |
| NSGA‑II (Non‑Dominated Sorting Genetic Algorithm II)                                                     | 1    |
| NMDS (Non‑Metric Multidimensional Scaling)                                                              | 1    |
| Nonlinear Regression (nlsLM)                                                                            | 1    |
| Nonparanormal Transformation                                                                            | 0    |
| Normalization of Raster Data                                                                            | 1    |
| Optical Flow Method                                                                                     | 1    |
| ODE System Modelling (Ordinary Differential Equation)                                                   | 1    |
| PARAFAC Modeling                                                                                        | 1    |
| Partial Correlation Analysis                                                                            | 1    |
| PELT (Pruned Exact Linear Time Algorithm)                                                               | 1    |
| Percentile-Based Extreme Precipitation Analysis                                                         | 1    |
| Permutation Feature Importance                                                                          | 1    |
| PID Control                                                                                            | 1    |
| Poisson Regression                                                                                      | 1    |
| Polynomial Basis Functions                                                                              | 1    |
| Power-Law Regression                                                                                    | 1    |
| PISO Algorithm (Pressure-Implicit Splitting of Operators)                                               | 1    |
| PCA (Probabilistic Cellular Automata)                                                                   | 0    |
| PCRO‑SL (Probabilistic Coral Reef Optimization)                                                         | 0    |
| PDF (Probability Density Function) Analysis                                                             | 1    |
| Process‑Based Modeling (Custom Respiration Models)              | 1    |
| Projection Pursuit Model (PPM)                                 | 0    |
| Pruning of Edges                                                 | 0    |
| Pseudo‑Absence Sampling                                         | 1    |
| Pseudo‑Labeling Algorithm                                       | 0    |
| Python‑Based Data Processing                                    | 1    |
| Python‑Based Logistic Curve Fitting                             | 0    |
| Quadratic Classifier                                             | 0    |
| Quadratic Model                                                  | 0    |
| Quadratic Programming (QP) Optimization                         | 0    |
| Quadratic Regression                                             | 0    |
| Quantile Mapping                                                | 1    |
| Quantile Regression Forest (QRF)                                | 1    |
| Quantile‑Based Outlier Removal                                  | 0    |
| Quartic Model                                                    | 0    |
| Radial Basis Functions (RBF)                                    | 1    |
| Rank‑Based Regression Analysis                                  | 0    |
| Ray‑Casting Algorithm                                            | 0    |
| Real‑Coded Accelerated Genetic Algorithm (RAGA)                 | 0    |
| Recurrent Neural Network (RNN)                                  | 1    |
| Reduced‑Order Dual Decomposition                                | 0    |
| Regression Analysis with Control Variables                      | 1    |
| Relay‑Based Switching                                            | 0    |
| Reranking Algorithm                                              | 0    |
| Residual Learning                                                | 1    |
| River Profile Analysis (Knickpoint Detection)                   | 1    |
| Robustness Checks with Dummy Variable Regression                | 1    |
| Robocentric Occupancy Grid Mapping (ROG‑Map)                    | 0    |
| Rule‑Based Control Logic                                        | 1    |
| Runge‑Kutta Method                                              | 1    |
| Runoff Process Vectorization (RPV)                              | 1    |
| Safe Flight Corridor (SFC) Generation                                                                                               | 0    |
| Seasonal ARIMA with Exogenous Variables (SARIMAX)                                                                                   | 1    |
| Seasonal-Trend Decomposition using LOESS (STL)                                                                                     | 1    |
| Self-Organizing Maps (SOM)                                                                                                         | 1    |
| Semi-Analytical Inversion Model (iSAM)                                                                                              | 0    |
| Semi-Implicit Method for Pressure-Linked Equations (SIMPLE)                                                                         | 1    |
| Signed Distance Transform (SDT)                                                                                                     | 1    |
| Simplex Lattice Design                                                                                                             | 0    |
| Simulation-Based Rule Curve Optimization                                                                                           | 1    |
| SMART (Simultaneous Multiplicative Algebraic Reconstruction Technique)                                                              | 0    |
| SIREN (Sinusoidal Representation Networks)                                                                                          | 0    |
| Skeletonisation Algorithm                                                                                                          | 0    |
| Slope–Area Analysis                                                                                                                | 1    |
| SMART with Tikhonov Regularization                                                                                                  | 0    |
| SPACETIME Algorithm                                                                                                                | 0    |
| Sparse Linear Method (SLIM)                                                                                                        | 1    |
| Spatially-Constrained Clustering (ClustGeo)                                                                                         | 1    |
| SPIEC-EASI (Sparse InversE Covariance Estimation)                                                                                   | 1    |
| Spill Adjustment Optimization via Brent’s Method                                                                                    | 0    |
| Spline Quantile Regression                                                                                                         | 1    |
| Split Learning (SL)                                                                                                                | 0    |
| Stacked Ensembles (SE)                                                                                                             | 1    |
| Statistical Threshold Analysis for Tropical Cyclogenesis                                                                           | 1    |
| Stochastic and Deterministic ODEs                                                                                                  | 1    |
| Swath Profiling                                                                                                                    | 0    |
| TC Detection via OWZP Method                                                                                                       | 0    |
| Temporal Downscaling Algorithm Using Proxy Hydrological Data                                                                        | 1    |
| Text Analysis using Keyword-Based Sentence Classification                                                                           | 1    |
| Time Series Cross-Correlation Analysis                                                                                             | 1    |
| Time-Nonhomogeneous Continuous-Time Markov Chain (CTMC)                                                                             | 0    |
| Time-Series Temporal Lag Selection                                                                                                | 1    |
| Tracking-Driven Classification Algorithm                                                                                          | 0    |
| Trust Region Methods                                                                                                               | 1    |
| Twin-Delayed DDPG (TD3)                                                                                                            | 0    |
| Two-Stage Robust Optimization                                                                                                     | 1    |
| Urban Building Energy Modeling (UBEM)                            | 1    |
| Variational Inference (VI)                                       | 1    |
| Vector Autoregression (VAR)                                     | 1    |
| Virtual Screening                                                | 0    |
| ViSIR Hybrid Architecture                                        | 0    |
| Voigt Profile Fitting                                            | 0    |
| Volume of Fluid (VOF) Method                                     | 0    |
| Voronoi-Based Skeleton Extraction                                | 0    |
| Watershed Segmentation                                           | 1    |
| Weighted Sampling for Class Imbalance                            | 1    |
| Word2Vec                                                         | 0    |
