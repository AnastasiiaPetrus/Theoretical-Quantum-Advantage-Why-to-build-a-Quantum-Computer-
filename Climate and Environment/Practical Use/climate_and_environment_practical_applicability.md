# Supplementary Table S1. Practical Use of Classical Algorithms in Climate and Environmental Sciences

## Overview

This supplementary material systematically maps classical algorithms to their real-world adoption in climate and environmental science and engineering. Each algorithm or method is assessed using the **fuse** (Field Use Evidence Score) criterion, which is set to `1` if the method is actively applied in practical (operational, industrial, or regulatory) workflows according to recent reports, guidelines, or published case studies; otherwise, fuse is set to `0`.  
The evaluation is based on an extensive review of international and national agency documents (e.g., US EPA, NOAA, EEA, Chinese MEE), recent peer-reviewed literature, and technical best-practice publications.

## Table Legend

- **Algorithm & Method**: Cluster of related algorithms or models (may include variants and common abbreviations).
- **fuse**:  
  - `1`: Demonstrated and established practical use in climate/environmental science (including operational monitoring, regulatory reporting, or major industrial projects, 2010–2024).  
  - `0`: No significant evidence of practical application in the above domains.

---

## Example: TOP-10 Practical Algorithms

| Algorithm & Method                                                         | fuse | Brief Example/Reference                                                                                   |
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


## Supplementary Table: Full fuse Mapping
| Algorithm & Method | fuse |
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
| Advection–Diffusion Modeling (Advection–Diffusion Modeling, Advection–Diffusion Modeling) | 1 |
| 4D-Var (Four-Dimensional Variational Data Assimilation) (4D-Var (Four-Dimensional Variational Data Assimilation)) | 1 |
| 5-fold cross-validation (5-fold cross-validation) | 1 |
| Abscisic acid-induced gene regulation (Abscisic acid-induced gene regulation) | 0 |
| Absolute Percent Bias (Pbias) | 1 |
| Absolute Relative Error (Absolute Relative Error, Absolute Relative Error (ARE)) | 1 |
| Accumulated Latitudinal Dispersal Calculation (Accumulated Latitudinal Dispersal Calculation) | 0 |
| Actor-Critic Methods (Actor-Critic Methods) | 0 |
| AdaBoost (AdaBoost) | 1 |
| Adams-Bashforth Two-Step Time Integration (Adams-Bashforth Two-Step Time Integration) | 1 |
| Adaptive Combiner (Adaptive Combiner) | 0 |
| Adaptive Multiple Importance Sampling (Adaptive Multiple Importance Sampling) | 0 |
| Adaptive Neuro-Fuzzy Inference System (Adaptive Neuro-Fuzzy Inference System, Adaptive Neuro-Fuzzy Inference Systems (ANFIS)) | 1 |
| Additive Regression (Additive Regression, Additive Regression (AR)) | 1 |
| Adjoint Method (Adjoint Method) | 1 |
| Advantage Actor-Critic (Advantage Actor-Critic, A2C) | 0 |
| Adversarial Loss (Adversarial Loss) | 0 |
| Aethalometer-based source apportionment (Aethalometer-based source apportionment) | 0 |
| AI-Supported Object Detection Pipeline (AI-Supported Object Detection Pipeline) | 0 |
| Algorithmic Rules for Temporary Migration Event Identification (Algorithmic Rules for Temporary Migration Event Identification) | 0 |
| Alternating Direction Method of Multipliers (Alternating Direction Method of Multipliers, ADMM) | 1 |
| ANCOM-BC2 (ANCOM-BC2) | 0 |
| APSIM (APSIM (Agricultural Production Systems Simulator), APSIM (Agricultural Production Systems sIMulator)) | 1 |
| Barry-Macaulay Model (Barry-Macaulay Model, Barry‑Macaulay et al. model) | 0 |
| Area Under the Curve (Area Under the Curve, Area Under the Curve (AUC), Receiver Operating Characteristic Curve, ROC curve) | 1 |
| Aspen Plus Process Simulation (Aspen Plus Process Simulation, Aspen Plus V11 process simulation) | 1 |
| Asymmetric Laplace Distribution (Asymmetric Laplace Distribution) | 0 |
| Atrous Spatial Pyramid Pooling (Atrous Spatial Pyramid Pooling, ASPP) | 0 |
| Attention-based Mapper (Attention-based Mapper, Transformer Encoder) | 0 |
| Augmented Dickey-Fuller Test (Augmented Dickey-Fuller Test, Augmented Dickey-Fuller test (ADF)) | 1 |
| Augmented Random Search (Augmented Random Search, ARS) | 0 |
| Autocorrelation Function (Autocorrelation Function, ACF) | 1 |
| Backward Selection using Akaike Information Criterion (Backward Selection using Akaike Information Criterion) | 1 |
| Backward trajectory analysis (Backward trajectory analysis) | 1 |
| Bagging (Bagging, Bagging (Bootstrap Aggregating), Bootstrap Aggregating) | 1 |
| Bayesian Inversion (Bayesian Inversion) | 0 |
| Binomial Likelihood Modeling (Binomial, Binomial Likelihood Modeling) | 0 |
| Bayesian Nonparametric Partial Clustering (Bayesian Nonparametric Partial Clustering, BNP-PC) | 0 |
| Best Subset Regression Analysis (Best Subset Regression Analysis) | 0 |
| Betweenness Centrality (BC) | 0 |
| Bi-spectral Method (Bi-spectral Method, Bi-spectral method) | 0 |
| Bicubic Interpolation (Bicubic Interpolation, Bicubic interpolation) | 1 |
| Binary Dynamic Programming (Binary Dynamic Programming, Binary dynamic programming) | 0 |
| Biogeochemical Modeling (Biogeochemical Modeling, Biogeochemical (BGC) Modeling (ODE-based)) | 1 |
| Bottom-Up Energy Modeling (Bottom-Up Energy Modeling) | 0 |
| Boolean Matching (Boolean Matching, Boolean matching) | 0 |
| Bootstrap Resampling (Bootstrap Resampling, Bootstrap resampling) | 1 |
| CIE Lab Color Model (CIE Lab Color Model, CIE Lab color model) | 0 |
| Bray-Curtis Dissimilarity (Bray-Curtis Dissimilarity, Bray–Curtis Dissimilarity Metric) | 1 |
| Breaks For Additive Season and Trend (Breaks For Additive Season and Trend, BFAST) | 1 |
| Brier Score (Brier Score, Multicategory Brier Skill Score (MBSS)) | 1 |
| Categorical Cross-Entropy Loss (Categorical Cross-Entropy Loss, Categorical cross-entropy loss) | 0 |
| Centered Log-Ratio Transformation (Centered Log-Ratio Transformation, Centered log-ratio (clr) transformation) | 1 |
| Central Composite Design (Central Composite Design, Central Composite Design (CCD)) | 1 |
| Change Vector Analysis (Change Vector Analysis, CVA) | 1 |
| Chao1 Estimator (Chao1 Estimator, Chao1 estimator) | 0 |
| CHARMm Algorithm (CHARMm Algorithm, CHARMm algorithm) | 0 |
| Chemical Reaction Rate Equation (Chemical Reaction Rate Equation, Chemical reaction rate equations) | 1 |
| Chi-analysis (Chi-analysis (Chi Mapping), χ-analysis (Chi mapping)) | 0 |
| Climate Match Algorithm (Climate Match Algorithm) | 0 |
| Climate Niche Envelope Approach (Climate Niche Envelope Approach) | 1 |
| CLIMEX-DYMEX (CLIMEX-DYMEX) | 0 |
| Clustal Omega (Clustal Omega) | 0 |
| Clustering Algorithm for Segment Detection (Clustering Algorithm for Macro- and Meso-Segment Detection) | 0 |
| Clustering Methods for Trajectory Segmentation | 0 |
| Co-Clustering (Co-Clustering) | 0 |
| Coincidence Analysis (Coincidence Analysis, CNA) | 0 |
| Column-and-Constraint Generation Algorithm (Column-and-Constraint Generation Algorithm, C&CGA) | 0 |
| Comparative Molecular Field Analysis (Comparative Molecular Field Analysis, CoMFA) | 0 |
| Competitive Technology Intelligence (Competitive Technology Intelligence, CTI) | 0 |
| Cobweb Model (Cobweb Model, Cobweb model) | 0 |
| Computational Fluid Dynamics Simulation (Computational Fluid Dynamics Simulation, CFD simulation, Computational Fluid Dynamics (CFD) simulation) | 1 |
| COMSOL Multiphysics Simulation (COMSOL Multiphysics Simulation, COMSOL Multiphysics simulation) | 1 |
| Conditional Autoregressive Value-at-Risk (Conditional Autoregressive Value-at-Risk, Conditional Autoregressive Value-at-Risk (CaViaR): SAV, AD, AS, IG variants) | 0 |
| Conditional Logistic Regression (Conditional Logistic Regression) | 1 |
| Connectivity Probability (CP) | 0 |
| Consistency Distillation (Consistency Distillation) | 0 |
| Composite Signal Modeling (Composite Signal Modeling, TSI components summation model) | 0 |
| Constrained Delaunay Triangulation (Constrained Delaunay Triangulation, Constrained Delaunay triangulation (CDT)) | 0 |
| Constraint-Driven Optimization (Constraint-Driven Optimization, constraint-driven optimization) | 0 |
| Construction of Climate Risk Indices (Construction of Climate Risk Indices) | 1 |
| Contrastive Learning (Contrastive Learning, InfoNCE Loss) | 0 |
| Convex Combination of Dissimilarity Matrices (Convex Combination of Dissimilarity Matrices, Convex Combination of Dissimilarity Matrices (via mixing parameter α)) | 0 |
| Conditional Autoregressive Model (Conditional Autoregressive Model, Conditional Autoregressive (CAR) Model) | 0 |
| Conditional Probability Modeling (Conditional Probability Modeling) | 1 |
| Consistency Model (Consistency Model, Consistency Models (CM)) | 0 |
| Convolutional Time Series Forecasting Backbone (TimesNet) | 0 |
| Correction Method for Population Weighting (Correction Method for Population Weighting) | 1 |
| Correlation Analysis (Correlation Analysis, correlation analysis) | 1 |
| Cost Adjustment Coefficient System (Cost Adjustment Coefficient System, cost adjustment coefficient system) | 0 |
| Crank–Nicolson Numerical Scheme (Crank–Nicolson Numerical Scheme) | 1 |
| Critical States Explainability (Critical States Explainability, Critical States explainability method) | 0 |
| CrossQ (CrossQ) | 0 |
| Curve Fitting (Curve Fitting, curve fitting) | 1 |
| Custom Quad Generation Algorithm (Custom Quad Generation Algorithm, custom quad generation algorithm) | 0 |
| DADA2 (DADA2) | 0 |
| Dagum Gini Coefficient Decomposition (Dagum Gini Coefficient Decomposition, Dagum Gini coefficient decomposition) | 0 |
| Data Augmentation (Data Augmentation, Data Augmentation (including GANs)) | 0 |
| Data Crawling and Filtering (Data Crawling and Filtering, data crawling and filtering) | 0 |
| Data Transformation (Data Transformation) | 1 |
| Data Visualization (Data Visualization, Data visualization) | 1 |
| Data-Level Fusion (Data-Level Fusion, Feature-level fusion, Data-level fusion, Decision-level fusion) | 1 |
| Decentralised Training Decentralised Execution Algorithm (Decentralised Training Decentralised Execution Algorithm, Decentralised Training Decentralised Execution (DTDE) algorithms) | 0 |
| Decision-Level Fusion (Decision-Level Fusion, Decision-level fusion) | 0 |
| Delaunay Triangulation (Delaunay Triangulation, Delaunay triangulation) | 1 |
| Density Functional Theory (Density Functional Theory, DFT calculations) | 0 |
| Design–Build–Test–Learn Cycle (Design–Build–Test–Learn Cycle, Design–Build–Test–Learn (DBTL) cycle) | 0 |
| Desirability Function Approach (Desirability Function Approach, Desirability function approach (Derringer and Suich)) | 0 |
| Deterministic Policy Gradient (Deterministic Policy Gradient, DPG) | 0 |
| DETR (Detection Transformer) (DETR (Detection Transformer), DETR (DEtection TRansformer)) | 0 |
| Direct Absorption Spectroscopy (Direct Absorption Spectroscopy, DAS) | 1 |
| Directed Graph Construction (Directed Graph Construction, Directed Graph Construction from Feature Importances) | 0 |
| DiRienzo–Zurbenko Algorithm (DiRienzo–Zurbenko Algorithm, DZA Smoothing) | 0 |
| Distance-based Redundancy Analysis (Distance-based Redundancy Analysis, dbRDA) | 0 |
| DistilBERT (DistilBERT, DistilBERT (Transformer-based text embeddings)) | 0 |
| Distorted Born Approximation (Distorted Born Approximation, DBA) | 0 |
| DNA Sequence Alignment (DNA Sequence Alignment, DNA sequence alignment) | 0 |
| DOMAIN Algorithm (DOMAIN Algorithm) | 0 |
| Dropout Regularization (Dropout Regularization, Dropout regularisation) | 0 |
| Dynamic Dewpoint Isotherm Method (Dynamic Dewpoint Isotherm Method, Dynamic Dewpoint Isotherm (DDI) method) | 0 |
| Dynamic MIDAS-QRF (Dynamic MIDAS-QRF) | 0 |
| Dynamic Programming (Dynamic Programming, Dynamic programming, binary dynamic programming) | 1 |
| Dynamic Time Warping (Dynamic Time Warping, DTW) | 1 |
| Dynamic Vapour Sorption (Dynamic Vapour Sorption, DVS) | 0 |
| Early Stopping (Early Stopping, Early stopping) | 0 |
| Edge Density (Edge Density, Edge Density (ED)) | 0 |
| Edge Length (Edge Length, Edge Length (EL)) | 0 |
| Edge-Greedy Search (Edge-Greedy Search, Edge-Greedy Search for Causal Graph Construction) | 0 |
| eDMD with Dictionary Learning (eDMD with Dictionary Learning, eDMD-DL) | 0 |
| Effective Mesh Size (Effective Mesh Size, Effective Mesh Size (EM)) | 0 |
| Efficient-Net (EfficientNet) | 0 |
| Empirical Mode Decomposition (Empirical Mode Decomposition, EMD) | 1 |
| Energy and Mass Conservation Equation (Energy and Mass Conservation Equation, energy and mass conservation equations) | 1 |
| Energy Lattice Points (Energy Lattice Points, Energy Lattice Points (ELP)) | 0 |
| Engle’s ARCH Test (Engle’s ARCH Test, Engle’s ARCH test) | 1 |
| Entropy Weight Method (Entropy Weight Method, Entropy Weight Method) | 0 |
| Enzymatic Catalysis (Enzymatic Catalysis, Enzymatic catalysis) | 1 |
| Equal-Area Quadratic Spline (Equal-Area Quadratic Spline, Equal-area quadratic spline) | 1 |
| Equivalent Connectivity (EC) | 0 |
| ERA5 Meteorological Reanalysis (ERA5 Meteorological Reanalysis, ERA5 meteorological reanalysis) | 1 |
| Ewald Summation (Ewald Summation) | 0 |
| Exergy Analysis (Exergy Analysis, exergy analysis) | 1 |
| Expectation-Maximization Algorithm (Expectation-Maximization Algorithm, Expectation-Maximization (EM) Algorithm) | 1 |
| Experience Replay (Experience Replay) | 0 |
| Explainable Artificial Intelligence (Explainable Artificial Intelligence, Explainable AI (XAI)) | 0 |
| Exponential Smoothing (Exponential Smoothing) | 1 |
| Extended Bayesian Information Criterion (Extended Bayesian Information Criterion, Extended Bayesian Information Criterion (EBIC)) | 1 |
| Extended Connectivity Fingerprint (ECFP) | 0 |
| Extended Dynamic Mode Decomposition (Extended Dynamic Mode Decomposition, eDMD) | 0 |
| Extended Fourier Amplitude Sensitivity Testing (Extended Fourier Amplitude Sensitivity Testing, EFAST) | 0 |
| Extreme Learning Machine (Extreme Learning Machine, ELM) | 1 |
| Extreme Value Theory (Extreme Value Theory) | 1 |
| FAIR Digital Object (FAIR Digital Object, FAIR Digital Objects (FAIR Data Management Abstraction)) | 0 |
| Feature Space Dissimilarity Matrix (Feature Space Dissimilarity Matrix) | 0 |
| Feature-Level Fusion (Feature-Level Fusion, Feature-level fusion) | 1 |
| Finite Difference Simulator (Finite Difference Simulator) | 1 |
| Finite Volume Method (Finite Volume Method, FVM) | 1 |
| Finstad Parameterization (Finstad Parameterization, Finstad parameterization) | 0 |
| Copula Modeling (Copula Modeling, Copulas for Multivariate Hazard Modelling) | 1 |
| Floquet Theory (Floquet Theory, Floquet Theory (for seasonal R₀)) | 0 |
| Côté and Konrad Model (Côté and Konrad Model, Côté and Konrad model (2005)) | 0 |
| Frost Number Calculation (Frost Number Calculation, Frost number (FN) calculation) | 1 |
| Fuzzy Overlay Analysis (Fuzzy Overlay Analysis) | 1 |
| Galerkin Method (Galerkin Method, Galerkin Method (for PDE discretization)) | 1 |
| Gamma Regression (Gamma Regression) | 1 |
| GARCH / GARCH-MIDAS (GARCH / GARCH-MIDAS) | 1 |
| Gaussian Copula (GC) | 1 |
| Coulomb and Viscous Friction Model (Coulomb and Viscous Friction Model, Coulomb and viscous friction model) | 0 |
| Gaussian Noise Sensitivity Test (GNST) | 0 |
| CSM-CROPGRO-Cotton (DSSAT) Process Model (CSM-CROPGRO-Cotton (DSSAT) Process Model, CSM-CROPGRO-Cotton (DSSAT) process model) | 1 |
| General Regression Neural Network (GRNN) | 1 |
| Generalized Estimating Equations (Generalized Estimating Equations, GEE) | 1 |
| Generalized Gamma Distribution (Generalized Gamma Distribution, Generalized gamma distribution) | 0 |
| Generalized Likelihood Uncertainty Estimation (Generalized Likelihood Uncertainty Estimation, Generalized Likelihood Uncertainty Estimation (GLUE)) | 1 |
| Dirichlet Process Mixture Model (Dirichlet Process Mixture Model, DPM) | 0 |
| Dispersal–Extinction–Cladogenesis Model (Dispersal–Extinction–Cladogenesis Model, Dispersal–Extinction–Cladogenesis (DEC) Model) | 0 |
| Generalized Three-Cornered Hat Method (Generalized Three-Cornered Hat Method, Generalized three-cornered hat (TCH) method) | 0 |
| GeoCAT (GeoCAT, GeoCAT (Geospatial Conservation Assessment Tool)) | 0 |
| End-Member Modeling Algorithm (End-Member Modeling Algorithm, End-Member Modeling Algorithm (EMMA)) | 1 |
| Geographical Distance-Based Dissimilarity Matrix (Geographical Distance-Based Dissimilarity Matrix) | 0 |
| GEOS (Goddard Earth Observing System) (GEOS (Goddard Earth Observing System)) | 1 |
| Getis-Ord Gi* Statistic (Getis-Ord Gi* Statistic) | 1 |
| GF–SG Fusion Algorithm (GF–SG Fusion Algorithm, GF‑SG fusion algorithm) | 0 |
| GIS-Based Spatial Analysis (GIS-Based Spatial Analysis, GIS-based spatial analysis) | 1 |
| GIS-Based Spatial Multi-Criteria Evaluation (GIS-Based Spatial Multi-Criteria Evaluation, GIS-based Spatial Multi-Criteria Evaluation) | 1 |
| Global and Local Bivariate Moran’s I (Global and Local Bivariate Moran’s I, Global and local bivariate Moran’s I, LISA) | 1 |
| Global Physics-Informed Algorithm (Global Physics-Informed Algorithm) | 0 |
| Gower Distance (Gower Distance, Gower distance) | 1 |
| Graham’s Scan (Graham’s Scan, Graham’s scan (convex hull)) | 1 |
| Energy Limited Capacity Factor Curve Model (ELCFCM) | 0 |
| Energy Substitution Simulation Model (Energy Substitution Simulation Model, Energy substitution simulation model) | 0 |
| Graphical LASSO (Graphical LASSO, gLASSO) | 0 |
| Gravity Search Optimization Algorithm (Gravity Search Optimization Algorithm, GVSAO) | 0 |
| Fama–French Three-Factor and Five-Factor Model Regressions | 0 |
| FLEX Hydrological Model with Dynamic Identification Analysis (FLEX Hydrological Model with Dynamic Identification Analysis, FLEX hydrological model with Dynamic Identification Analysis (DYNIA)) | 1 |
| Grey Wolf Optimizer (Grey Wolf Optimizer, GWO) | 1 |
| GRI-Mech (GRI-Mech, GRI-Mech 3.0) | 0 |
| Grid Interpolation (Grid Interpolation, grid interpolation (linear)) | 1 |
| Grid-based Pond Density Calculation (Grid-based Pond Density Calculation, Grid-based pond density calculation) | 0 |
| Grid-based Remote Sensing Data Assimilation (Grid-based Remote Sensing Data Assimilation, grid-based remote sensing data assimilation) | 1 |
| FLEXPART (FLEXPART, FLEXPART (Lagrangian particle dispersion model)) | 1 |
| Grouped Cross-Validation (Grouped Cross-Validation, grouped cross-validation) | 1 |
| Gaussian Graphical Model (GGM) | 0 |
| H2O AutoML (H2O AutoML) | 0 |
| Halo-Exchange Communication via YAXT Library (Halo-Exchange Communication via YAXT Library, Halo-exchange communication via YAXT library) | 0 |
| Harmonic Tidal Analysis (Harmonic Tidal Analysis) | 1 |
| Harmony Search (Harmony Search) | 1 |
| Heat Transfer Algorithm (Heat Transfer Algorithm, Heat transfer algorithm) | 1 |
| Generalized Additive Model (GAM, GAMLSS) | 1 |
| Geodetector Model (Geodetector Model, Geodetector model (GDM)) | 0 |
| Heterogeneous Acid-Base Catalysis (Heterogeneous Acid-Base Catalysis, Heterogeneous acid-base catalysis) | 1 |
| Hierarchical Frequency-Based Location Estimation (Hierarchical Frequency-Based Location Estimation) | 0 |
| Hilbert Spectral Analysis (Hilbert Spectral Analysis) | 0 |
| Hobday-Based Anomaly Detection Algorithm (Hobday-Based Anomaly Detection Algorithm) | 0 |
| Homogeneous Acid-Base Catalysis (Homogeneous Acid-Base Catalysis, Homogeneous acid-base catalysis) | 1 |
| Hotelling’s T² Control Chart (Hotelling’s T² Control Chart, Hotelling’s T² control chart) | 1 |
| HYDRUS-1D (HYDRUS-1D) | 1 |
| Hyperparameter Optimization (Hyperparameter Optimization) | 1 |
| Ice Flow Correction (Ice Flow Correction, Ice flow correction) | 0 |
| Ideal Gas Law (Ideal Gas Law, Ideal gas law) | 1 |
| Immobilized Lipase Catalysis (Immobilized Lipase Catalysis, Immobilized lipase catalysis) | 0 |
| Improved 1-D DAS Method (Improved 1-D DAS Method, Improved 1-D DAS method) | 0 |
| Improved 2-D DAS Method (Improved 2-D DAS Method, Improved 2-D DAS method) | 0 |
| Improved Grey Wolf Optimizer (Improved Grey Wolf Optimizer, IGWO) | 1 |
| Improved Particle Swarm Optimization–Support Vector Machine (Improved Particle Swarm Optimization–Support Vector Machine, IPSO‑SVM) | 0 |
| Independent Proximal Policy Optimization (Independent Proximal Policy Optimization, IPPO) | 0 |
| Indicator Species Analysis (Indicator Species Analysis, IndVal.g) | 1 |
| Informer (Informer) | 0 |
| Initial-Condition-Dependent Finite-Time Stabilizing Controller (Initial-Condition-Dependent Finite-Time Stabilizing Controller) | 0 |
| Integral Equation Method (Integral Equation Method, IEM) | 1 |
| Intensity Constraint Loss (Intensity Constraint Loss) | 0 |
| Interferometric Coherence Technique (Interferometric Coherence Technique, Interferometric Coherence (CDC-based) Techniques) | 0 |
| Interquartile Range-Based Outlier Detection (Interquartile Range-Based Outlier Detection, IQR-based Outlier Detection) | 1 |
| Intersection over Union Metric (Intersection over Union Metric, IoU (Intersection over Union) metric) | 1 |
| Interval-Valued Fuzzy Set (Interval-Valued Fuzzy Set, Interval-Valued Fuzzy Sets (IVFS)) | 0 |
| GREET Model (GREET Model, GREET model) | 0 |
| Isotonic Regression (Isotonic Regression) | 1 |
| Iterative Optimization (Iterative Optimization, iterative optimization) | 1 |
| Jenks Natural Breaks Classification (Jenks Natural Breaks Classification, Jenks natural breaks classification) | 1 |
| Group-Based Trajectory Modeling (Group-Based Trajectory Modeling) | 0 |
| K-Fold Cross-Validation (K-Fold Cross-Validation, KFCV) | 1 |
| Kaplan–Meier Survival Analysis (Kaplan–Meier Survival Analysis, Kaplan–Meier survival analysis) | 0 |
| Kernelized Hypothesis Testing (Kernelized Hypothesis Testing, HSIC) | 0 |
| Kling–Gupta Efficiency (Kling–Gupta Efficiency, KGE) | 1 |
| Kolmogorov–Smirnov Test (Kolmogorov–Smirnov Test, Kolmogorov–Smirnov test (KS test)) | 1 |
| Kolmogorov–Zurbenko Filter (Kolmogorov–Zurbenko Filter, Kolmogorov–Zurbenko (KZ) Filter) | 1 |
| Kolmogorov–Zurbenko Periodogram (Kolmogorov–Zurbenko Periodogram, Kolmogorov–Zurbenko (KZ) Periodogram) | 1 |
| Koopman Operator Theory (Koopman Operator Theory) | 0 |
| Kriging | 1 |
| Nonlinear Conjugate Gradient Method (Nonlinear Conjugate Gradient Method, Nonlinear Conjugate Gradient Methods) | 1 |
| Nonlinear Regression (Non-LR) | 1 |
| Nonparanormal Transformation (Nonparanormal Transformation) | 0 |
| Normalization of Raster Data (Normalization of Raster Data) | 1 |
| Multinomial Logit Model (Multinomial Logit Model, Multinomial Logit Model) | 1 |
| Negative Binomial Mixed-Effects Model (Negative Binomial Mixed-Effects Model, Negative Binomial Mixed-Effects Models) | 1 |
| One-way ANOVA (One-way ANOVA) | 1 |
| OpenMP Parallelism (OpenMP Parallelism, OpenMP Parallelism (shared memory)) | 1 |
| OpenVisus (OpenVisus, OpenVisus (Out-of-Core, Multiresolution Framework)) | 0 |
| Optical Flow Method (Optical Flow Method, Optical Flow Method) | 1 |
| Optimized Hot Spot Analysis (Optimized Hot Spot Analysis, Optimized Hot Spot Analysis (ArcGIS Tool)) | 1 |
| Overexpression of Metabolic Enzymes (Overexpression of Metabolic Enzymes, Overexpression of metabolic enzymes) | 0 |
| Paired T-test (Paired T-test, Paired T-test) | 1 |
| Parallel Factor Analysis (ParFacAnal) | 0 |
| Partial Autocorrelation Function (Partial Autocorrelation Function, PACF) | 1 |
| Partial Correlation Analysis (Partial Correlation Analysis, Partial correlation analysis) | 1 |
| Partially Observable Stochastic Game (Partially Observable Stochastic Game, Partially Observable Stochastic Games (POSG)) | 0 |
| Particle Swarm Optimization–Artificial Neural Network (PSO‑ANN) | 1 |
| Particle Swarm Optimization–Support Vector Machine (Particle Swarm Optimization–Support Vector Machine, PSO‑SVM) | 0 |
| PELT Algorithm (PELT Algorithm, PELT Algorithm) | 0 |
| Peng-Robinson Thermodynamic Property Method (Peng-Robinson Thermodynamic Property Method, Peng-Robinson thermodynamic property method) | 1 |
| Percentile-Based Extreme Precipitation Analysis (Percentile-Based Extreme Precipitation Analysis, Percentile-based Extreme Precipitation Analysis) | 1 |
| Permutation Feature Importance (Permutation Feature Importance, Permutation Feature Importance) | 1 |
| Phenomics (Phenomics, Phenomics) | 0 |
| PID Control (PID Control, PID control) | 1 |
| Piecewise Linear Cost Function (Piecewise Linear Cost Function, Piecewise Linear Cost Functions) | 1 |
| Planck Function (Planck Function, Planck function) | 1 |
| Poisson Regression (Poisson Regression) | 1 |
| Policy Gradient Method (Policy Gradient Method, Policy Gradient Methods) | 0 |
| Polynomial Basis Function (Polynomial Basis Function, Polynomial Basis Functions) | 1 |
| Post-training Quantization (Post-training Quantization, Post-training quantisation (float16)) | 0 |
| Power Signal Feedback Method (Power Signal Feedback Method, Power Signal Feedback (PSF) method) | 0 |
| Power-Law Regression (Power-La+B632w Regression) | 1 |
| Power-Law Regression (Power-Law Regression) | 1 |
| Predictive Analytics (Predictive Analytics) | 1 |
| Prescriptive Analytics (Prescriptive Analytics) | 0 |
| Pressure-Implicit with Splitting of Operators (Pressure-Implicit with Splitting of Operators, PISO algorithm) | 1 |
| Physics-Based Forward Modeling (Physics-Based Forward Modeling, Physics-based forward modelling) | 1 |
| Probabilistic Cellular Automata (ProbabCA) | 0 |
| Probabilistic Coral Reef Optimization (Probabilistic Coral Reef Optimization with Substrate Layers (PCRO-SL)) | 0 |
| Probability Density Function Analysis (Probability Density Function Analysis, Probability Density Function (PDF) Analysis) | 1 |
| Priestley-Taylor Jet Propulsion Laboratory Model (Priestley-Taylor Jet Propulsion Laboratory Model, Priestley-Taylor Jet Propulsion Laboratory (PT-JPL) model) | 1 |
| Process-Based Modeling (Process-Based Modeling, Process-Based Modeling) | 1 |
| Pruning of Edges Based on Statistical Criteria (Pruning of Edges Based on Statistical Criteria) | 0 |
| Pseudo-Absence Sampling (Pseudo-Absence Sampling, Pseudo-absence sampling) | 1 |
| Projection Pursuit Model (Projection Pursuit Model, Projection Pursuit Model (PPM)) | 0 |
| Pseudo-Labeling Algorithm (Pseudo-Labeling Algorithm, Pseudo-Labeling Algorithm) | 0 |
| PSPNet (PSPNet) | 0 |
| Pyrolysis (Pyrolysis) | 0 |
| Python-Based Data Processing (Python-Based Data Processing, Python-based data processing) | 1 |
| Python-Based Logistic Curve Fitting (Python-Based Logistic Curve Fitting, Python-Based Logistic Curve Fitting (4-parameter, grid-based)) | 0 |
| Q-residuals Control Chart (Q-residuals Control Chart, Q-residuals control chart) | 0 |
| Quadratic Classifier (Quadratic Classifier, Quadratic Classifier) | 0 |
| Quadratic Regression (QuadrR) | 1 |
| Quantile-Based Outlier Removal (Quantile-Based Outlier Removal, Quantile-based outlier removal) | 1 |
| Reaction-Based Parameterized Sunspot Cycle Model (Reaction-Based Parameterized Sunspot Cycle Model, Reaction-Based Parameterized Sunspot Cycle Model) | 0 |
| Quaternion Arithmetic (Quaternion Arithmetic) | 0 |
| Radial Basis Function (Radial Basis Function, Radial Basis Functions (RBF)) | 1 |
| Radiative Transfer Theory (Radiative Transfer Theory, Radiative Transfer Theory (RTT)) | 1 |
| Random Grid Search (Random Grid Search, Random Grid Search) | 1 |
| Random Node Removal (Random Node Removal, Random node removal) | 0 |
| Random Search (Random Search, Random search) | 1 |
| Random Subspace (Random Subspace, Random Subspace Method (RSS), RSS) | 1 |
| Random Subspace Method (Random Subspace Method, RSS) | 1 |
| Range Alignment (Range Alignment, Range alignment) | 0 |
| Rank-Based Regression Analysis | 1 |
| Rank-Based Statistical Framework (Rank-Based Statistical Framework) | 0 |
| Raster-Based Size Function Generation (Raster-Based Size Function Generation, Raster-based size function generation) | 0 |
| Raster-to-Vector Post-Processing (Raster-to-Vector Post-Processing, Raster-to-vector post-processing) | 0 |
| Ray-Casting Algorithm (Ray-Casting Algorithm, Ray-casting algorithm) | 0 |
| Realizable k–ε Turbulence Model (Realizable k–ε Turbulence Model, Realizable k–ε turbulence model) | 1 |
| Reaction-Diffusion Equation (Reaction-Diffusion Equation) | 1 |
| Reactive Distillation (Reactive Distillation, Reactive distillation) | 0 |
| ROSATTA Model (ROSATTA Model, ROSATTA Model) | 0 |
| Receiver Operating Characteristic Curve (Receiver Operating Characteristic Curve, ROC curve) | 1 |
| Reduced-Order Dual Decomposition (Reduced-Order Dual Decomposition) | 0 |
| Redundant Computation for Halo Regions (Redundant Computation for Halo Regions, Redundant Computation for halo regions (communication avoidance)) | 0 |
| REINFORCE (REINFORCE) | 0 |
| Relay-Based Switching (Relay-Based Switching, Relay-based switching) | 0 |
| Reranking Algorithm (Reranking Algorithm, Reranking Algorithm (green-accuracy trade-off utility score)) | 0 |
| Residual Learning (Residual Learning) | 0 |
| ResNet-34 | 0 |
| Reynolds-averaged Navier–Stokes Equations (Reynolds-averaged Navier–Stokes Equations, Reynolds-averaged Navier–Stokes (RANS) equations) | 1 |
| Richards Equation (Richards Equation, Richards Equation) | 1 |
| River Profile Analysis Using Knickpoint Detection (River Profile Analysis Using Knickpoint Detection) | 0 |
| Robustness Checks with Dummy Variable Regression | 1 |
| ROSA Software (ROSA Software, ROSA software) | 0 |
| SEIR Compartmental Model (SEIR Compartmental Model, SEIR Compartmental Model (with seasonality)) | 1 |
| Rule-Based Control Logic (Rule-Based Control Logic, Rule-based control logic) | 1 |
| Runge-Kutta Method (Runge-Kutta Method, Runge-Kutta Method (numerical solver)) | 1 |
| Runoff Process Vectorization (Runoff Process Vectorization, Runoff Process Vectorization (RPV)) | 0 |
| SAC (Soft Actor-Critic) (SAC (Soft Actor-Critic), SAC, Soft Actor-Critic) | 0 |
| Satellite-Derived Shoreline Extraction Using CoastSat (Satellite-Derived Shoreline Extraction Using CoastSat, Satellite-Derived Shoreline Extraction using CoastSat) | 0 |
| Seasonal-Trend Decomposition Using LOESS (Seasonal-Trend Decomposition Using LOESS, Seasonal-Trend Decomposition using LOESS (STL)) | 1 |
| sed_eval Toolbox (sed_eval Toolbox, sed\_eval toolbox) | 0 |
| Segmented Regression (SegR) | 1 |
| Semi-Analytical Inversion Model (Semi-Analytical Inversion Model, semi-analytical inversion model (iSAM)) | 0 |
| Self-Organizing Map (SOM) | 1 |
| Solution-Diffusion Transport Model (Solution-Diffusion Transport Model, Solution-diffusion transport model) | 1 |
| Semi-Implicit Method for Pressure-Linked Equations (Semi-Implicit Method for Pressure-Linked Equations, SIMPLE) | 1 |
| Semi-Supervised Learning (Semi-Supervised Learning, SSL) | 0 |
| Shapiro–Wilk Test (Shapiro–Wilk Test, Shapiro–Wilk test) | 1 |
| Signed Distance Transform (Signed Distance Transform, SDT) | 0 |
| Simplex Lattice Design (Simplex Lattice Design, Simplex lattice design) | 0 |
| Simulation-Based Rule Curve Optimization (Simulation-Based Rule Curve Optimization, Simulation-Based Rule Curve Optimization) | 0 |
| Simultaneous Multiplicative Algebraic Reconstruction Technique (Simultaneous Multiplicative Algebraic Reconstruction Technique, SMART) | 0 |
| Singular Value Decomposition (Singular Value Decomposition, SVD) | 1 |
| Sinusoidal Representation Network (Sinusoidal Representation Network, SIREN) | 0 |
| Skeletonisation Algorithm (Skeletonisation Algorithm, Skeletonisation Algorithm) | 0 |
| Sparse Linear Methods (SLIM ElasticNet-based) | 0 |
| Slope–Area Analysis (Slope–Area Analysis) | 0 |
| SMART Algorithm with Tikhonov Regularization (SMART Algorithm with Tikhonov Regularization, SMART algorithm with Tikhonov regularization) | 0 |
| Soil and Water Assessment Tool (Soil and Water Assessment Tool, SWAT) | 1 |
| Standard k–ε Turbulence Model (Standard k–ε Turbulence Model, Standard k–ε turbulence model) | 1 |
| SPACETIME Algorithm (SPACETIME Algorithm, SPACETIME Algorithm (MDL-based Causal Discovery, Regime/Context Partitioning)) | 0 |
| Spatial Dimension Reduction (Spatial Dimension Reduction, Spatial Dimension Reduction (Global and Zonal Averaging)) | 1 |
| Spatial Fuzzy Kappa (Spatial Fuzzy Kappa, Spatial Fuzzy Kappa) | 1 |
| Spatially-Constrained Clustering | 0 |
| SPIEC-EASI (SPIEC-EASI) | 0 |
| Spill Adjustment Optimization via Brent’s Method (Spill Adjustment Optimization via Brent’s Method) | 0 |
| Split Learning (Split Learning, SL) | 0 |
| State Space Model (State Space Model, State Space Models (SSM)) | 1 |
| Standard Scoring Function (Standard Scoring Function, Standard Scoring Functions) | 0 |
| Taylor–Green Vortex Model (Taylor–Green Vortex Model, Taylor–Green vortex model) | 0 |
| Statistical and Machine Learning Methods (Statistical and Machine Learning Methods, Statistical & Machine Learning Methods:) | 1 |
| Statistical Forecasting (Statistical Forecasting, Statistical Forecasting:) | 1 |
| Statistical Threshold Analysis for TC Genesis (Statistical Threshold Analysis for TC Genesis) | 0 |
| Stefan–Boltzmann Law (Stefan–Boltzmann Law, Stefan–Boltzmann law) | 1 |
| Stereoscopic Neural Network-Based Image Recognition (Stereoscopic Neural Network-Based Image Recognition) | 0 |
| Stochastic and Deterministic ODEs (Stochastic and Deterministic ODEs, Stochastic and Deterministic Ordinary Differential Equations (ODEs)) | 1 |
| Stochastic Weight Averaging (Stochastic Weight Averaging, SWA, SWAG) | 0 |
| Student’s T-test (Student’s T-test, Student’s t-test, t-test) | 1 |
| Sutherland’s Law (Sutherland’s Law, Sutherland’s Law (dynamic viscosity)) | 1 |
| SVD++ (SVD++) | 0 |
| Swath Profiling (Swath Profiling) | 0 |
| Synthetic Carbon Fixation Pathway (Synthetic Carbon Fixation Pathway, Synthetic carbon fixation pathway) | 0 |
| Tanh Activation Function (Tanh Activation Function, tanh activation function) | 0 |
| Temperature-Sensitive Charging Model (Temperature-Sensitive Charging Model, Temperature-sensitive charging model) | 0 |
| TC Detection via OWZP Method (TC Detection via OWZP Method, TC detection via OWZP method) | 0 |
| TD3 (Twin Delayed DDPG) (TD3 (Twin Delayed DDPG), TD3) | 0 |
| Thermal Battery Dynamics Modeling (Thermal Battery Dynamics Modeling, Thermal battery dynamics modeling) | 1 |
| Temporal Downscaling Algorithm (Temporal Downscaling Algorithm, Temporal Downscaling Algorithm) | 1 |
| Terrain Analysis (Terrain Analysis, Terrain Analysis) | 1 |
| Text Analysis using Keyword-Based Sentence Classification (Text Analysis using Keyword-Based Sentence Classification, Text analysis using keyword-based sentence classification) | 0 |
| Thermodynamic Equilibrium Constant Calculation (Thermodynamic Equilibrium Constant Calculation, Thermodynamic equilibrium constant calculation) | 1 |
| Thompson “Aerosol-Aware” Microphysics Scheme (Thompson “Aerosol-Aware” Microphysics Scheme, Thompson “aerosol-aware” microphysics scheme) | 1 |
| Time Series Cross-Correlation Analysis (Time Series Cross-Correlation Analysis, Time Series Cross-Correlation Analysis) | 1 |
| Time-Nonhomogeneous Continuous-Time Markov Chain (Time-Nonhomogeneous Continuous-Time Markov Chain, Time-nonhomogeneous Continuous-time Markov chain) | 0 |
| Time-Series Temporal Lag Selection (Time-Series Temporal Lag Selection, Time-Series Temporal Lag Selection) | 1 |
| Time-Weighted Dynamic Time Warping (Time-Weighted Dynamic Time Warping, TWDTW) | 1 |
| Time–Rate–Distance Formula (Time–Rate–Distance Formula, Time–rate–distance formula) | 0 |
| Top-Down and Downscaling Method (Top-Down and Downscaling Method, Top-down and downscaling method) | 1 |
| Tracking-Driven Classification Algorithm (Tracking-Driven Classification Algorithm) | 0 |
| Transistor-Based Switching (Transistor-Based Switching) | 0 |
| Truncated Quantile Critics (Truncated Quantile Critics, TQC) | 0 |
| Trust Region Method (Trust Region Method, Trust region method) | 1 |
| Tukey’s Multiple Comparison Test (Tukey’s Multiple Comparison Test, Tukey’s multiple comparison test) | 1 |
| Tunable Diode Laser Absorption Spectroscopy (Tunable Diode Laser Absorption Spectroscopy, Tunable Diode Laser Absorption Spectroscopy (TDLAS)) | 1 |
| Two-Layer Quasi-Geostrophic Ocean Model (Two-Layer Quasi-Geostrophic Ocean Model, Two-layer quasi-geostrophic ocean model) | 1 |
| Two-Line Thermometry Method (Two-Line Thermometry Method, Two-line thermometry method) | 0 |
| Two-Stage Robust Optimization (Two-Stage Robust Optimization, Two-stage robust optimization) | 1 |
| Ultrasound-Assisted Transesterification (Ultrasound-Assisted Transesterification, Ultrasound-assisted transesterification) | 0 |
| Unsupervised Learning Algorithm (Unsupervised Learning Algorithm, Unsupervised Learning Algorithms) | 1 |
| Urban Building Energy Modeling (Urban Building Energy Modeling, Urban Building Energy Modeling (UBEM)) | 1 |
| Value-Weighted Portfolio Return Computation (Value-Weighted Portfolio Return Computation, Value-weighted Portfolio Return Computation) | 0 |
| van Genuchten–Mualem Model (van Genuchten–Mualem Model, van Genuchten–Mualem model) | 1 |
| Variable Importance in Projection Score (Variable Importance in Projection Score, VIP Score) | 1 |
| Vector Autoregression (VAR) | 1 |
| Virtual Screening (Virtual Screening, Virtual screening) | 0 |
| Viscoelastic and Wave Modeling (Viscoelastic and Wave Modeling, Viscoelastic and wave modeling) | 1 |
| ViSIR Hybrid Architecture (ViSIR Hybrid Architecture) | 0 |
| ViT + SIREN Integration (ViT + SIREN Integration) | 0 |
| Voigt Profile Fitting (Voigt Profile Fitting, Voigt profile fitting) | 0 |
| Volume of Fluid Method (Volume of Fluid Method, VOF method) | 1 |
| Voronoi-Based Skeleton Extraction (Voronoi-Based Skeleton Extraction, Voronoi-based skeleton extraction) | 0 |
| Watershed Segmentation (Watershed Segmentation, Watershed segmentation) | 1 |
| Wavelength Division Multiplexing (Wavelength Division Multiplexing, WDM) | 0 |
| Weighted Fire Count (Weighted Fire Count, Weighted fire count) | 1 |
| Weighted Sampling for Class Imbalance (Weighted Sampling for Class Imbalance, Weighted sampling for class imbalance) | 1 |
| Word2Vec (Word2Vec, word2vec) | 1 |
| Xception-65 (Xception-65, Xception-65 (as encoder)) | 0 |
| ZFP (Lossy Compression Algorithm) (ZFP (Lossy Compression Algorithm), ZFP) | 0 |
| ZIP (Lossless Compression Algorithm) (ZIP (Lossless Compression Algorithm), ZIP) | 0 |
| Kwiatkowski-Phillips-Schmidt-Shin Test (Kwiatkowski-Phillips-Schmidt-Shin Test, Kwiatkowski-Phillips-Schmidt-Shin test (KPSS)) | 1 |
| L-BFGS Optimization Algorithm (L-BFGS Optimization Algorithm, Limited-memory BFGS-B Method (L-BFGS-B)) | 1 |
| L-curve Method (L-curve Method, L-curve method) | 0 |
| L2 Regularization (L2 Regularization) | 1 |
| Lagrangian Particle Tracking (Lagrangian Particle Tracking, Lagrangian particle tracking) | 1 |
| Heat Transfer Correlation Model (Heat Transfer Correlation Model, Heat transfer correlation models) | 1 |
| Langmuir Parameter (Langmuir Parameter, Langmuir parameter) | 1 |
| Laplace Approximation (Laplace Approximation) | 0 |
| Helsinki University of Technology Model (Helsinki University of Technology Model, Helsinki University of Technology (HUT) Model) | 0 |
| Latent Dirichlet Allocation (Latent Dirichlet Allocation, LDA, Latent Dirichlet Allocation (LDA)) | 1 |
| LAVD-based Eddy Detection Algorithm (LAVD-based Eddy Detection Algorithm) | 0 |
| Least Squares Method (Least Squares Method) | 1 |
| Least-Squares Inversion with Smoothness-Constrained Regularization (Least-Squares Inversion with Smoothness-Constrained Regularization, RES2DINV software) | 0 |
| Leave-One-Year-Out Cross-Validation (Leave-One-Year-Out Cross-Validation) | 1 |
| Homology Modeling (Homology Modeling, homology modeling) | 0 |
| Limited-memory BFGS-B Method (Limited-memory BFGS-B Method, L-BFGS-B) | 1 |
| Linear Discriminant Analysis (Linear Discriminant Analysis, LDA, Linear discriminant analysis) | 1 |
| Linear Fitting (Linear Fitting, Linear Fitting (e.g., for Pₖ vs. Fo relation)) | 1 |
| In Silico Modeling (In Silico Modeling, In silico modeling) | 0 |
| Inverse Modeling (Inverse Modeling) | 1 |
| Linear-Based Time Series Forecasting Backbone (Linear-Based Time Series Forecasting Backbone, DLinear) | 0 |
| Linearized Multi-block ADMM with Regularization (Linearized Multi-block ADMM with Regularization) | 0 |
| Ljung–Box Test (Ljung–Box Test, Ljung–Box test) | 1 |
| ISO 12494 Analytical Model (ISO 12494 Analytical Model, ISO 12494 analytical model) | 1 |
| Local Outlier Factor (LOF) | 1 |
| Local Relief Metric (Local Relief Metric, Local relief metric) | 1 |
| Johansen Model (Johansen Model, Johansen model) | 1 |
| Lowess Smoothing (Lowess Smoothing, LOWESS, Locally Weighted Scatterplot Smoothing) | 1 |
| Kinetic Modeling of Methanol Synthesis (Kinetic Modeling of Methanol Synthesis, Kinetic modeling of methanol synthesis) | 0 |
| Mann–Whitney U Test (Mann–Whitney U Test) | 1 |
| Manual Thresholding (Manual Thresholding, Manual thresholding) | 0 |
| Marker-Assisted Selection (Marker-Assisted Selection, Marker-assisted selection) | 0 |
| Markov Decision Process (Markov Decision Process, MDP) | 0 |
| Marquardt–Levenberg Algorithm (Marquardt–Levenberg Algorithm, Levenberg–Marquardt Algorithm) | 1 |
| MaskFormer (MaskFormer) | 0 |
| Logarithmic Wind Profile Model (Logarithmic Wind Profile Model, Logarithmic wind profile model) | 1 |
| Mass Curve Technique (Mass Curve Technique, Mass Curve Technique (MCT)) | 0 |
| Match Climates Regional Algorithm (Match Climates Regional Algorithm) | 0 |
| Maximal Information Coefficient (Maximal Information Coefficient, MIC) | 1 |
| Maximum Likelihood Classification (Maximum Likelihood Classification, MLC) | 1 |
| Maximum Power Point Tracking Control (Maximum Power Point Tracking Control, Maximum Power Point Tracking (MPPT) control) | 1 |
| Maximum-Likelihood Phylogenetic Analysis (Maximum-Likelihood Phylogenetic Analysis, maximum-likelihood phylogenetic analysis) | 0 |
| Mean Squared Error Loss (Mean Squared Error Loss, Mean Squared Error (MSE) Loss) | 1 |
| Mean Teacher Framework (Mean Teacher Framework) | 0 |
| Membrane Separation (Membrane Separation, membrane separation) | 1 |
| Lu Model (Lu Model, Lu et al. model) | 0 |
| Metropolis-within-Gibbs Sampling (Metropolis-within-Gibbs Sampling) | 0 |
| Metropolis, Metropolis-Hastings, Gibbs Sampling (Metropolis, Metropolis-Hastings, Gibbs Sampling) | 1 |
| Micro-emulsification (Micro-emulsification) | 0 |
| Microwave-Assisted Transesterification (Microwave-Assisted Transesterification, Microwave-assisted transesterification) | 0 |
| Migration Event Detection Algorithm (Migration Event Detection Algorithm) | 0 |
| Minimum Bounding Rectangle (Minimum Bounding Rectangle, Minimum bounding rectangle) | 0 |
| MIR Method (MIR Method, MIR method) | 0 |
| Mixed Data Sampling (Mixed Data Sampling, Mixed Data Sampling (MIDAS)) | 1 |
| Mass Balance Modeling (Mass Balance Modeling, Mass Balance Modeling) | 1 |
| Mixed-Finite Element Scheme (Mixed-Finite Element Scheme, Mixed-Finite Element Scheme for atmospheric dynamics discretization) | 1 |
| Mixed-Integer Linear Programming (Mixed-Integer Linear Programming, Mixed-Integer Linear Programming (MILP)) | 1 |
| Messinger Model (Messinger Model, Messinger model) | 1 |
| Mixture Density Network (Mixture Density Network, Mixture Density Networks (MDN)) | 0 |
| Michigan Microwave Canopy Scattering Model (Michigan Microwave Canopy Scattering Model, MIMICS) | 0 |
| Mixed-Effects Modeling (Mixed-Effects Modeling, Mixed-Effects Modeling (LME4)) | 1 |
| Mixing Model (Mixing Model, Mixing Models) | 1 |
| Modified Response Matrix Method (Modified Response Matrix Method, Modified Response Matrix method) | 1 |
| Moffat Uncertainty Analysis Method (Moffat Uncertainty Analysis Method, Moffat uncertainty analysis method) | 0 |
| Molecular Docking (Molecular Docking, molecular docking) | 0 |
| Molecular Dynamics Simulation (Molecular Dynamics Simulation, Molecular dynamics simulation) | 1 |
| Monodispersed MVD Approximation (Monodispersed MVD Approximation, Monodispersed MVD approximation) | 0 |
| Monte Carlo Dropout (MC-Dropout) | 0 |
| Morphological Operations (Morphological Operations, Morphological operations) | 0 |
| Morphological Thinning (Morphological Thinning, Morphological Thinning) | 0 |
| Moving Average Filter (Moving Average Filter, Moving average filter) | 1 |
| MPI Parallelism (MPI Parallelism, MPI Parallelism (distributed memory)) | 1 |
| Multi-Criteria Performance Evaluation (Multi-Criteria Performance Evaluation, Multi-criteria performance evaluation) | 1 |
| Model-Agnostic Meta-Learning (Model-Agnostic Meta-Learning, MAML) | 0 |
| Multi-label Machine Learning (Multi-label Machine Learning) | 0 |
| Multi-Objective Evolutionary Algorithm (Multi-Objective Evolutionary Algorithm, MOEA; Borg) | 1 |
| Multi-Objective Optimization (Multi-Objective Optimization, Multi-objective optimization) | 1 |
| Linear Regression (LR, Least-Squares LR, Multitask LR, Multivariate LR) | 1 |
| Multihead Self-Attention Mechanism (Multihead Self-Attention Mechanism, Multihead Self-Attention Mechanism) | 0 |
| Multiscale Geographically Weighted Regression (Multiscale Geographically Weighted Regression, MGWR) | 1 |
| Linear Regression (LR, Least-Squares LR, Multi-Task LR, Multivariate LR) | 1 |
| Naive Bayes (Naive Bayes, Naive Bayes) | 1 |
| Nash–Sutcliffe Efficiency (Nash–Sutcliffe Efficiency, Nash-Sutcliffe Efficiency (NSE)) | 1 |
| NDVI and Land Surface Temperature Indices (NDVI and Land Surface Temperature Indices, NDVI and land surface temperature indices) | 1 |
| NDWI (NDWI) | 1 |
| Modified ISO 12494 Model (Modified ISO 12494 Model, Modified ISO 12494 model) | 1 |
| Neighbor-Joining Method (Neighbor-Joining Method, Neighbor-joining method) | 0 |
| Network-Based Path Filtering (Network-Based Path Filtering, Network-based path filtering) | 0 |
| Non-metric Multidimensional Scaling (NMDS) | 1 |


