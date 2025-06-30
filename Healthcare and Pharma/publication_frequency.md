# Supplementary Table S2. Standardized and Aggregated Algorithm Frequency (fpub)  
**Domain:** Healthcare and Pharma (Keyword: "healthcare and pharma AND algorithm")  
**Dataset:** 100 articles (2024-2025) from PubMed, ScienceDirect  

---

## Methodology

- Extracted algorithms and method names from titles, abstracts, keywords, and text snippets.
- Standardized synonymous and closely related variants under unified algorithm names.
- Counted total mentions = 633 (aggregate from dataset).
- Normalized frequency \( fpub = \frac{\text{mentions}}{633} \) to represent relative importance.

---

## Top Algorithms and Methods by Frequency

| Algorithm / Method                                             | Mentions | fpub   |
|---------------------------------------------------------------|----------|--------|
| Convolutional Neural Networks (CNN, including U-Net, ResNet)  | 39       | 0.062  |
| Random Forest (RF)                                             | 31       | 0.049  |
| Logistic Regression (including Penalized, Multivariable)      | 28       | 0.044  |
| Neural Networks (ANN, DNN, RNN, LSTM, GRU)                     | 22       | 0.035  |
| Deep Learning (DL, including Autoencoders, GANs)              | 20       | 0.032  |
| Support Vector Machine (SVM, SVC)                             | 15       | 0.024  |
| Gradient Boosting (XGBoost, CatBoost, GBDT)                   | 14       | 0.022  |
| Transformer Models (GPT, BERT, Vision Transformer, etc.)      | 13       | 0.021  |
| K-means Clustering                                            | 10       | 0.016  |
| Principal Component Analysis (PCA)                            | 8        | 0.013  |
| Bayesian Algorithms (Bayesian Regression, Frameworks)         | 7        | 0.011  |
| Cox Proportional Hazards Model                                | 6        | 0.009  |
| SHAP (Shapley Additive Explanations) and Explainable AI (XAI) | 6        | 0.009  |
| Clustering (Hierarchical, HDBSCAN, K-prototype)                | 5        | 0.008  |
| Reinforcement Learning (RL)                                    | 4        | 0.006  |
| Ensemble Learning Methods (Stacking, Bagging, Boosting)       | 4        | 0.006  |
| Mixed-Integer Programming (MIP) and Optimization Heuristics   | 3        | 0.005  |
| Rule-Based Algorithms                                         | 3        | 0.005  |
| Random Survival Forests (RSF)                                 | 2        | 0.003  |
| Log-ratio Lasso Regression                                    | 1        | 0.002  |

---

## Summary and Observations

- **Deep learning** and **CNN architectures** (U-Net, ResNet) dominate, reflecting widespread use in medical imaging and diagnostics.
- **Random Forest** and **Logistic Regression** are prevalent classical machine learning methods across clinical prediction and classification tasks.
- Transformer-based models (GPT, BERT, Vision Transformer) are increasingly adopted, especially in NLP and image analysis applications.
- Ensemble and boosting methods (XGBoost, CatBoost) are common in tabular data and survival analysis.
- Explainability techniques such as SHAP and broader XAI approaches are gaining importance in healthcare settings.
- Clustering algorithms (K-means, hierarchical, HDBSCAN) support patient stratification and phenotyping.
- Survival analysis methods like Cox proportional hazards and Random Survival Forests are used for time-to-event outcomes.
- Optimization and metaheuristic algorithms (GA, ACO, NSGA-II) appear in pharmaceutical manufacturing and supply chain contexts.
- Bayesian methods and probabilistic frameworks are used for inference and decision support.
- Rule-based algorithms and classical statistics remain important for coding and diagnosis algorithms.
- Growing application of generative models (GANs, Autoencoders) for synthetic data generation and augmentation.
- The dataset reflects a balanced mix of classical, modern ML, and specialized healthcare algorithms tailored to diverse clinical and pharmaceutical challenges.
