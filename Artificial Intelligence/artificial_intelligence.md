# Artificial Intelligence: Algorithm and Method Extraction, Mapping, and Quantum Analogue Status

## Introduction

This file documents the full process and results of algorithm and method extraction, clustering, practical use assessment, and quantum analogue mapping for the domain of Artificial Intelligence.

## Search Parameters

- **Keywords:** `artificial intelligence AND algorithm`
- **Platforms:** arXiv, SemanticScholar
- **Time window:** 2025
- **Sample size:** 150 articles per keyword/platform combination

_Other parameters, keywords, and platforms can be adjusted as needed._

## Selection Process Overview

For the domain of Artificial Intelligence \(D\), we compile a broad list of classical algorithms and methods \(A\) known to be used in the field. We then assign an **importance score** \(\mathcal{I}(A, D)\) to each algorithm or method based on:
- its frequency in domain-specific scientific literature, and
- whether it is confirmed to be practically used in that domain.

Formally:
\[
\mathcal{I}(A, D) = \mathrm{fpub}(A, D) \times \mathrm{fuse}(A, D)
\]
where  
- \(\mathrm{fpub}(A, D)\) is the normalized frequency of mentions of \(A\) in the literature (see below),  
- \(\mathrm{fuse}(A, D)\) is a binary indicator: 1 if \(A\) is in practical/operational use in domain \(D\), 0 otherwise.

## Process Summary

- Algorithms and methods were extracted from titles, abstracts, keywords, and text snippets.
- Synonymous and related variants unified under standard names.
- Practical use assigned via recent literature and application reports (2025).
- Quantum analogue status mapped from up-to-date quantum computing and machine learning research.

---

## Table Legend

- **Algorithm & Method:** Cluster of related algorithms and models (variants unified).
- **mentions:** Number of literature references in the aggregated sample (2025).
- **fpub:** Normalized publication frequency.
- **fuse:** Practical use in AI workflows (`1` = yes, `0` = no).
- **\(\mathcal{I}(A, D)\):** Importance score.

---

## Results Table: Top-10 Classical Algorithms and Methods in Artificial Intelligence

| Algorithm & Method                                            | mentions | fpub  | fuse | \( \mathcal{I}(A, D) \) | Quantum Analogue                | Notes                                                                                   |
|-------------------------------------------------------------|----------|-------|------|-------------------------|-------------------------------|-----------------------------------------------------------------------------------------|
| Convolutional Neural Networks (CNN, ResNet, U-Net, EfficientNet) | 120      | 0.103 | 1    | 0.103                   | Quantum CNN (QCNN)             | Core deep learning models for vision and multimodal tasks; quantum CNN actively researched. |
| Random Forest (RF)                                           | 58       | 0.050 | 1    | 0.050                   | Quantum Random Forest (QRF)    | Popular ensemble method; quantum variants experimental.                                |
| Support Vector Machine (SVM, Linear SVC variants)            | 55       | 0.047 | 1    | 0.047                   | Quantum SVM (QSVM)             | Classical ML staple; QSVM is a mature quantum algorithm.                               |
| Gradient Boosting (XGBoost, LightGBM, GB, NGBoost)           | 41       | 0.035 | 1    | 0.035                   | Quantum Gradient Boosting**    | Widely used boosting method; quantum boosting still theoretical/experimental.          |
| Reinforcement Learning (RL, PPO, Q-Learning, DQN, PbRL)      | 39       | 0.033 | 1    | 0.033                   | Quantum RL                    | Broad RL applications; quantum RL research ongoing.                                   |
| Genetic Algorithm (GA) and Metaheuristics (PSO, ACO, WOA, Bee Colony, etc.) | 38       | 0.033 | 1    | 0.033                   | Quantum GA (QGA)               | Optimization methods; quantum GA under active research.                               |
| Long Short-Term Memory (LSTM) and Variants (BiLSTM, GRU)     | 37       | 0.032 | 1    | 0.032                   | Quantum LSTM                  | Sequence and time series modeling; quantum LSTM proposed.                             |
| Decision Tree (DT) and Ensembles (Boosted, Stacked)          | 31       | 0.027 | 1    | 0.027                   | Quantum Decision Tree (QDT)    | Interpretable classifiers; quantum DT in research.                                   |
| Transformer Models (Multi-Head Attention, BERT)               | 30       | 0.026 | 1    | 0.026                   | Quantum Transformers*          | Leading NLP models; quantum transformer research in early stages.                    |
| K-Nearest Neighbors (KNN)                                    | 20       | 0.017 | 1    | 0.017                   | Quantum KNN                   | Instance-based learning; quantum KNN proposed.                                        |

---

## Quantum Analogy

Most top classical AI algorithms have corresponding quantum analogues at varying maturity levels:  
- Quantum CNN, Quantum SVM, Quantum Random Forest and Quantum PCA are well-studied.  
- Quantum Gradient Boosting and Quantum Transformers are emerging but mostly theoretical.  
- Quantum Reinforcement Learning and Quantum Genetic Algorithms show experimental promise.  
- Quantum KNN and Quantum Decision Trees have early research prototypes.

These quantum counterparts promise speedups and enhanced learning capabilities for complex AI tasks.

---

## Notes

- All Top-10 methods show confirmed practical use (`fuse=1`) in 2025 AI research.
- Quantum analogues exist for each top method, though some are still nascent.
- The mix of classical and quantum algorithms reflects AI’s foundational and rapidly evolving nature.

---

## Supplementary Table: Full List of Algorithms and Methods

| Algorithm & Method                                    | mentions | fpub  | fuse | \( \mathcal{I}(A, D) \) |
|-----------------------------------------------------|----------|-------|------|--------------------------|
| Convolutional Neural Networks (CNN, ResNet, U-Net)  | 120      | 0.103 | 1    | 0.103                    |
| Random Forest (RF)                                  | 58       | 0.050 | 1    | 0.050                    |
| Support Vector Machine (SVM, Linear SVC)            | 55       | 0.047 | 1    | 0.047                    |
| Gradient Boosting (XGBoost, LightGBM, GB)           | 41       | 0.035 | 1    | 0.035                    |
| Reinforcement Learning (RL, PPO, Q-Learning, DQN)   | 39       | 0.033 | 1    | 0.033                    |
| Genetic Algorithm and Metaheuristics (PSO, ACO, WOA) | 38       | 0.033 | 1    | 0.033                    |
| Long Short-Term Memory (LSTM, BiLSTM, GRU)          | 37       | 0.032 | 1    | 0.032                    |
| Decision Tree (DT) and Ensembles                     | 31       | 0.027 | 1    | 0.027                    |
| Transformer Models (Multi-Head Attention, BERT)     | 30       | 0.026 | 1    | 0.026                    |
| K-Nearest Neighbors (KNN)                           | 20       | 0.017 | 1    | 0.017                    |
| Bayesian Algorithms (Optimization, Knowledge Transfer) | 17       | 0.015 | 1    | 0.015                    |
| Autoencoders (VAE, Denoising AE)                     | 15       | 0.013 | 1    | 0.013                    |
| Graph Neural Networks (GNN, GCN)                     | 15       | 0.013 | 1    | 0.013                    |
| Kalman Filter and Particle Filter                    | 13       | 0.011 | 1    | 0.011                    |
| Clustering Algorithms (K-means, Markov, Fuzzy C-Means) | 13       | 0.011 | 1    | 0.011                    |
| Naïve Bayes Classifier                               | 11       | 0.009 | 1    | 0.009                    |
| Formal Verification & Static/Dynamic Analysis        | 10       | 0.009 | 1    | 0.009                    |
| Monte Carlo Methods (MCMC, Metropolis-Hastings)       | 9        | 0.008 | 1    | 0.008                    |
| Differential Privacy and Federated Learning           | 9        | 0.008 | 1    | 0.008                    |
| YOLO Object Detection Series                           | 9        | 0.008 | 1    | 0.008                    |
| Ensemble Learning (Stacking, Bagging, Boosting)       | 8        | 0.007 | 1    | 0.007                    |
| Adversarial Attack/Defense Methods                     | 8        | 0.007 | 1    | 0.007                    |

---

## Raw Data and Logs

- [Artificial Intelligence / Publication Frequency](https://github.com/AnastasiiaPetrus/Theoretical-Quantum-Advantage-Why-to-build-a-Quantum-Computer-/blob/main/Artificial%20Intelligence/publication_frequency.md)  
- [Artificial Intelligence / Practical Use](https://github.com/AnastasiiaPetrus/Theoretical-Quantum-Advantage-Why-to-build-a-Quantum-Computer-/blob/main/Artificial%20Intelligence/practical_applicability.md)  

---

*Sources:*  
- Aggregated algorithm and method frequency data (2025)  
- Practical use mapping (fuse)  
- Quantum analogue status based on current quantum machine learning and computing literature (2025)
