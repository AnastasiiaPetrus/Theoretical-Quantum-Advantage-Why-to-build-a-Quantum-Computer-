# Finance and Economics: Algorithm and Method Extraction, Mapping, and Quantum Analogue Status

## Introduction

This file documents the full process and results of algorithm and method extraction, clustering, practical use assessment, and quantum analogue mapping for the domain of Finance and Economics.

## Search Parameters

- **Keywords:** `finance and economics AND algorithm`
- **Platforms:** Google Scholar, ScienceDirect
- **Time window:** 2024–2025
- **Sample size:** 100 articles per keyword/platform combination

_Other parameters, keywords, and platforms can be adjusted as needed._

## Selection Process Overview

For the domain of Finance and Economics \(D\), we compile a broad list of classical algorithms and methods \(A\) known to be used in the field. We then assign an **importance score** \(\mathcal{I}(A, D)\) to each algorithm or method based on:
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

- Articles were collected and reviewed for explicit mentions of algorithms or methods in titles, abstracts, and keywords.
- All classical algorithms and methods were extracted, then clustered by name and application similarity.
- For each cluster, practical use (\(\mathrm{fuse}\)) was assigned based on recent literature, case studies, and industry reports (2024–2025).
- Quantum analogue status was mapped for each practically used cluster, based on recent quantum machine learning and quantum computing literature.

---

## Table Legend

- **Algorithm & Method:** Standardized cluster of related algorithms and methods (variants unified).
- **mentions:** Number of literature references in the aggregated sample (2024–2025).
- **fpub:** Normalized publication frequency (\( \mathrm{fpub}(A, D) = \frac{\text{mentions}}{\sum_{A'} \text{mentions}} \)).
- **fuse:** Practical use in finance and economics (`1` = yes, `0` = no).
- **\(\mathcal{I}(A, D)\):** Importance score; only methods with \(\mathcal{I}(A, D) \geq \theta\) (here, \(\theta = 0.01\)) are considered significant for quantum mapping.

---

## Results Table: Top-10 Classical Algorithms and Methods in Finance and Economics

| Algorithm & Method                                | mentions | fpub  | fuse | \( \mathcal{I}(A, D) \) | Quantum Analogue                 | Notes                                                                                               |
|-------------------------------------------------|----------|-------|------|-------------------------|--------------------------------|---------------------------------------------------------------------------------------------------|
| Support Vector Machine (SVM, SVR)                | 54       | 0.087 | 1    | 0.087                   | Quantum SVM (QSVM)              | Widely used in classification and regression tasks; QSVM is a mature quantum algorithm.            |
| Genetic Algorithm (GA, Adaptive, Quantum, Hybrid)| 40       | 0.064 | 1    | 0.064                   | Quantum Genetic Algorithm (QGA) | Popular in portfolio optimization; QGA research ongoing.                                           |
| Random Forest (RF)                               | 38       | 0.061 | 1    | 0.061                   | Quantum Random Forest (QRF)     | Applied in fraud detection and risk modeling; quantum variants are experimental.                   |
| Decision Tree (DT, C4.5, ID3)                    | 30       | 0.048 | 1    | 0.048                   | Quantum Decision Tree (QDT)     | Used in risk classification; quantum decision trees proposed but early-stage.                      |
| Long Short-Term Memory (LSTM, BiLSTM, ILSTM)    | 28       | 0.045 | 1    | 0.045                   | Quantum LSTM                   | Financial time series forecasting; quantum LSTM models emerging.                                   |
| Logistic Regression (LR)                         | 24       | 0.039 | 1    | 0.039                   | Quantum Logistic Regression     | Credit scoring and fraud detection; quantum analogues demonstrated.                               |
| Neural Networks (ANN, DNN, Feedforward, RNN)    | 22       | 0.035 | 1    | 0.035                   | Quantum Neural Networks (QNN)   | Broad use in forecasting and classification; quantum neural networks under active research.       |
| Particle Swarm Optimization (PSO)                | 15       | 0.024 | 1    | 0.024                   | Quantum PSO                    | Used for optimization tasks; quantum swarm methods are experimental.                              |
| Convolutional Neural Networks (CNN)              | 14       | 0.022 | 1    | 0.022                   | Quantum CNN (QCNN)              | Sentiment analysis and fraud detection; quantum CNN research ongoing.                             |
| Reinforcement Learning (Q-learning, DQN, RL)     | 14       | 0.022 | 1    | 0.022                   | Quantum Reinforcement Learning  | Algorithmic trading and decision-making; quantum RL proposed in literature.                        |

---

## Quantum Analogy

Classical algorithms dominating finance and economics—such as SVM, Genetic Algorithms, Random Forests, Decision Trees, LSTM, Logistic Regression, and Neural Networks—have corresponding quantum machine learning or quantum optimization analogues.

Key highlights:  
- **Quantum SVM** is a well-studied quantum algorithm with potential speedups in classification tasks.  
- **Quantum Annealing** and **QAOA** provide frameworks for combinatorial optimization akin to Genetic Algorithms and PSO.  
- Quantum neural networks (QNN), quantum convolutional networks (QCNN), and quantum LSTM are emerging to address time series and pattern recognition challenges in finance.  
- Quantum reinforcement learning is at an experimental stage but promising for adaptive finance strategies.

This alignment reflects a growing research interest in leveraging quantum computational advantages for complex financial modeling and decision-making.

---

## Notes

- All Top-10 classical algorithms and methods have confirmed practical use in finance and economics, with `fuse=1`.
- Quantum analogues exist or are proposed for all Top-10 methods, though some remain experimental.
- Importance scores correlate strongly with both practical adoption and quantum algorithm research activity.
- The diversity of classical methods mirrors the multifaceted challenges in financial forecasting, risk management, and optimization.

---

## Supplementary Table: Full List of Algorithms and Methods

| Algorithm & Method                                   | mentions | fpub  | fuse | \( \mathcal{I}(A, D) \) |
|----------------------------------------------------|----------|-------|------|--------------------------|
| Support Vector Machine (SVM, SVR)                   | 54       | 0.087 | 1    | 0.087                    |
| Genetic Algorithm (GA, including Adaptive, Quantum, Hybrid) | 40       | 0.064 | 1    | 0.064                    |
| Random Forest (RF)                                  | 38       | 0.061 | 1    | 0.061                    |
| Decision Tree (DT, C4.5, ID3)                      | 30       | 0.048 | 1    | 0.048                    |
| Long Short-Term Memory (LSTM, BiLSTM, ILSTM)       | 28       | 0.045 | 1    | 0.045                    |
| Logistic Regression (LR)                            | 24       | 0.039 | 1    | 0.039                    |
| Neural Networks (ANN, DNN, Feedforward, RNN)       | 22       | 0.035 | 1    | 0.035                    |
| Particle Swarm Optimization (PSO)                   | 15       | 0.024 | 1    | 0.024                    |
| Convolutional Neural Networks (CNN)                 | 14       | 0.022 | 1    | 0.022                    |
| Reinforcement Learning (Q-learning, DQN, RL)        | 14       | 0.022 | 1    | 0.022                    |
| K-means Clustering                                 | 13       | 0.021 | 1    | 0.021                    |
| Simulated Annealing (SA)                           | 11       | 0.018 | 1    | 0.018                    |
| Gradient Boosting Machines (GBM, XGBoost, CatBoost) | 11       | 0.018 | 1    | 0.018                    |
| Graph Neural Networks (GNN, GCN)                   | 7        | 0.011 | 1    | 0.011                    |
| Apriori / FP-Growth / Association Rule Mining      | 6        | 0.010 | 0    | 0.000                    |
| Tabu Search                                        | 5        | 0.008 | 0    | 0.000                    |
| Monte Carlo Methods                                | 5        | 0.008 | 1    | 0.008                    |
| Support Vector Regression (SVR)                    | 5        | 0.008 | 1    | 0.008                    |
| Bayesian Methods (Regression, MCMC, PCA)           | 5        | 0.008 | 1    | 0.008                    |
| AutoRegressive Models (ARIMA, ARMA, GARCH)          | 5        | 0.008 | 1    | 0.008                    |
| Deep Deterministic Policy Gradient (DDPG)           | 3        | 0.005 | 0    | 0.000                    |
| Fuzzy Logic / Fuzzy Comprehensive Evaluation        | 3        | 0.005 | 1    | 0.005                    |
| Game Theory (Nash Equilibrium, etc.)                | 3        | 0.005 | 1    | 0.005                    |
| Hidden Markov Models (HMM)                          | 3        | 0.005 | 1    | 0.005                    |

---

## Raw Data and Logs

- [Finance and Economics / Publication Frequency](https://github.com/AnastasiiaPetrus/Theoretical-Quantum-Advantage-Why-to-build-a-Quantum-Computer-/blob/main/Finance%20and%20Economic/publication_frequency.md)  
- [Finance and Economics / Practical Use](https://github.com/AnastasiiaPetrus/Theoretical-Quantum-Advantage-Why-to-build-a-Quantum-Computer-/blob/main/Finance%20and%20Economic/practical_applicability.md)  

---

*Sources:*  
- Aggregated algorithm and method frequency data (2024–2025)  
- Practical use mapping (fuse)  
- Quantum analogue status based on recent quantum machine learning and quantum computing literature (2024–2025)
