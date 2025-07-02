# Systematic Algorithm Survey by Domain

## Overview

This repository presents a reproducible methodology for collecting, screening, and analyzing scientific literature to extract and cluster classical algorithms across key application domains. For each domain, the process maps the real-world usage of classical algorithms and investigates the current or potential existence of quantum analogues. The methodology is designed to be transparent and extensible, allowing for validation, extension, and community collaboration.

## Table of Contents

1. Purpose and Motivation  
2. Domain Coverage  
3. Article Collection Protocol  
4. Screening, Extraction, and Clustering Pipeline  
5. Algorithm Ranking and Significance Metrics  
6. Practical Use and Quantum Analogue Mapping  
7. Results Structure  
8. Limitations and Next Steps  
9. How to Contribute  
10. Contact  

---

## Purpose and Motivation

Many scientific and applied fields rely on sophisticated algorithms to address complex real-world problems. This project aims to systematically capture which classical algorithms are actually in use within each domain, cluster similar algorithmic approaches, and provide a map of which have quantum analogues, are under quantum development, or currently have none. All steps are described to maximize reproducibility and support independent verification or further expansion.

---

## Domain Coverage

The methodology is currently being applied to the following domains, with further domains to be added as the project develops:

| Domain                          | Primary Data Sources                                         |
|----------------------------------|-------------------------------------------------------------|
| Climate and Environment         | arXiv, Scopus, Springer                                     |
| Healthcare and Pharma           | PubMed, Semantic Scholar, Scopus                            |
| Finance and Economics           | Scopus, SSRN                                                |
| Logistics and Supply Management | IEEE Xplore, Springer, Semantic Scholar                     |
| Defense and Security            | arXiv, IEEE Xplore, ACM Digital Library, Semantic Scholar   |
| Artificial Intelligence         | arXiv, Semantic Scholar                                     |

---

## Article Collection Protocol

1. **Query Design**  
   For each domain, both domain-specific and algorithmic keywords are used, for example, `climate`, `healthcare`, `finance`, `logistics`, `defense`, and `AI`, as well as `algorithm`, `model`, `simulation`, `optimization`, and `forecasting`. Combinations such as `climate AND algorithm` or `finance AND optimization` are included.

2. **Temporal Filter**  
   The initial dataset is limited to articles published in the last one to two years, for example, 2024–2025.

3. **Platform and Database Coverage**  
   Each domain is covered by two to four relevant scientific platforms or databases, as listed above.

4. **Initial Sample Size**  
   For every platform and keyword combination, the top 50 most relevant articles are selected. The expected total is 150 articles per domain.

---

## Screening, Extraction, and Clustering Pipeline

1. **Full Inclusion and Review**  
   All collected articles are included for analysis, regardless of whether algorithms are mentioned in the title or abstract. Each article is reviewed in detail.

2. **Extraction of Classical Algorithms**  
   From each article, all classical algorithms, models, or methods described or applied are extracted and recorded. If no algorithm is mentioned, this is also noted for completeness.

3. **Clustering and Name Normalization**  
   Extracted algorithm names are grouped into clusters that merge synonyms, variants, or closely related techniques. This ensures consistent reporting and accurate frequency analysis across the literature.

4. **Frequency Sorting and Ranking**  
   All algorithm clusters are counted and ranked according to their frequency of appearance across the collected articles in each domain.

---

## Algorithm Ranking and Significance Metrics

Algorithm significance within each domain is determined as follows:

- `A`: algorithm cluster  
- `D`: domain  
- `fpub(A, D)`: normalized frequency of `A` in `D`, i.e., count of `A` divided by the sum of counts for all clusters in `D`  
- `α`: weighting coefficient (default 1)  
- `fuse(A, D)`: binary indicator (1 if practical use is confirmed for `A` in `D`, 0 otherwise)

For each cluster with `fuse = 1`, the importance score is computed:  

I(A, D) = α * fpub(A, D)

A significance threshold `τ` (for example, 0.015) may be applied to highlight only the most relevant clusters for further analysis.

---

## Practical Use and Quantum Analogue Mapping

1. **Assessment of Practical Use (fuse Assignment)**  
   For each algorithm cluster, a practical use check is performed. This involves searching for evidence of real-world application (such as industry reports, case studies, open-source projects, or other credible sources).
   - Each cluster is assigned a binary score:
     - `fuse = 1` (practically used)
     - `fuse = 0` (no clear evidence of practical use)
   - Only clusters with `fuse = 1` are included in the quantum analogue mapping stage.

2. **Quantum Analogue Mapping**  
   For each cluster with `fuse = 1`, the literature and other resources are reviewed to determine the existence or development status of quantum analogues. The quantum status is categorized as:
   - **Exists**: a well-established quantum analogue is available
   - **In development**: an active research topic or prototype quantum analogue exists
   - **Not available**: no quantum analogue currently known

---

## Results Structure

- The main outputs for each domain include:
  - A table of all algorithm clusters, their frequencies, and practical use (`fuse`) scores
  - For each cluster with `fuse = 1`, a column indicating the quantum analogue status
  - Supporting notes and references for `fuse` and quantum assessments

- Example summary table:

| Algorithm Cluster | Frequency | fpub | fuse | Importance I(A, D) | Quantum Analogue          | Notes |
|-------------------|-----------|------|------|--------------------|---------------------------|-------|
| ...               | ...       | ...  | 1/0  | ...                | Exists / In development / Not available | ...   |

- Raw data, screening logs, and details of search queries are stored in subfolders for transparency and reproducibility.

---

## Limitations and Next Steps

Current limitations include restricted temporal and database coverage, occasional lack of full-text access, and the rapidly changing landscape of quantum algorithm research.  
Planned improvements include expanding the time window, adding new databases and domains, automating clustering and mapping, and inviting community validation of practical use and quantum status assignments.

---

## How to Contribute

Contributions are encouraged, including recommendations for new domains or databases, corrections or improvements to clustering and mapping, and the addition of articles or quantum analogues. Please use issues or pull requests, or contact the maintainer directly.

---

## Contact

Author and maintainer:  
Anastasiia Petrus  
anastasiiapetrus@gmail.com

---

This README and methodology will be updated as new domains, results, and insights are added.

