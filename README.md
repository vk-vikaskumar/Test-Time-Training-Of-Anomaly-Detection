# Test-Time Training of Anomaly Detection in Time Series (In Progress)  

## Overview  
This repository contains the implementation and research findings from my Master's thesis on **Test-Time Training for Anomaly Detection in Time Series**. The work builds on the concepts introduced in the paper [About Test-time training for outlier detection](https://arxiv.org/abs/2404.03495), adapting its methodology to time-series data (simulated [GUTENTAG](https://github.com/TimeEval/GutenTAG/tree/main) and real datasets).  

## Problem Statement  
Anomaly detection in time-series data is challenging, especially in real-world applications where data distributions shift over time. Traditional supervised models require extensive labeled data and retraining, which is impractical in dynamic environments.  

**Test-time training (TTT)** offers a promising solution by adapting the model dynamically at inference time, leveraging self-supervised learning to improve performance without requiring additional labeled data.  

However, directly applying TTT-based approaches like DOUST to time-series data presents several challenges:  
- **Feature extraction and selection:** Capturing temporal dependencies effectively.  
- **Loss function optimization:** Ensuring stability and generalization for time-series patterns.  
- **Explainability:** Understanding test-time adaptation behavior for trust and interpretability.  

## Solution Approach  
This thesis aims to **adapt and enhance the DOUST algorithm for time-series anomaly detection** by:  
✅ **Modifying loss functions** to improve adaptation stability.  
✅ **Optimizing feature extraction and selection** for time-series characteristics.  
✅ **Enhancing interpretability** using an ensemble-based approach to explain feature importance.  
✅ **Validating performance** on **simulated** and **real-world** datasets.  

## Status
🚧 In Progress: Active development and experimentation. Results and code will be updated as the research progresses.
