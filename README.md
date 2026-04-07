# SDO Event Prediction

## Overview

A machine learning project that predicts solar events using image data from NASA's **Solar Dynamics Observatory (SDO)**. The system queries SDO/AIA solar imagery parameters across multiple wavelengths and applies classification models to forecast solar events such as flares, coronal holes, and active regions.

Image parameters are fetched from the DMlab API and cross-referenced with solar event records from the ISD temporal event API to build labeled training data. Multiple ML models are benchmarked for binary event classification.

---

## Features

- Multi-wavelength solar image analysis across 9 AIA wavelength bands
- Extraction of 10 statistical and texture-based image features per observation
- Binary classification for 4 solar event types: Flares (FL), Coronal Holes (CH), Active Regions (AR), and Sunspots/Sigmoids (SG)
- Time-series data loading with configurable observation windows
- Multiple ML model implementations for comparative evaluation

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Data Retrieval | Requests (DMlab API + ISD Event API) |
| Data Processing | NumPy, Pandas, XML parsing |
| Models | Logistic Regression, Lasso Regression |
| Analysis | Jupyter Notebook |
| Data Source | NASA SDO/AIA (Solar Dynamics Observatory) |

**Image features used:** Entropy, Mean, Std. Deviation, Fractal Dimension, Skewness, Kurtosis, Uniformity, Relative Smoothness, Tamura Contrast, Tamura Directionality

**Wavelengths analyzed:** 94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å
