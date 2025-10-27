# SENTINEL - Multi-INT Early Warning Platform

**Synthetic Multi-Intelligence Fusion for Threat Detection and Geolocation**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

Advanced multi-intelligence fusion platform combining:
- **OPIR (Overhead Persistent Infrared)**: Thermal event detection and classification
- **RF/SIGINT**: Emitter geolocation and signal classification
- **Multi-INT Fusion**: Cross-correlation for enhanced threat assessment

⚠️ **Note:** This project uses ONLY synthetic data and public datasets. All scenarios are simulated.

---

##  Goal Technical Capabilities

### OPIR Module
- Thermal event simulation (launches, explosions, fires, aircraft)
- Background modeling with diurnal/seasonal variation
- Multi-algorithm detection (temporal differencing, anomaly detection, rise-time analysis)
- CNN-based event classification
- Kalman filter tracking

### RF Module
- Radar signal generation (various types: early warning, fire control, air search)
- Communication signal generation (FM, PSK, QAM)
- Energy detection and CFAR
- Feature extraction (pulse width, PRF, modulation parameters)
- CNN-based modulation classification
- TDOA/FDOA geolocation algorithms

### Fusion Module
- Temporal-spatial correlation
- Confidence scoring
- Event type classification
- Threat assessment

---

## Tech Stack

**Core:** Python 3.9+, NumPy, SciPy  
**ML:** PyTorch, scikit-learn  
**Signal Processing:** FilterPy, librosa  
**Geospatial:** GeoPandas, Folium  
**API:** FastAPI, Uvicorn  
**Dashboard:** Streamlit, Plotly  

---
##  Development Roadmap

**Phase 1: Foundation** COMPLETE 
- [X] Project setup and architecture 
- [X] Synthetic data generation (OPIR + RF) 
    - Started with 10,000 samples
- [X] Basic detection algorithms 

**Phase 2: Detection & Classification**
- [X] OPIR event detection
- [X] RF signal processing
- [X] ML model training

**Phase 3: Geolocation**
- TDOA/FDOA algorithms
- Multi-sensor fusion
- Uncertainty quantification

**Phase 4: Multi-INT Fusion**
- Cross-correlation algorithms
- Confidence scoring
- Threat assessment

**Phase 5: Production**
- API development
- Dashboard creation
- Documentation and testing


## Quick Start

*(Coming soon - Phase 2 in progress)*

---

##  Project Structure
```
sentinel-multi-int-platform/
├── src/
│   ├── opir/           # OPIR detection and classification
│   ├── rf/             # RF signal processing and geolocation
│   ├── fusion/         # Multi-INT fusion algorithms
│   └── utils/          # Shared utilities
├── models/             # Trained models
├── data/               # Synthetic data
├── notebooks/          # Jupyter notebooks for exploration
├── api/                # FastAPI endpoints
├── dashboard/          # Streamlit dashboard
└── tests/              # Unit tests
```

---

## Contact

**Michael Gurule**  
Data Scientist | ML Engineer  

- LinkedIn: [linkedin.com/in/michaelgurule](https://linkedin.com/in/michaelgurule)
- Email: michaelgurule1164@gmail.com
- GitHub: [github.com/michael-gurule](https://github.com/michael-gurule)

---

## ⚠️ Security & Ethics Notice

This project:
- Uses ONLY synthetic/simulated data
- Does NOT process classified information
- Does NOT access real sensor data
- Is for EDUCATIONAL purposes demonstrating signal processing and fusion techniques

All algorithms and methodologies are based on publicly available research and unclassified information.

---

*Building production-grade intelligence systems for portfolio demonstration*