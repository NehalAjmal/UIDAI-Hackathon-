# Smart-Aadhaar: AI-Driven Geospatial Segmentation for Service Optimization

## Project Overview
This project was developed for the **UIDAI Aadhaar Data-Driven Innovation Hackathon 2026**.

**Problem Statement:** Unlocking Societal Trends in Aadhaar Enrolment and Updates.

**Solution:** We developed a predictive analytical framework using **Unsupervised Machine Learning (K-Means Clustering)** to classify India's districts into distinct "Service Personas." By analyzing the ratio of new Enrolments (Growth) to Biometric/Demographic Updates (Maintenance), we identified specific infrastructure needs for different regions (e.g., Mega Kendras vs. Mobile Vans).

## Key Features
* **Data Aggregation:** Automated ingestion of sharded CSV datasets (Enrolment, Biometric, Demographic).
* **Service Intensity Index (SII):** A custom composite metric to quantify operational stress.
* **AI Segmentation:** K-Means clustering to identify high-pressure "Critical" districts vs. "Stable" zones.
* **Visualization:** Automated generation of a geospatial cluster scatter plot.

## Prerequisites
To run the analysis script, you need **Python 3.8+** and the following libraries:

* pandas
* matplotlib
* seaborn
* scikit-learn

You can install the dependencies using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Dataset Structure
The script expects the UIDAI datasets to be present in the root directory with the following folder structure (as provided in the hackathon resources):

```plaintext
/UIDAI-Hackathon/
│
├── analysis.py                    # The main source code
├── README.md                      # This file
│
├── api_data_aadhar_biometric/     # Folder containing biometric CSV shards
├── api_data_aadhar_demographic/   # Folder containing demographic CSV shards
└── api_data_aadhar_enrolment/     # Folder containing enrolment CSV shards
```

## How to Run
1. Open your terminal or command prompt.
2. Navigate to the project directory.
3. Run the python script:

```bash
python analysis.py
```

## Output
The script will generate two key outputs:

1. **Console Output:** A summary of the clusters and a list of the top 5 "Critical High-Traffic" districts (including their SII Scores).
2. **Visualization:** A PNG image named `ml_cluster_analysis_fixed.png` will be saved in the same directory. This plot visualizes the "Growth vs. Maintenance" clusters.

## Methodology
The core logic relies on the Service Intensity Index (SII):

```
SII = (0.4 × NormEnrolment) + (0.6 × NormUpdates)
```

* **Cluster A (Critical):** High SII, High Updates (Requires Permanent Infrastructure).
* **Cluster B (Growth):** High Enrolment, Low Updates (Requires Mobile Camps).
* **Cluster C (Stable):** Low Intensity (Requires Maintenance).

## Author
**Team Name/ID:** UIDAI_3526

**Team Members:->**
- Nehal Ajmal
- Navistha Pandey
- Pragati Tiwari
- Mohd Amaan