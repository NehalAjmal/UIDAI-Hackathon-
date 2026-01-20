import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# --- PART 1: DATA INGESTION (Handling Split Files) ---

def load_sharded_dataset(folder_name):
    """
    Reads all CSV files in a folder and combines them into one DataFrame.
    """
    # Construct path to folder (handles current directory context)
    path = os.path.join(os.getcwd(), folder_name)
    
    # Find all CSV files in that folder
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not all_files:
        print(f"WARNING: No CSV files found in {folder_name}")
        return pd.DataFrame() # Return empty if nothing found
    
    print(f"Loading {len(all_files)} files from {folder_name}...")
    
    # Read each file and append to a list
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(df)
    
    # Concatenate all chunks into one big DataFrame
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    return combined_df

# Load the three datasets based on your folder structure
print("--- Starting Data Load ---")
df_bio = load_sharded_dataset('api_data_aadhar_biometric')
df_demo = load_sharded_dataset('api_data_aadhar_demographic')
df_enrol = load_sharded_dataset('api_data_aadhar_enrolment')
print("--- Data Load Complete ---")

# --- PART 2: DATA PREPROCESSING ---

# 1. Clean Column Names (Strip whitespace just in case)
df_bio.columns = df_bio.columns.str.strip()
df_demo.columns = df_demo.columns.str.strip()
df_enrol.columns = df_enrol.columns.str.strip()

# 2. Aggregate Metrics by District
# Biometric Updates (Maintenance A)
# We assume columns are roughly: state, district, bio_age_5_17, bio_age_17_
# Adjusting to fill NaNs with 0 before summing
df_bio['total_bio'] = df_bio.get('bio_age_5_17', 0) + df_bio.get('bio_age_17_', 0)
bio_agg = df_bio.groupby(['state', 'district'])['total_bio'].sum().reset_index()

# Demographic Updates (Maintenance B)
df_demo['total_demo'] = df_demo.get('demo_age_5_17', 0) + df_demo.get('demo_age_17_', 0)
demo_agg = df_demo.groupby(['state', 'district'])['total_demo'].sum().reset_index()

# Enrolments (Growth)
# Summing 0-5, 5-17, and 18+
df_enrol['total_enrol'] = (
    df_enrol.get('age_0_5', 0) + 
    df_enrol.get('age_5_17', 0) + 
    df_enrol.get('age_18_greater', 0)
)
enrol_agg = df_enrol.groupby(['state', 'district'])['total_enrol'].sum().reset_index()

# --- PART 3: MERGING & SII CALCULATION ---

# Merge all into a Master DataFrame
master_df = enrol_agg.merge(bio_agg, on=['state', 'district'], how='outer').fillna(0)
master_df = master_df.merge(demo_agg, on=['state', 'district'], how='outer').fillna(0)

# Calculate Total Maintenance Load (Bio + Demo)
master_df['total_updates'] = master_df['total_bio'] + master_df['total_demo']

# Normalize (0 to 1 scale) for fair comparison
master_df['norm_enrol'] = master_df['total_enrol'] / master_df['total_enrol'].max()
master_df['norm_updates'] = master_df['total_updates'] / master_df['total_updates'].max()

# Service Intensity Index (SII)
# Formula: 40% weight to Growth, 60% to Maintenance
master_df['SII_Score'] = (0.4 * master_df['norm_enrol']) + (0.6 * master_df['norm_updates'])

# --- PART 4: AI/ML ENHANCEMENT (K-Means Clustering) ---

# Prepare data for ML
features = master_df[['norm_enrol', 'norm_updates']].copy()

# 1. Determine optimal clusters (Elbow Method simplified -> we'll use 4 for business logic)
# Cluster 0: Low Activity (Remote/Rural)
# Cluster 1: Maintenance Heavy (Metro/Migration Hubs)
# Cluster 2: Balanced/High Growth (Developing Cities)
# Cluster 3: Outliers/Super Critical

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
master_df['Cluster_ID'] = kmeans.fit_predict(features)

# Label the clusters meaningfully (Manual mapping based on centroids)
# We calculate mean values to assign names automatically
cluster_summary = master_df.groupby('Cluster_ID')[['norm_enrol', 'norm_updates']].mean()

def label_cluster(row):
    if row['norm_enrol'] > 0.5 and row['norm_updates'] > 0.5:
        return 'Critical High-Traffic'
    elif row['norm_updates'] > row['norm_enrol']:
        return 'Maintenance-Heavy (Urban)'
    elif row['norm_enrol'] > row['norm_updates']:
        return 'Growth-Driven (Rural/New)'
    else:
        return 'Low-Intensity / Stable'

# Apply names to the clusters dynamically
cluster_names = {}
for cluster_id, row in cluster_summary.iterrows():
    cluster_names[cluster_id] = label_cluster(row)

master_df['Cluster_Label'] = master_df['Cluster_ID'].map(cluster_names)

# --- PART 5: ADVANCED INSIGHTS GENERATION ---

print("\n=== ML-DRIVEN CLUSTER ANALYSIS ===")
print(master_df['Cluster_Label'].value_counts())

print("\nTop 3 Districts in 'Critical High-Traffic' Cluster:")
critical_districts = master_df[master_df['Cluster_Label'] == 'Critical High-Traffic']
print(critical_districts[['state', 'district', 'total_enrol', 'total_updates']].head(3))

# --- PART 6: PROFESSIONAL VISUALIZATION ---

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=master_df, 
    x='norm_enrol', 
    y='norm_updates', 
    hue='Cluster_Label',  # Color by ML Cluster
    style='Cluster_Label',
    palette='deep',
    s=100,
    alpha=0.7
)

# Annotate extreme outliers
top_outliers = master_df.sort_values(by='SII_Score', ascending=False).head(7)
for i in range(len(top_outliers)):
    row = top_outliers.iloc[i]
    plt.text(
        row['norm_enrol']+0.01, 
        row['norm_updates'], 
        row['district'], 
        fontsize=9, 
        weight='bold'
    )

plt.title('AI-Driven Segmentation of Aadhaar Service Demand', fontsize=14)
plt.xlabel('Normalized Enrolment Intensity (Growth)', fontsize=12)
plt.ylabel('Normalized Update Intensity (Maintenance)', fontsize=12)
plt.legend(title='District Service Persona', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig('ml_cluster_analysis.png')
print("\nAdvanced ML Plot saved as 'ml_cluster_analysis.png'.")