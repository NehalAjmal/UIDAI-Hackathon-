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

# --- PART 4: ROBUST CLUSTERING & LABELING ---

from sklearn.cluster import KMeans

# Prepare data
features = master_df[['norm_enrol', 'norm_updates']].copy()

# Run K-Means (4 Clusters)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
master_df['Cluster_ID'] = kmeans.fit_predict(features)

# --- THE FIX: DYNAMIC LABELING BASED ON RANK ---
# Instead of guessing thresholds, we rank clusters by their "Intensity"

# 1. Calculate the mean SII_Score for each cluster
cluster_stats = master_df.groupby('Cluster_ID')[['SII_Score', 'norm_enrol', 'norm_updates']].mean()

# 2. Sort clusters from Highest Impact to Lowest
sorted_clusters = cluster_stats.sort_values('SII_Score', ascending=False).index.tolist()

# 3. Assign labels based on Rank
cluster_names = {}

# Rank 1: The cluster with the highest scores
cluster_names[sorted_clusters[0]] = 'Critical High-Traffic (Metro)'

# Rank 4: The cluster with the lowest scores
cluster_names[sorted_clusters[-1]] = 'Low-Intensity / Stable'

# Middle Clusters: Distinguish by Type (Growth vs Maintenance)
for cluster_id in sorted_clusters[1:-1]:
    row = cluster_stats.loc[cluster_id]
    # If Updates are dominant relative to Enrolment
    if row['norm_updates'] > row['norm_enrol']:
        cluster_names[cluster_id] = 'Maintenance-Heavy (Urban)'
    else:
        cluster_names[cluster_id] = 'Growth-Driven (Rural/Dev)'

# Map the names back
master_df['Cluster_Label'] = master_df['Cluster_ID'].map(cluster_names)

# --- PART 5: INSIGHTS & VISUALIZATION ---

print("\n=== CORRECTED CLUSTER RESULTS ===")
print(master_df['Cluster_Label'].value_counts())

print("\nTop 5 Districts in 'Critical High-Traffic' Cluster:")
critical = master_df[master_df['Cluster_Label'] == 'Critical High-Traffic (Metro)']
print(critical[['state', 'district', 'SII_Score', 'total_enrol', 'total_updates']].sort_values('SII_Score', ascending=False).head(5))

# Visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=master_df, 
    x='norm_enrol', 
    y='norm_updates', 
    hue='Cluster_Label', 
    style='Cluster_Label',
    palette='Set1',
    s=100, 
    alpha=0.7
)

# Annotate Top 3 Critical Districts
top_3 = critical.sort_values('SII_Score', ascending=False).head(3)
for i in range(len(top_3)):
    row = top_3.iloc[i]
    plt.text(
        row['norm_enrol'], 
        row['norm_updates'], 
        row['district'], 
        fontsize=10, 
        weight='bold'
    )

plt.title('AI-Driven Service Segmentation (Corrected)', fontsize=14)
plt.xlabel('Growth Demand (Normalized)', fontsize=12)
plt.ylabel('Maintenance Demand (Normalized)', fontsize=12)
plt.legend(title='District Persona')
plt.grid(True, alpha=0.3)

plt.savefig('ml_cluster_analysis_fixed.png')
print("\nFixed plot saved as 'ml_cluster_analysis_fixed.png'.")