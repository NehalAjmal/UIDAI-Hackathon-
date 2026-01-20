import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Set style for professional-looking plots
sns.set_theme(style="whitegrid")

# --- STEP 1: DATA INGESTION ---
# We use this function to handle the "sharded" files (split into multiple parts)
def load_sharded_dataset(folder_name):
    """
    Scans a folder, finds all CSV fragments, and merges them into one clean dataset.
    """
    path = os.path.join(os.getcwd(), folder_name)
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not all_files:
        print(f"⚠️  Warning: No data found in {folder_name}")
        return pd.DataFrame()
    
    print(f"   -> Loading {len(all_files)} file shards from {folder_name}...")
    
    # Read and combine
    df_list = [pd.read_csv(f, index_col=None, header=0) for f in all_files]
    return pd.concat(df_list, axis=0, ignore_index=True)

print("\n--- 1. Starting Data Ingestion ---")
df_bio = load_sharded_dataset('api_data_aadhar_biometric')
df_demo = load_sharded_dataset('api_data_aadhar_demographic')
df_enrol = load_sharded_dataset('api_data_aadhar_enrolment')
print("--- Data Ingestion Complete ---\n")

# --- STEP 2: DATA PREPROCESSING ---
# Clean up column names to avoid errors with hidden spaces
df_bio.columns = df_bio.columns.str.strip()
df_demo.columns = df_demo.columns.str.strip()
df_enrol.columns = df_enrol.columns.str.strip()

# Aggregate Data by District
# We sum up the biometric updates (Bio), demographic updates (Demo), and new enrolments (Enrol)
print("--- 2. Aggregating District-Level Metrics ---")

# Biometric Updates (Maintenance Type A)
df_bio['total_bio'] = df_bio.get('bio_age_5_17', 0) + df_bio.get('bio_age_17_', 0)
bio_agg = df_bio.groupby(['state', 'district'])['total_bio'].sum().reset_index()

# Demographic Updates (Maintenance Type B)
df_demo['total_demo'] = df_demo.get('demo_age_5_17', 0) + df_demo.get('demo_age_17_', 0)
demo_agg = df_demo.groupby(['state', 'district'])['total_demo'].sum().reset_index()

# New Enrolments (Growth)
df_enrol['total_enrol'] = (
    df_enrol.get('age_0_5', 0) + 
    df_enrol.get('age_5_17', 0) + 
    df_enrol.get('age_18_greater', 0)
)
enrol_agg = df_enrol.groupby(['state', 'district'])['total_enrol'].sum().reset_index()

# --- STEP 3: CREATING THE MASTER DATASET ---
# Merge all three lists into one master view of every district
master_df = enrol_agg.merge(bio_agg, on=['state', 'district'], how='outer').fillna(0)
master_df = master_df.merge(demo_agg, on=['state', 'district'], how='outer').fillna(0)

# Calculate Indicators
master_df['total_updates'] = master_df['total_bio'] + master_df['total_demo']

# Normalize values (0 to 1) so we can compare small districts vs. big cities fairly
master_df['norm_enrol'] = master_df['total_enrol'] / master_df['total_enrol'].max()
master_df['norm_updates'] = master_df['total_updates'] / master_df['total_updates'].max()

# Calculate Service Intensity Index (SII)
# We give 60% weight to Updates because they are more operationally complex
master_df['SII_Score'] = (0.4 * master_df['norm_enrol']) + (0.6 * master_df['norm_updates'])

# --- STEP 4: AI SEGMENTATION (K-MEANS) ---
print("--- 3. Running AI Clustering Model ---")

# Prepare features for the AI
features = master_df[['norm_enrol', 'norm_updates']].copy()

# Run K-Means Clustering (Bucketing districts into 4 types)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
master_df['Cluster_ID'] = kmeans.fit_predict(features)

# Dynamic Labeling: Name the clusters based on their actual data
cluster_stats = master_df.groupby('Cluster_ID')[['SII_Score', 'norm_enrol', 'norm_updates']].mean()
sorted_clusters = cluster_stats.sort_values('SII_Score', ascending=False).index.tolist()

cluster_names = {}
# The cluster with the highest score is "Critical"
cluster_names[sorted_clusters[0]] = 'Critical High-Traffic (Metro)'
# The cluster with the lowest score is "Stable"
cluster_names[sorted_clusters[-1]] = 'Low-Intensity / Stable'

# The middle clusters are defined by whether they have more Enrolment or Updates
for cluster_id in sorted_clusters[1:-1]:
    row = cluster_stats.loc[cluster_id]
    if row['norm_updates'] > row['norm_enrol']:
        cluster_names[cluster_id] = 'Maintenance-Heavy (Urban)'
    else:
        cluster_names[cluster_id] = 'Growth-Driven (Rural/Dev)'

master_df['Cluster_Label'] = master_df['Cluster_ID'].map(cluster_names)

# --- STEP 5: GENERATING INSIGHTS ---
print("\n=== 🎯 AI ANALYSIS RESULTS ===")
print("District Segmentation Breakdown:")
print(master_df['Cluster_Label'].value_counts())

print("\n⚠️ TOP 5 CRITICAL DISTRICTS (Require Immediate Infrastructure):")
critical = master_df[master_df['Cluster_Label'] == 'Critical High-Traffic (Metro)']

# FIX: We keep 'norm_enrol' and 'norm_updates' so we can use them for plotting later
top_5 = critical[['state', 'district', 'SII_Score', 'total_enrol', 'total_updates', 'norm_enrol', 'norm_updates']] \
          .sort_values('SII_Score', ascending=False).head(5)

# We print only the relevant columns for the report (hiding the normalized ones for cleanliness)
print(top_5[['state', 'district', 'SII_Score', 'total_enrol', 'total_updates']].to_string(index=False))

# --- STEP 6: VISUALIZATION ---
print("\n--- 4. Generating Visual Reports ---")

# Plot 1: The Segmentation Map
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

# Annotate key cities (NOW THIS WILL WORK)
for i in range(len(top_5)):
    row = top_5.iloc[i]
    # We add a small offset (+0.01) so the text doesn't cover the dot
    plt.text(row['norm_enrol']+0.01, row['norm_updates'], row['district'], fontsize=9, weight='bold')

plt.title('AI-Driven Service Segmentation', fontsize=14)
plt.xlabel('Growth Demand (Normalized)', fontsize=12)
plt.ylabel('Maintenance Demand (Normalized)', fontsize=12)
plt.legend(title='District Persona')
plt.grid(True, alpha=0.3)
plt.savefig('ml_cluster_analysis_fixed.png')
print("   -> Saved 'ml_cluster_analysis_fixed.png'")

# Plot 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    master_df[['norm_enrol', 'norm_updates', 'SII_Score']].corr(), 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f"
)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.savefig('stat_correlation.png')
plt.close()
print("   -> Saved 'stat_correlation.png'")

# Plot 3: Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=master_df, 
    x='Cluster_Label', 
    y='SII_Score', 
    hue='Cluster_Label',
    palette='Set2',
    legend=False
)
plt.title('Stress Distribution by Cluster', fontsize=16)
plt.xlabel('')
plt.ylabel('SII Score')
plt.xticks(fontsize=10)
plt.savefig('stat_violin.png')
plt.close()
print("   -> Saved 'stat_violin.png'")

# Plot 4: Elbow Curve
inertia = []
K_range = range(1, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(features)
    inertia.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o', linestyle='--', color='purple', linewidth=3)
plt.title('K-Means Elbow Optimization', fontsize=16)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Model Error')
plt.annotate('Optimal k=3', xy=(3, inertia[2]), xytext=(4, inertia[2]+10),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.grid(True, alpha=0.3)
plt.savefig('stat_elbow.png')
plt.close()
print("   -> Saved 'stat_elbow.png'")

# Plot 5: Density Plot
plt.figure(figsize=(10, 8))
sns.kdeplot(
    data=master_df,
    x='norm_enrol',
    y='norm_updates',
    fill=True,
    cmap='viridis',
    thresh=0.05
)
plt.title('Geospatial Demand Density', fontsize=16)
plt.xlabel('Enrolment Intensity')
plt.ylabel('Update Intensity')
plt.savefig('stat_density.png')
plt.close()
print("   -> Saved 'stat_density.png'")

print("\n✅ Analysis Complete. All reports generated.")