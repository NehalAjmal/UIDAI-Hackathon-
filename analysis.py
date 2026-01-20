import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Set style for clean, professional plots
sns.set_theme(style="whitegrid")

# --- STEP 1: DATA INGESTION ---
def load_sharded_dataset(folder_name):
    path = os.path.join(os.getcwd(), folder_name)
    all_files = glob.glob(os.path.join(path, "*.csv"))
    if not all_files:
        return pd.DataFrame()
    df_list = [pd.read_csv(f, index_col=None, header=0) for f in all_files]
    return pd.concat(df_list, axis=0, ignore_index=True)

df_bio = load_sharded_dataset('api_data_aadhar_biometric')
df_demo = load_sharded_dataset('api_data_aadhar_demographic')
df_enrol = load_sharded_dataset('api_data_aadhar_enrolment')

# --- STEP 2: PREPROCESSING ---
df_bio.columns = df_bio.columns.str.strip()
df_demo.columns = df_demo.columns.str.strip()
df_enrol.columns = df_enrol.columns.str.strip()

# Aggregate Data
df_bio['total_bio'] = df_bio.get('bio_age_5_17', 0) + df_bio.get('bio_age_17_', 0)
bio_agg = df_bio.groupby(['state', 'district'])['total_bio'].sum().reset_index()

df_demo['total_demo'] = df_demo.get('demo_age_5_17', 0) + df_demo.get('demo_age_17_', 0)
demo_agg = df_demo.groupby(['state', 'district'])['total_demo'].sum().reset_index()

df_enrol['total_enrol'] = (
    df_enrol.get('age_0_5', 0) + 
    df_enrol.get('age_5_17', 0) + 
    df_enrol.get('age_18_greater', 0)
)
enrol_agg = df_enrol.groupby(['state', 'district'])['total_enrol'].sum().reset_index()

# --- STEP 3: MASTER DATASET ---
master_df = enrol_agg.merge(bio_agg, on=['state', 'district'], how='outer').fillna(0)
master_df = master_df.merge(demo_agg, on=['state', 'district'], how='outer').fillna(0)
master_df['total_updates'] = master_df['total_bio'] + master_df['total_demo']

# Normalize
master_df['norm_enrol'] = master_df['total_enrol'] / master_df['total_enrol'].max()
master_df['norm_updates'] = master_df['total_updates'] / master_df['total_updates'].max()

# SII Score (Stress Score)
master_df['SII_Score'] = (0.4 * master_df['norm_enrol']) + (0.6 * master_df['norm_updates'])

# --- STEP 4: AI CLUSTERING ---
features = master_df[['norm_enrol', 'norm_updates']].copy()
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
master_df['Cluster_ID'] = kmeans.fit_predict(features)

# Label Clusters with Simple Names
cluster_stats = master_df.groupby('Cluster_ID')[['SII_Score', 'norm_enrol', 'norm_updates']].mean()
sorted_clusters = cluster_stats.sort_values('SII_Score', ascending=False).index.tolist()

cluster_names = {}
cluster_names[sorted_clusters[0]] = 'Critical Zone (High Traffic)'
cluster_names[sorted_clusters[-1]] = 'Stable Zone (Low Traffic)'

for cluster_id in sorted_clusters[1:-1]:
    row = cluster_stats.loc[cluster_id]
    if row['norm_updates'] > row['norm_enrol']:
        cluster_names[cluster_id] = 'Migration Hub (High Updates)'
    else:
        cluster_names[cluster_id] = 'Growth Hub (New Enrolments)'

master_df['Cluster_Label'] = master_df['Cluster_ID'].map(cluster_names)

# --- STEP 5: VISUALIZATION (ALL GRAPHS) ---
print("Generating 5 Human-Readable Reports...")

# 1. SEGMENTATION MAP (Main Result)
plt.figure(figsize=(12, 8))
top_5 = master_df[master_df['Cluster_Label'] == 'Critical Zone (High Traffic)'].sort_values('SII_Score', ascending=False).head(5)
sns.scatterplot(data=master_df, x='norm_enrol', y='norm_updates', hue='Cluster_Label', style='Cluster_Label', palette='Set1', s=100, alpha=0.7)

for i in range(len(top_5)):
    row = top_5.iloc[i]
    plt.text(row['norm_enrol']+0.01, row['norm_updates'], row['district'], fontsize=10, weight='bold')

plt.title('District Classification: Where is the Infrastructure Need?', fontsize=16)
plt.xlabel('New User Growth (Children Enrolments)', fontsize=12)
plt.ylabel('Maintenance Load (Updates & Corrections)', fontsize=12)
plt.legend(title='District Type')
plt.savefig('ml_cluster_analysis_fixed.png')
print("- Saved Map")

# 2. DENSITY PLOT (Where is the crowd?)
plt.figure(figsize=(10, 8))
sns.kdeplot(data=master_df, x='norm_enrol', y='norm_updates', fill=True, cmap='viridis', thresh=0.05)
plt.title('Demand Hotspots: Where is the Pressure?', fontsize=16)
plt.xlabel('New User Growth', fontsize=12)
plt.ylabel('Maintenance Load', fontsize=12)
plt.savefig('stat_density.png')
print("- Saved Density Plot")

# 3. ELBOW CURVE (Accuracy Check)
inertia = []
K_range = range(1, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(features)
    inertia.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o', linestyle='--', color='purple', linewidth=3)
plt.title('AI Accuracy Check (Optimal Grouping)', fontsize=16)
plt.xlabel('Number of Groups', fontsize=12)
plt.ylabel('Error Rate (Lower is Better)', fontsize=12)
plt.annotate('Best Balance (k=3)', xy=(3, inertia[2]), xytext=(4, inertia[2]+10), arrowprops=dict(facecolor='black', shrink=0.05))
plt.grid(True, alpha=0.3)
plt.savefig('stat_elbow.png')
print("- Saved Accuracy Check")

# 4. VIOLIN PLOT (Stress Distribution)
plt.figure(figsize=(12, 6))
sns.violinplot(data=master_df, x='Cluster_Label', y='SII_Score', hue='Cluster_Label', palette='Set2', legend=False)
plt.title('Stress Levels per District Type', fontsize=16)
plt.xlabel('')
plt.ylabel('Operational Stress Score (0-1)')
plt.savefig('stat_violin.png')
print("- Saved Violin Plot")

# 5. CORRELATION (Data Check)
plt.figure(figsize=(10, 8))
sns.heatmap(master_df[['norm_enrol', 'norm_updates', 'SII_Score']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Data Relationship Check', fontsize=16)
plt.savefig('stat_correlation.png')
print("- Saved Data Check")

print("Done! All 5 images are ready.")