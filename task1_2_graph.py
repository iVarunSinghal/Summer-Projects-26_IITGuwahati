"""
=======================================================================
DELHIVERY GRAPH-BASED NETWORK INTELLIGENCE — TASKS 1 & 2
=======================================================================
Task 1 : Graph Construction & Data Pipeline
Task 2 : Bottleneck & Corridor Audit (Betweenness, Degree, Clustering,
         Chronically Delayed Corridors, SLA Breach Ranking)
=======================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────
print("=" * 65)
print("TASK 1 — GRAPH CONSTRUCTION & DATA PIPELINE")
print("=" * 65)

# Fixed Windows path using a raw string (r"...")
data_path = r"C:\Users\varun\Downloads\delivery_data.csv"

try:
    df = pd.read_csv(data_path)
    print(f"Successfully loaded data from: {data_path}")
except FileNotFoundError:
    print(f"Error: Could not find the file at {data_path}")
    print("Double-check that the filename is correct and you have permission to access it.")
    exit()

print(f"Raw rows: {len(df):,}   |   Columns: {len(df.columns)}")

# Parse datetime columns
for col in ['trip_creation_time','od_start_time','od_end_time','cutoff_timestamp']:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Derived features
df['hour_of_day']     = df['od_start_time'].dt.hour
df['day_of_week']     = df['od_start_time'].dt.dayofweek   # 0=Mon
df['time_bucket']     = pd.cut(df['hour_of_day'],
                               bins=[-1,5,11,17,23],
                               labels=['Night','Morning','Afternoon','Evening'])

df['delay_ratio']     = df['segment_actual_time'] / df['segment_osrm_time'].replace(0, np.nan)
df['sla_breach']      = df['segment_factor'] > 1.20       # >20% over OSRM
df['severe_breach']   = df['segment_factor'] > 1.50

# Drop rows with impossible times
df = df[(df['segment_actual_time'] > 0) & (df['segment_osrm_time'] > 0)]
df = df.dropna(subset=['source_center','destination_center'])

# Fill missing names
df['source_name']      = df['source_name'].fillna(df['source_center'])
df['destination_name'] = df['destination_name'].fillna(df['destination_center'])

print(f"Clean rows: {len(df):,}   |   Unique facilities: {df['source_center'].nunique() + df['destination_center'].nunique()}")
print(f"Training set: {(df['data']=='training').sum():,}   |   Test set: {(df['data']=='test').sum():,}")
print(f"FTL trips: {(df['route_type']=='FTL').sum():,}  |   Carting trips: {(df['route_type']=='Carting').sum():,}")
print(f"Overall SLA breach rate: {df['sla_breach'].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────
# 2. BUILD CORRIDOR-LEVEL EDGE TABLE
# ─────────────────────────────────────────────────────────────────────
print("\n— Building corridor-level edge weights...")

corridor = (
    df.groupby(['source_center','destination_center','route_type','time_bucket'])
    .agg(
        trip_count        = ('trip_uuid',    'count'),
        median_delay_ratio= ('delay_ratio',  'median'),
        mean_factor       = ('segment_factor','mean'),
        sla_breach_pct    = ('sla_breach',   'mean'),
        severe_breach_pct = ('severe_breach','mean'),
        avg_actual_min    = ('segment_actual_time','mean'),
        avg_osrm_min      = ('segment_osrm_time','mean'),
        avg_dist_km       = ('segment_osrm_distance','mean'),
    )
    .reset_index()
)

# Aggregate across time buckets for the graph edges (all-time view)
edge_df = (
    df.groupby(['source_center','destination_center','route_type'])
    .agg(
        trip_count        = ('trip_uuid',    'count'),
        median_delay_ratio= ('delay_ratio',  'median'),
        mean_factor       = ('segment_factor','mean'),
        sla_breach_pct    = ('sla_breach',   'mean'),
        severe_breach_pct = ('severe_breach','mean'),
        avg_actual_min    = ('segment_actual_time','mean'),
        avg_osrm_min      = ('segment_osrm_time','mean'),
        avg_dist_km       = ('segment_osrm_distance','mean'),
    )
    .reset_index()
)

# Weight = median actual-vs-OSRM delay ratio (captures real congestion)
edge_df['edge_weight'] = edge_df['median_delay_ratio'].clip(upper=50)

print(f"Total corridors (edges): {len(edge_df):,}")
print(f"Chronically delayed corridors (factor>1.2): "
      f"{(edge_df['mean_factor']>1.2).sum():,} "
      f"({(edge_df['mean_factor']>1.2).mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# 3. CONSTRUCT DIRECTED WEIGHTED GRAPH
# ─────────────────────────────────────────────────────────────────────
print("\n— Constructing directed weighted graph...")

G = nx.DiGraph()

# Add nodes with facility metadata
facility_stats = (
    df.groupby('source_center')
    .agg(
        outbound_trips    = ('trip_uuid',    'count'),
        avg_factor        = ('segment_factor','mean'),
        sla_breach_rate   = ('sla_breach',   'mean'),
        facility_name     = ('source_name',  'first'),
    )
    .reset_index()
    .rename(columns={'source_center':'center'})
)

for _, row in facility_stats.iterrows():
    G.add_node(row['center'],
               name=row['facility_name'],
               outbound_trips=row['outbound_trips'],
               avg_factor=row['avg_factor'],
               sla_breach_rate=row['sla_breach_rate'])

# Add edges
for _, row in edge_df.iterrows():
    G.add_edge(
        row['source_center'],
        row['destination_center'],
        route_type        = row['route_type'],
        weight            = row['edge_weight'],
        trip_count        = row['trip_count'],
        median_delay_ratio= row['median_delay_ratio'],
        mean_factor       = row['mean_factor'],
        sla_breach_pct    = row['sla_breach_pct'],
        avg_actual_min    = row['avg_actual_min'],
        avg_osrm_min      = row['avg_osrm_min'],
        avg_dist_km       = row['avg_dist_km'],
    )

print(f"Graph nodes (facilities): {G.number_of_nodes():,}")
print(f"Graph edges (corridors):  {G.number_of_edges():,}")
print(f"Weakly connected components: {nx.number_weakly_connected_components(G)}")

# Largest weakly connected component
largest_wcc = max(nx.weakly_connected_components(G), key=len)
G_lcc = G.subgraph(largest_wcc).copy()
print(f"Largest component — nodes: {G_lcc.number_of_nodes()}, edges: {G_lcc.number_of_edges()}")

# ─────────────────────────────────────────────────────────────────────
# TASK 2 — BOTTLENECK & CORRIDOR AUDIT
# ─────────────────────────────────────────────────────────────────────
print("\n")
print("=" * 65)
print("TASK 2 — BOTTLENECK & CORRIDOR AUDIT")
print("=" * 65)

# 2a) Degree centrality
in_degree  = dict(G.in_degree())
out_degree = dict(G.out_degree())
total_deg  = {n: in_degree.get(n,0) + out_degree.get(n,0) for n in G.nodes()}

# 2b) Betweenness centrality (sample for speed on large graph)
print("Computing betweenness centrality (sampled)...")
G_ug = G_lcc.to_undirected()
betweenness = nx.betweenness_centrality(G_lcc, normalized=True, k=min(500, G_lcc.number_of_nodes()))
nx.set_node_attributes(G_lcc, betweenness, 'betweenness')

# 2c) Clustering coefficient (on undirected projection)
clustering = nx.clustering(G_ug)

# 2d) PageRank (weighted by delay)
pagerank = nx.pagerank(G_lcc, weight='weight', alpha=0.85, max_iter=300)

# Build node-level metrics dataframe
node_metrics = pd.DataFrame({
    'center':        list(G.nodes()),
    'in_degree':     [in_degree.get(n,0)  for n in G.nodes()],
    'out_degree':    [out_degree.get(n,0) for n in G.nodes()],
    'total_degree':  [total_deg.get(n,0)  for n in G.nodes()],
    'betweenness':   [betweenness.get(n,0) for n in G.nodes()],
    'clustering':    [clustering.get(n,0)  for n in G.nodes()],
    'pagerank':      [pagerank.get(n,0)    for n in G.nodes()],
}).merge(facility_stats.rename(columns={'center':'center'}), on='center', how='left')

# Composite bottleneck score (normalise & combine)
def minmax(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)

node_metrics['bottleneck_score'] = (
    0.35 * minmax(node_metrics['betweenness']) +
    0.25 * minmax(node_metrics['sla_breach_rate'].fillna(0)) +
    0.20 * minmax(node_metrics['total_degree']) +
    0.20 * minmax(node_metrics['pagerank'])
)

node_metrics = node_metrics.sort_values('bottleneck_score', ascending=False)
print("\nTop 15 Bottleneck Hubs:")
print(node_metrics[['center','facility_name','in_degree','out_degree',
                     'betweenness','sla_breach_rate','bottleneck_score']].head(15).to_string(index=False))

# 2e) Chronic delay corridors (actual > OSRM by >20%)
chronic = edge_df[edge_df['mean_factor'] > 1.20].copy()
chronic = chronic.sort_values('sla_breach_pct', ascending=False)
print(f"\nChronic delay corridors (factor>1.2): {len(chronic)}")
print(chronic[['source_center','destination_center','route_type',
               'trip_count','mean_factor','sla_breach_pct','avg_dist_km']].head(15).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# VISUALISATION 1: Top Bottleneck Hubs
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')

top15 = node_metrics.head(15).copy()
top15['short_name'] = top15['facility_name'].apply(lambda x: str(x)[:25] if pd.notna(x) else 'Unknown')

colors = plt.cm.Reds(np.linspace(0.4, 0.95, 15))[::-1]

# Left: bottleneck score bar
bars = axes[0].barh(range(15), top15['bottleneck_score'].values, color=colors, edgecolor='#30363d')
axes[0].set_yticks(range(15))
axes[0].set_yticklabels(top15['short_name'].values, fontsize=8, color='#e6edf3')
axes[0].set_xlabel('Composite Bottleneck Score', color='#e6edf3', fontsize=10)
axes[0].set_title('Top 15 Bottleneck Hubs\n(Betweenness + SLA Breach + Degree + PageRank)',
                  color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
axes[0].tick_params(colors='#8b949e')
axes[0].spines['bottom'].set_color('#30363d')
axes[0].spines['left'].set_color('#30363d')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].invert_yaxis()

# Right: SLA breach rate scatter
sc = axes[1].scatter(
    top15['betweenness'] * 100,
    top15['sla_breach_rate'] * 100,
    s=top15['total_degree'] * 8 + 50,
    c=top15['bottleneck_score'],
    cmap='Reds', alpha=0.85, edgecolors='white', linewidths=0.5
)
for _, row in top15.head(8).iterrows():
    axes[1].annotate(row['short_name'][:18],
                     (row['betweenness']*100, row['sla_breach_rate']*100),
                     fontsize=6.5, color='#e6edf3',
                     xytext=(5, 3), textcoords='offset points')

axes[1].set_xlabel('Betweenness Centrality (%)', color='#e6edf3', fontsize=10)
axes[1].set_ylabel('SLA Breach Rate (%)', color='#e6edf3', fontsize=10)
axes[1].set_title('Betweenness vs SLA Breach Rate\n(bubble size = total degree)',
                  color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
axes[1].tick_params(colors='#8b949e')
for sp in axes[1].spines.values():
    sp.set_color('#30363d')
cbar = plt.colorbar(sc, ax=axes[1])
cbar.set_label('Bottleneck Score', color='#e6edf3', fontsize=8)
cbar.ax.yaxis.set_tick_params(color='#8b949e')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')

plt.suptitle('Delhivery Network — Bottleneck Hub Identification',
             color='#e6edf3', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

# Fixed Output Path
plt.savefig('fig_bottleneck_hubs.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("\nSaved: fig_bottleneck_hubs.png")

# ─────────────────────────────────────────────────────────────────────
# VISUALISATION 2: Delay heatmap by time of day & route type
# ─────────────────────────────────────────────────────────────────────
delay_heatmap = df.groupby(['time_bucket','route_type'])['segment_factor'].mean().unstack()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor('#0d1117')

# Heatmap
ax = axes[0]
ax.set_facecolor('#161b22')
heat_data = df.pivot_table(index='time_bucket', columns='route_type',
                           values='segment_factor', aggfunc='mean')
im = ax.imshow(heat_data.values, cmap='RdYlGn_r', aspect='auto', vmin=1.0, vmax=2.5)
ax.set_xticks(range(len(heat_data.columns)))
ax.set_xticklabels(heat_data.columns, color='#e6edf3', fontsize=11)
ax.set_yticks(range(len(heat_data.index)))
ax.set_yticklabels(heat_data.index, color='#e6edf3', fontsize=11)
for i in range(len(heat_data.index)):
    for j in range(len(heat_data.columns)):
        ax.text(j, i, f'{heat_data.values[i,j]:.2f}',
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white')
plt.colorbar(im, ax=ax, label='Avg Delay Factor')
ax.set_title('Delay Factor by Time of Day & Route Type',
             color='#e6edf3', fontsize=11, fontweight='bold', pad=10)

# SLA breach by hour
ax2 = axes[1]
ax2.set_facecolor('#161b22')
hourly = df.groupby(['hour_of_day','route_type'])['sla_breach'].mean().reset_index()
for rt, grp in hourly.groupby('route_type'):
    color = '#FF6B6B' if rt == 'FTL' else '#4ECDC4'
    ax2.plot(grp['hour_of_day'], grp['sla_breach']*100,
             marker='o', markersize=4, label=rt, color=color, linewidth=2)
ax2.set_xlabel('Hour of Day', color='#e6edf3', fontsize=10)
ax2.set_ylabel('SLA Breach Rate (%)', color='#e6edf3', fontsize=10)
ax2.set_title('SLA Breach Rate by Hour of Day', color='#e6edf3', fontsize=11, fontweight='bold')
ax2.tick_params(colors='#8b949e')
ax2.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=9)
ax2.set_xticks(range(0, 24, 2))
for sp in ax2.spines.values(): sp.set_color('#30363d')
ax2.grid(alpha=0.15, color='white')
ax2.axhline(y=80, color='red', linestyle='--', alpha=0.4, linewidth=1)
ax2.text(0.5, 81, '80% breach line', color='#FF6B6B', fontsize=8)

plt.suptitle('Temporal Delay Patterns across the Delhivery Network',
             color='#e6edf3', fontsize=13, fontweight='bold')
plt.tight_layout()

# Fixed Output Path
plt.savefig('fig_temporal_delay.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("Saved: fig_temporal_delay.png")

# ─────────────────────────────────────────────────────────────────────
# VISUALISATION 3: Corridor delay distribution
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor('#0d1117')

ax = axes[0]
ax.set_facecolor('#161b22')
for rt, color in [('FTL','#FF6B6B'), ('Carting','#4ECDC4')]:
    sub = edge_df[edge_df['route_type']==rt]['mean_factor'].clip(0,5)
    ax.hist(sub, bins=50, alpha=0.7, color=color, label=rt, density=True, edgecolor='none')
ax.axvline(x=1.2, color='yellow', linestyle='--', linewidth=1.5, label='SLA threshold (1.2x)')
ax.set_xlabel('Mean Delay Factor (Actual/OSRM)', color='#e6edf3')
ax.set_ylabel('Density', color='#e6edf3')
ax.set_title('Distribution of Corridor Delay Factors', color='#e6edf3', fontweight='bold')
ax.tick_params(colors='#8b949e')
ax.legend(facecolor='#161b22', labelcolor='#e6edf3')
for sp in ax.spines.values(): sp.set_color('#30363d')

ax2 = axes[1]
ax2.set_facecolor('#161b22')
top10_chronic = chronic.head(10)
top10_chronic['corridor'] = top10_chronic['source_center'].str[:8] + ' → ' + top10_chronic['destination_center'].str[:8]
ax2.barh(range(len(top10_chronic)), top10_chronic['sla_breach_pct']*100,
         color=plt.cm.Reds(np.linspace(0.5,0.9,10))[::-1])
ax2.set_yticks(range(len(top10_chronic)))
ax2.set_yticklabels(top10_chronic['corridor'].values, fontsize=8, color='#e6edf3')
ax2.set_xlabel('SLA Breach Rate (%)', color='#e6edf3')
ax2.set_title('Top 10 Chronically Delayed Corridors\n(Factor > 1.2x)', color='#e6edf3', fontweight='bold')
ax2.tick_params(colors='#8b949e')
ax2.invert_yaxis()
for sp in ax2.spines.values(): sp.set_color('#30363d')

plt.tight_layout()

# Fixed Output Path
plt.savefig('fig_corridor_delay.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("Saved: fig_corridor_delay.png")

# ─────────────────────────────────────────────────────────────────────
# SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────
# Fixed Output Paths to save in your current directory
node_metrics.to_csv('node_metrics.csv', index=False)
edge_df.to_csv('corridor_edge_table.csv', index=False)
chronic.to_csv('chronic_corridors.csv', index=False)

# Save graph
nx.write_gexf(G, 'delhivery_network.gexf')

# Fixed missing closing parenthesis/quote
print("\n[SUCCESS] Task 1 & 2 complete. Artifacts saved.")
print(f"  node_metrics.csv        — {len(node_metrics)} facilities")
print(f"  corridor_edge_table.csv — {len(edge_df)} corridors")
print(f"  chronic_corridors.csv   — {len(chronic)} chronic corridors")
print(f"  delhivery_network.gexf  — full graph")