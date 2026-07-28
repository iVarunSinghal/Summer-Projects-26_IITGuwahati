"""
=======================================================================
TASK 5 — NETWORK OPERATIONS STRATEGY MEMO
=======================================================================
Audience : Head of Network Operations, Delhivery
Format   : Business consulting deliverable (2 pages)
Content  : Top 5 bottleneck hubs, corridor interventions,
           quantified impact, route-type strategic insight
=======================================================================
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json, textwrap, datetime

# ─────────────────────────────────────────────────────────────────────
# LOAD ALL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────
df           = pd.read_csv('delivery_data.csv')
node_metrics = pd.read_csv('node_metrics.csv')
edge_df      = pd.read_csv('corridor_edge_table.csv')
chronic      = pd.read_csv('chronic_corridors.csv')

for col in ['od_start_time']:
    df[col] = pd.to_datetime(df[col], errors='coerce')
df['hour_of_day'] = df['od_start_time'].dt.hour
df['sla_breach']  = (df['segment_factor'] > 1.20).astype(int)
df = df[(df['segment_actual_time'] > 0) & (df['segment_osrm_time'] > 0)]
df = df.dropna(subset=['source_center','destination_center'])

with open('revenue_impact.json') as f:
    rev = json.load(f)
with open('enhanced_task3_results.json') as f:
    eta = json.load(f)

# ─────────────────────────────────────────────────────────────────────
# COMPUTE MEMO NUMBERS
# ─────────────────────────────────────────────────────────────────────
nm = node_metrics.copy()
top5 = nm.head(5).copy()
top5 = top5.merge(
    df.groupby('source_center').agg(
        total_trips   = ('trip_uuid','count'),
        breaches      = ('sla_breach','sum'),
        breach_rate   = ('sla_breach','mean'),
        avg_delay     = ('segment_factor','mean'),
    ).reset_index().rename(columns={'source_center':'center'}),
    on='center', how='left'
)
top5['breach_contrib_pct'] = top5['breaches'] / df['sla_breach'].sum() * 100
top5['rev_risk_lakh']      = top5['breaches'] * rev['penalty_per_breach_inr'] / 1e5
top5['savings_lakh']       = top5['rev_risk_lakh'] * rev['assumed_upgrade_improvement_pct'] / 100

# FTL vs Carting insight
carting_high_breach = (
    df[df['route_type']=='Carting']
    .groupby(['source_center','destination_center'])
    .agg(trips=('trip_uuid','count'), breach_rate=('sla_breach','mean'))
    .reset_index()
)
carting_convert = carting_high_breach[
    (carting_high_breach['breach_rate'] > 0.90) &
    (carting_high_breach['trips'] > 20)
].sort_values('breach_rate', ascending=False)
print(f"Carting corridors to consider converting to FTL: {len(carting_convert)}")

# Best time windows
hourly = df.groupby('hour_of_day')['sla_breach'].mean()
best_hours = hourly.nsmallest(4).index.tolist()
worst_hours = hourly.nlargest(4).index.tolist()

print(f"\nTop 5 Bottleneck Hubs for Memo:")
print(top5[['facility_name','breach_rate','breach_contrib_pct','rev_risk_lakh','savings_lakh']].to_string(index=False))
print(f"\nBest dispatch hours: {best_hours}")
print(f"Worst dispatch hours: {worst_hours}")

# ─────────────────────────────────────────────────────────────────────
# BUILD STRATEGY MEMO — FULL 2-PAGE PDF-STYLE FIGURE
# ─────────────────────────────────────────────────────────────────────
BRAND_DARK  = '#0D1B2A'
BRAND_MID   = '#1B2A3B'
BRAND_LIGHT = '#2C3E50'
ACCENT_RED  = '#E63946'
ACCENT_TEAL = '#2EC4B6'
ACCENT_GOLD = '#F4A261'
TEXT_LIGHT  = '#E8F4F8'
TEXT_MID    = '#BDC9D1'
TEXT_FAINT  = '#7F9CAB'

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor(BRAND_DARK)

# ─── HEADER ─────────────────────────────────────────────────────────
ax_hdr = fig.add_axes([0, 0.955, 1, 0.045])
ax_hdr.set_facecolor(ACCENT_RED)
ax_hdr.text(0.02, 0.55, 'DELHIVERY  |  NETWORK OPERATIONS STRATEGY MEMO', 
            color='white', fontsize=12, fontweight='bold', va='center', 
            transform=ax_hdr.transAxes)
ax_hdr.text(0.98, 0.55, f'CONFIDENTIAL  |  {datetime.date.today().strftime("%B %Y")}',
            color=(1,1,1,0.8), fontsize=9, va='center', ha='right',
            transform=ax_hdr.transAxes)
ax_hdr.axis('off')

# ─── TITLE BLOCK ────────────────────────────────────────────────────
ax_title = fig.add_axes([0.03, 0.895, 0.94, 0.055])
ax_title.set_facecolor(BRAND_DARK)
ax_title.text(0.0, 0.85, 'Optimizing Delivery ETAs with Graph-Based Network Intelligence',
              color=TEXT_LIGHT, fontsize=17, fontweight='bold', va='top',
              transform=ax_title.transAxes)
ax_title.text(0.0, 0.35,
              'TO: Head of Network Operations  |  FROM: Data Science Team  |  SUBJECT: Priority Hub Interventions & ETA Improvement Roadmap',
              color=TEXT_MID, fontsize=9, va='top', transform=ax_title.transAxes)
ax_title.axis('off')

# ─── DIVIDER ────────────────────────────────────────────────────────
ax_div = fig.add_axes([0.03, 0.892, 0.94, 0.002])
ax_div.set_facecolor(ACCENT_RED)
ax_div.axis('off')

# ─── KPI STRIP ──────────────────────────────────────────────────────
kpis = [
    ('84.5%', 'Network SLA\nBreach Rate', ACCENT_RED),
    ('120,178', 'Total SLA Breaches\n(dataset period)', ACCENT_GOLD),
    (f"₹{rev['total_revenue_at_risk_lakhs']:.0f}L", 'Revenue\nat Risk', ACCENT_RED),
    ('14.9%', 'MAE Improvement\n(Graph vs Baseline)', ACCENT_TEAL),
    ('39.1%', 'Within-15%\nAccuracy (Graph)', ACCENT_TEAL),
    (f"₹{rev['top3_hub_savings_lakhs']:.0f}L", 'Recoverable\n(Top 3 Hubs)', ACCENT_TEAL),
]
for i, (val, label, color) in enumerate(kpis):
    x = 0.03 + i * 0.156
    ax_k = fig.add_axes([x, 0.830, 0.148, 0.058])
    ax_k.set_facecolor(BRAND_MID)
    ax_k.text(0.5, 0.72, val,   color=color,    fontsize=16, fontweight='bold',
              ha='center', va='top', transform=ax_k.transAxes)
    ax_k.text(0.5, 0.25, label, color=TEXT_MID, fontsize=7.5,
              ha='center', va='bottom', transform=ax_k.transAxes, linespacing=1.4)
    ax_k.set_xlim(0,1); ax_k.set_ylim(0,1)
    ax_k.add_patch(mpatches.FancyBboxPatch((0,0),1,1,
                   boxstyle='round,pad=0.02', facecolor='none',
                   edgecolor=color, linewidth=1.2, transform=ax_k.transAxes))
    ax_k.axis('off')

# ─── SECTION A: EXECUTIVE SUMMARY ───────────────────────────────────
ax_a = fig.add_axes([0.03, 0.742, 0.44, 0.082])
ax_a.set_facecolor(BRAND_MID)
ax_a.text(0.015, 0.94, '▌ EXECUTIVE SUMMARY', color=ACCENT_TEAL,
          fontsize=10, fontweight='bold', va='top', transform=ax_a.transAxes)

summary = (
    "Our graph-based analysis of 142,267 trip segments across Delhivery's logistics "
    "network reveals that 84.5% of all trips breach the SLA threshold of 1.2× OSRM "
    "estimated time. A directed weighted graph of 1,657 facilities and 2,783 corridors "
    "was constructed, and composite bottleneck scoring (betweenness centrality, "
    "PageRank, SLA breach rate, degree) identified 5 critical hubs that together "
    "account for 35.2% of all SLA breaches. Upgrading the top 3 hubs alone can recover "
    f"₹{rev['top3_hub_savings_lakhs']:.0f}L in avoided penalty exposure. Our graph-enhanced "
    "ETA model reduces prediction error by 14.9% (MAE: 13.0→11.1 min) versus the "
    "OSRM-only baseline, enabling smarter dispatch decisions network-wide."
)
wrapped = textwrap.fill(summary, width=88)
ax_a.text(0.015, 0.72, wrapped, color=TEXT_LIGHT, fontsize=7.8, va='top',
          transform=ax_a.transAxes, linespacing=1.5)
ax_a.axis('off')

# ─── SECTION B: ROUTE TYPE INSIGHT ──────────────────────────────────
ax_b = fig.add_axes([0.49, 0.742, 0.44, 0.082])
ax_b.set_facecolor(BRAND_MID)
ax_b.text(0.015, 0.94, '▌ KEY FINDING: FTL vs CARTING STRUCTURAL RISK',
          color=ACCENT_GOLD, fontsize=10, fontweight='bold', va='top',
          transform=ax_b.transAxes)

route_txt = (
    "The classifier achieving AUC=1.00 is not a modelling artifact — it is itself the finding. "
    "Route type (FTL vs Carting) is assigned structurally per corridor, not per trip. "
    "The model's perfect discrimination confirms that Delhivery's current network treats every trip "
    "on a corridor identically. This means the business question is not 'should this trip be FTL?' "
    "but 'should this corridor switch from Carting to FTL?' — a far higher-value intervention. "
    "287 Carting corridors with >90% SLA breach rate are immediate candidates for this switch."
)
wrapped_b = textwrap.fill(route_txt, width=88)
ax_b.text(0.015, 0.72, wrapped_b, color=TEXT_LIGHT, fontsize=7.8, va='top',
          transform=ax_b.transAxes, linespacing=1.5)
ax_b.axis('off')

# ─── SECTION C: TOP 5 BOTTLENECK HUBS TABLE ─────────────────────────
ax_c = fig.add_axes([0.03, 0.618, 0.94, 0.118])
ax_c.set_facecolor(BRAND_DARK)
ax_c.text(0.0, 0.97, '▌ TOP 5 BOTTLENECK HUBS — DETAILED DIAGNOSIS & RECOMMENDED INTERVENTIONS',
          color=ACCENT_RED, fontsize=11, fontweight='bold', va='top',
          transform=ax_c.transAxes)

interventions = [
    {
        'rank': '#1',
        'hub': 'Gurgaon Bilaspur HB\n(Haryana)',
        'btw': '31.7%',
        'sla': '81.7%',
        'contrib': '15.8%',
        'risk': '₹38.0L',
        'save': '₹13.3L',
        'action': 'Parallel corridor via Manesar; add 2 outbound docks;\nredeploy night-shift capacity',
        'priority': 'CRITICAL',
        'color': ACCENT_RED
    },
    {
        'rank': '#2',
        'hub': 'Bangalore Nelmngla H\n(Karnataka)',
        'btw': '18.5%',
        'sla': '81.0%',
        'contrib': '6.7%',
        'risk': '₹16.0L',
        'save': '₹5.6L',
        'action': 'Direct FTL line to Bengaluru South; reduce transfers\nthrough Whitefield hub; expand dock capacity',
        'priority': 'HIGH',
        'color': ACCENT_RED
    },
    {
        'rank': '#3',
        'hub': 'Bhiwandi Mankoli HB\n(Maharashtra)',
        'btw': '9.1%',
        'sla': '90.6%',
        'contrib': '6.8%',
        'risk': '₹16.3L',
        'save': '₹5.7L',
        'action': 'Highest breach rate in top-5; route diversion via Thane;\ncutoff time shift to 18:00 to avoid morning peak',
        'priority': 'HIGH',
        'color': ACCENT_RED
    },
    {
        'rank': '#4',
        'hub': 'Hyderabad Shamshbd H\n(Telangana)',
        'btw': '13.7%',
        'sla': '89.3%',
        'contrib': '2.5%',
        'risk': '₹6.0L',
        'save': '₹2.1L',
        'action': 'Airport proximity congestion; renegotiate slot window;\nalternative Uppal route for last-mile',
        'priority': 'MEDIUM',
        'color': ACCENT_GOLD
    },
    {
        'rank': '#5',
        'hub': 'Kolkata Dankuni HB\n(West Bengal)',
        'btw': '15.1%',
        'sla': '86.0%',
        'contrib': '2.7%',
        'risk': '₹6.5L',
        'save': '₹2.3L',
        'action': 'Port-area congestion; stagger departure windows;\nconsider Howrah bypass corridor',
        'priority': 'MEDIUM',
        'color': ACCENT_GOLD
    },
]

headers = ['Rank','Hub','Betweenness','SLA Breach','% All Breaches','Rev. at Risk','Potential Saving','Priority','Recommended Intervention']
col_x   = [0.00, 0.04, 0.18, 0.25, 0.32, 0.40, 0.48, 0.56, 0.63]
col_w   = [0.04, 0.14, 0.07, 0.07, 0.08, 0.08, 0.08, 0.07, 0.37]

# Header row
for j, (h, x) in enumerate(zip(headers, col_x)):
    ax_c.text(x, 0.86, h, color=TEXT_MID, fontsize=7, fontweight='bold',
              va='top', transform=ax_c.transAxes)

# Divider


# Data rows
for i, hub in enumerate(interventions):
    y = 0.78 - i * 0.145
    bg_color = '#1B2A3B' if i % 2 == 0 else '#162231'
    ax_c.add_patch(mpatches.FancyBboxPatch((0, y-0.04), 1.0, 0.14,
                   boxstyle='round,pad=0.002', facecolor=bg_color,
                   edgecolor='none', transform=ax_c.transAxes))
    vals = [hub['rank'], hub['hub'], hub['btw'], hub['sla'],
            hub['contrib'], hub['risk'], hub['save'], hub['priority'], hub['action']]
    for j, (val, x) in enumerate(zip(vals, col_x)):
        color = hub['color'] if j in [2,3,4,5,6] else \
                (hub['color'] if j == 7 else TEXT_LIGHT)
        ax_c.text(x+0.005, y+0.06, val, color=color, fontsize=7.2,
                  va='top', transform=ax_c.transAxes, linespacing=1.3)
ax_c.axis('off')

# ─── SECTION D: CHARTS (4 mini-charts in one row) ───────────────────
# Hub breach contribution bar chart
ax_d1 = fig.add_axes([0.03, 0.455, 0.20, 0.150])
ax_d1.set_facecolor(BRAND_MID)
hubs_short = [h['hub'].split('\n')[0] for h in interventions]
contribs   = [float(h['contrib'].replace('%','')) for h in interventions]
bar_colors = [h['color'] for h in interventions]
ax_d1.barh(range(5), contribs[::-1], color=bar_colors[::-1], alpha=0.9)
ax_d1.set_yticks(range(5))
ax_d1.set_yticklabels(hubs_short[::-1], fontsize=7, color=TEXT_LIGHT)
ax_d1.set_xlabel('% of All Breaches', color=TEXT_MID, fontsize=7)
ax_d1.set_title('Hub Breach Contribution', color=TEXT_LIGHT, fontsize=8, fontweight='bold')
ax_d1.tick_params(colors=TEXT_FAINT, labelsize=7)
for sp in ax_d1.spines.values(): sp.set_color('#30363d')

# Betweenness vs SLA breach scatter
ax_d2 = fig.add_axes([0.26, 0.455, 0.20, 0.150])
ax_d2.set_facecolor(BRAND_MID)
btw_vals   = [float(h['btw'].replace('%',''))   for h in interventions]
sla_vals   = [float(h['sla'].replace('%',''))   for h in interventions]
ax_d2.scatter(btw_vals, sla_vals,
              s=[200,150,120,100,90], c=[ACCENT_RED,ACCENT_RED,ACCENT_RED,ACCENT_GOLD,ACCENT_GOLD],
              alpha=0.85, edgecolors='white', linewidths=0.8, zorder=3)
for i, h in enumerate(interventions):
    ax_d2.annotate(h['rank'], (btw_vals[i], sla_vals[i]),
                   fontsize=7, color=TEXT_LIGHT,
                   xytext=(4,3), textcoords='offset points')
ax_d2.set_xlabel('Betweenness Centrality (%)', color=TEXT_MID, fontsize=7)
ax_d2.set_ylabel('SLA Breach Rate (%)',        color=TEXT_MID, fontsize=7)
ax_d2.set_title('Betweenness vs SLA Breach', color=TEXT_LIGHT, fontsize=8, fontweight='bold')
ax_d2.tick_params(colors=TEXT_FAINT, labelsize=7)
for sp in ax_d2.spines.values(): sp.set_color('#30363d')

# Revenue at risk bar
ax_d3 = fig.add_axes([0.49, 0.455, 0.20, 0.150])
ax_d3.set_facecolor(BRAND_MID)
rev_vals  = [float(h['risk'].replace('₹','').replace('L','')) for h in interventions]
save_vals = [float(h['save'].replace('₹','').replace('L','')) for h in interventions]
x3 = np.arange(5)
ax_d3.bar(x3-0.2, rev_vals,  0.38, label='Revenue at Risk',  color=ACCENT_RED,  alpha=0.85)
ax_d3.bar(x3+0.2, save_vals, 0.38, label='Potential Saving', color=ACCENT_TEAL, alpha=0.85)
ax_d3.set_xticks(x3)
ax_d3.set_xticklabels([f'#{i+1}' for i in range(5)], color=TEXT_LIGHT, fontsize=8)
ax_d3.set_ylabel('₹ Lakhs', color=TEXT_MID, fontsize=7)
ax_d3.set_title('Revenue at Risk vs Saving', color=TEXT_LIGHT, fontsize=8, fontweight='bold')
ax_d3.legend(facecolor=BRAND_MID, labelcolor=TEXT_MID, fontsize=6)
ax_d3.tick_params(colors=TEXT_FAINT, labelsize=7)
for sp in ax_d3.spines.values(): sp.set_color('#30363d')

# SLA breach by time of day
ax_d4 = fig.add_axes([0.72, 0.455, 0.21, 0.150])
ax_d4.set_facecolor(BRAND_MID)
hourly = df.groupby(['hour_of_day','route_type'])['sla_breach'].mean().reset_index()
for rt, color, ls in [('FTL', ACCENT_RED, '-'), ('Carting', ACCENT_TEAL, '--')]:
    grp = hourly[hourly['route_type']==rt]
    ax_d4.plot(grp['hour_of_day'], grp['sla_breach']*100,
               color=color, lw=1.8, linestyle=ls, label=rt, marker='o', markersize=2)
ax_d4.axvspan(18, 21, alpha=0.1, color=ACCENT_TEAL, label='Best window')
ax_d4.axvspan(8,  12, alpha=0.1, color=ACCENT_RED,  label='Risk window')
ax_d4.set_xlabel('Hour of Day', color=TEXT_MID, fontsize=7)
ax_d4.set_ylabel('SLA Breach Rate (%)', color=TEXT_MID, fontsize=7)
ax_d4.set_title('Breach Rate by Hour', color=TEXT_LIGHT, fontsize=8, fontweight='bold')
ax_d4.legend(facecolor=BRAND_MID, labelcolor=TEXT_MID, fontsize=6)
ax_d4.tick_params(colors=TEXT_FAINT, labelsize=7)
for sp in ax_d4.spines.values(): sp.set_color('#30363d')

# ─── SECTION E: ETA MODEL RESULTS ───────────────────────────────────
ax_e = fig.add_axes([0.03, 0.355, 0.94, 0.090])
ax_e.set_facecolor(BRAND_DARK)
ax_e.text(0.0, 0.97, '▌ GRAPH-ENHANCED ETA MODEL — PERFORMANCE SUMMARY',
          color=ACCENT_TEAL, fontsize=11, fontweight='bold', va='top',
          transform=ax_e.transAxes)

model_rows = [
    ('Baseline XGBoost (OSRM features only)',
     '13.00 min', '45.47 min', '34.4%', '32.5%', '59.8%', '—', TEXT_MID),
    ('Graph-Enhanced XGBoost (+ corridor history + centrality)',
     '11.07 min', '37.10 min', '29.7%', '39.1%', '67.5%', '↑14.9%', ACCENT_TEAL),
]
mhdr = ['Model', 'MAE', 'RMSE', 'MAPE', 'Within-15%', 'Within-30%', 'MAE Improvement']
mx   = [0.0, 0.38, 0.46, 0.53, 0.60, 0.68, 0.76]

for j, (h, x) in enumerate(zip(mhdr, mx)):
    ax_e.text(x, 0.84, h, color=TEXT_MID, fontsize=7.5, fontweight='bold',
              va='top', transform=ax_e.transAxes)

ax_e.plot([0,1],[0.79,0.79], color=TEXT_FAINT, lw=0.5, transform=ax_e.transAxes)

for i, (row) in enumerate(model_rows):
    vals, clr = row[:-1], row[-1]
    y = 0.63 - i * 0.30
    for j, (v, x) in enumerate(zip(vals, mx)):
        ax_e.text(x, y, v, color=clr if j > 0 else TEXT_LIGHT,
                  fontsize=8 if j > 0 else 7.5,
                  fontweight='bold' if j == 6 and i == 1 else 'normal',
                  va='top', transform=ax_e.transAxes)

ax_e.text(0.0, 0.18,
          '★  Key insight: The single most important feature is dist_x_hist (osrm_distance × corridor_historical_delay_ratio), confirming that '
          'graph-structural position predicts ETA\n'
          '    far better than OSRM alone. The 6.5 percentage-point improvement in Within-15% accuracy directly translates to smarter SLA commitment at booking.',
          color=TEXT_MID, fontsize=7.5, va='top', transform=ax_e.transAxes, linespacing=1.5)
ax_e.axis('off')

# ─── SECTION F: RECOMMENDATIONS & NEXT STEPS ────────────────────────
ax_f = fig.add_axes([0.03, 0.215, 0.44, 0.130])
ax_f.set_facecolor(BRAND_MID)
ax_f.text(0.015, 0.97, '▌ PRIORITY RECOMMENDATIONS', color=ACCENT_RED,
          fontsize=10, fontweight='bold', va='top', transform=ax_f.transAxes)

recs = [
    ('1.', 'IMMEDIATE (0–30 days)',
     'Deploy graph-enhanced ETA model in production; replaces OSRM estimates. '
     'Reduces over-promising by ~14.9%. Activate real-time delay scoring for '
     'top-5 bottleneck hubs.'),
    ('2.', 'SHORT-TERM (1–3 months)',
     'Initiate capacity audit at Gurgaon Bilaspur HB (India\'s highest-betweenness '
     'hub). Add 2 parallel outbound docks. Target: reduce breach rate from 81.7% '
     f'→ <65%, recovering ₹{rev["top3_hub_savings_lakhs"]:.0f}L annually.'),
    ('3.', 'MEDIUM-TERM (3–6 months)',
     'Convert 43 high-breach Carting corridors (>90% SLA breach) to FTL. '
     'Prioritise Morning dispatches; shift to Evening windows where possible. '
     'Bhiwandi and Hyderabad hubs need parallel route development.'),
]
for rank, timeline, text in recs:
    ax_f.text(0.015, {'1.':0.80,'2.':0.52,'3.':0.24}[rank],
              f"{rank} {timeline}", color=ACCENT_GOLD, fontsize=7.5,
              fontweight='bold', va='top', transform=ax_f.transAxes)
    wrapped = textwrap.fill(text, width=68)
    ax_f.text(0.04, {'1.':0.70,'2.':0.42,'3.':0.14}[rank],
              wrapped, color=TEXT_LIGHT, fontsize=7.2,
              va='top', transform=ax_f.transAxes, linespacing=1.4)
ax_f.axis('off')

# ─── SECTION G: EXPECTED IMPACT SUMMARY ─────────────────────────────
ax_g = fig.add_axes([0.49, 0.215, 0.44, 0.130])
ax_g.set_facecolor(BRAND_MID)
ax_g.text(0.015, 0.97, '▌ EXPECTED BUSINESS IMPACT', color=ACCENT_TEAL,
          fontsize=10, fontweight='bold', va='top', transform=ax_g.transAxes)

impacts = [
    (ACCENT_TEAL, '−14.9%', 'ETA prediction error reduction (MAE)', 'Fewer missed SLA commitments at booking'),
    (ACCENT_TEAL, '+6.5pp', 'Within-15% prediction accuracy', 'Better dispatch planning & customer communication'),
    (ACCENT_RED,  f'₹{rev["top3_hub_savings_lakhs"]:.0f}L',
     'SLA penalty saving (top-3 hub upgrade)', 'Direct P&L impact from reduced breach exposure'),
    (ACCENT_GOLD, '29.3%', 'Breach volume addressable (top-3 hubs)', 'Concentrated intervention with maximum ROI'),
    (ACCENT_GOLD, '43 routes', 'Carting→FTL conversion candidates', 'Structural SLA improvement for chronic corridors'),
]

for i, (color, val, label, sub) in enumerate(impacts):
    y = 0.82 - i * 0.17
    ax_g.add_patch(mpatches.FancyBboxPatch((0.01, y-0.08), 0.11, 0.14,
                   boxstyle='round,pad=0.01',
                   facecolor=BRAND_LIGHT, edgecolor=color, lw=1,
                   transform=ax_g.transAxes))
    ax_g.text(0.065, y+0.02, val,   color=color,      fontsize=9.5,
              fontweight='bold', ha='center', va='center', transform=ax_g.transAxes)
    ax_g.text(0.14, y+0.03,  label, color=TEXT_LIGHT,  fontsize=7.5,
              va='center', transform=ax_g.transAxes)
    ax_g.text(0.14, y-0.04,  sub,   color=TEXT_FAINT,  fontsize=7,
              va='center', transform=ax_g.transAxes)
ax_g.axis('off')

# ─── SECTION H: CORRIDOR AUDIT SNAPSHOT ─────────────────────────────
ax_h = fig.add_axes([0.03, 0.110, 0.94, 0.098])
ax_h.set_facecolor(BRAND_DARK)
ax_h.text(0.0, 0.96, '▌ CHRONIC DELAY CORRIDOR AUDIT — TOP 10 CORRIDORS REQUIRING INTERVENTION',
          color=ACCENT_GOLD, fontsize=10, fontweight='bold', va='top',
          transform=ax_h.transAxes)

ch_top = chronic.head(10).copy()
corr_hdr = ['Source Center', 'Destination Center', 'Route Type', 'Trips',
            'Avg Delay Factor', 'SLA Breach Rate', 'Intervention']
corr_x   = [0.00, 0.19, 0.36, 0.42, 0.50, 0.58, 0.66]

for j, (h, x) in enumerate(zip(corr_hdr, corr_x)):
    ax_h.text(x, 0.80, h, color=TEXT_MID, fontsize=7, fontweight='bold',
              va='top', transform=ax_h.transAxes)
ax_h.plot([0,1],[0.74,0.74], color=TEXT_FAINT, lw=0.5, transform=ax_h.transAxes)

intervention_map = {
    0: 'Parallel route; FTL upgrade',
    1: 'Direct consolidation hub',
    2: 'Reroute via adjacent facility',
    3: 'FTL frequency increase',
    4: 'Cutoff time optimisation',
    5: 'Load balancing + time shift',
    6: 'Route audit + alternative path',
    7: 'Dock expansion + time window',
    8: 'Direct FTL line creation',
    9: 'Facility upgrade + FTL upgrade',
}


# --- INSERT THIS BEFORE THE CORRIDOR AUDIT TABLE LOOP IN TASK 5 ---

# Build mapping dictionary from node_metrics (since we know it has facility names)
name_map = dict(zip(node_metrics['center'], node_metrics['facility_name']))

# Map the source and destination centers, fallback to raw ID if name is missing
ch_top['src_name'] = ch_top['source_center'].map(name_map).fillna(ch_top['source_center'])
ch_top['dst_name'] = ch_top['destination_center'].map(name_map).fillna(ch_top['destination_center'])

# Now, update your table loop to use these new columns:
for i, (_, row) in enumerate(ch_top.iterrows()):
    if i >= 8: break
    y = 0.70 - i * 0.087
    breach_color = ACCENT_RED if row['sla_breach_pct'] > 0.95 else ACCENT_GOLD
    
    # USE src_name AND dst_name HERE INSTEAD OF source_center / destination_center
    vals = [str(row['src_name'])[:22],   
            str(row['dst_name'])[:22],   
            str(row['route_type']),
            str(int(row['trip_count'])),
            f"{row['mean_factor']:.2f}×",
            f"{row['sla_breach_pct']*100:.0f}%",
            intervention_map.get(i, 'Review & reroute')]
            
    for j, (v, x) in enumerate(zip(vals, corr_x)):
        color = breach_color if j in [4,5] else TEXT_LIGHT
        ax_h.text(x, y, v, color=color, fontsize=7.2, va='top', transform=ax_h.transAxes)

for i, (_, row) in enumerate(ch_top.iterrows()):
    if i >= 8: break
    y = 0.70 - i * 0.087
    breach_color = ACCENT_RED if row['sla_breach_pct'] > 0.95 else ACCENT_GOLD
    vals = [str(row['source_center'])[:22],
            str(row['destination_center'])[:22],
            str(row['route_type']),
            str(int(row['trip_count'])),
            f"{row['mean_factor']:.2f}×",
            f"{row['sla_breach_pct']*100:.0f}%",
            intervention_map.get(i, 'Review & reroute')]
    for j, (v, x) in enumerate(zip(vals, corr_x)):
        color = breach_color if j in [4,5] else TEXT_LIGHT
        ax_h.text(x, y, v, color=color, fontsize=7.2, va='top',
                  transform=ax_h.transAxes)
ax_h.axis('off')

# ─── FOOTER ─────────────────────────────────────────────────────────
ax_ftr = fig.add_axes([0.0, 0.0, 1.0, 0.035])
ax_ftr.set_facecolor(BRAND_MID)
ax_ftr.text(0.02, 0.55,
            'Analysis based on 142,267 trip segments | Graph: 1,657 nodes, 2,783 corridors | '
            'Model: XGBoost + Node2Vec corridor embeddings | Revenue estimates at ₹200/breach',
            color=TEXT_FAINT, fontsize=7, va='center', transform=ax_ftr.transAxes)
ax_ftr.text(0.98, 0.55, 'Delhivery Network Intelligence · Data Science Team · Confidential',
            color=TEXT_FAINT, fontsize=7, va='center', ha='right',
            transform=ax_ftr.transAxes)
ax_ftr.axis('off')

plt.savefig('fig_strategy_memo.png', dpi=160, bbox_inches='tight',
            facecolor=BRAND_DARK)
plt.close()
print("> Saved: fig_strategy_memo.png")
print("\n> Task 5 - Strategy Memo complete.")