import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 13

# --- New Baselines ---
baseline_t2w = 54.34
baseline_i2w = 55.04

# --- Absolute Latencies (seconds) ---
# DiCache and FasterCache kept the same; WorldCache updated
dicache_t2w = 40.82
dicache_i2w = 39.68

fastercache_t2w = 34.51
fastercache_i2w = 32.75

worldcache_t2w = 26.28
worldcache_i2w = 24.48

# --- Compute percentage reductions ---
def pct_reduction(baseline, val):
    return (baseline - val) / baseline * 100

methods = ['DiCache', 'FasterCache', 'WorldCache']
t2w_latencies = [dicache_t2w, fastercache_t2w, worldcache_t2w]
i2w_latencies = [dicache_i2w, fastercache_i2w, worldcache_i2w]

t2w_pcts = [pct_reduction(baseline_t2w, l) for l in t2w_latencies]
i2w_pcts = [pct_reduction(baseline_i2w, l) for l in i2w_latencies]

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(methods))
width = 0.3

# Colors matching the original figure
color_t2w = '#6B8E6B'  # muted green
color_i2w = '#7B9EC7'  # muted blue

bars1 = ax.bar(x - width/2, t2w_pcts, width, label='Text2World (T2W)',
               color=color_t2w, edgecolor='black', linewidth=0.8,
               hatch='xx', alpha=0.85)
bars2 = ax.bar(x + width/2, i2w_pcts, width, label='Image2World (I2W)',
               color=color_i2w, edgecolor='black', linewidth=0.8,
               hatch='//', alpha=0.85)

# --- Annotations ---
for bar, pct, lat in zip(bars1, t2w_pcts, t2w_latencies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
            f'+{pct:.1f}%\n({lat:.2f}s)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar, pct, lat in zip(bars2, i2w_pcts, i2w_latencies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
            f'+{pct:.1f}%\n({lat:.2f}s)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Axes ---
ax.set_ylabel('Latency reduction vs Baseline (%)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=13, fontweight='bold')
ax.set_ylim(0, 75)
ax.set_yticks(range(0, 80, 10))
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

# Subtitle
ax.text(0.5, -0.1, f'Baselines: T2W={baseline_t2w}s, I2W={baseline_i2w}s.',
        ha='center', va='top', transform=ax.transAxes,
        fontsize=12, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/share_2/users/umair_nawaz/World-Models/WorldCache/Models/Cosmos-Predict2.5/assets/figures/latency_reduction.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/share_2/users/umair_nawaz/World-Models/WorldCache/Models/Cosmos-Predict2.5/assets/figures/latency_reduction.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure saved!")
print(f"\nPercentage reductions:")
for m, t, i in zip(methods, t2w_pcts, i2w_pcts):
    print(f"  {m}: T2W={t:.1f}%, I2W={i:.1f}%")
