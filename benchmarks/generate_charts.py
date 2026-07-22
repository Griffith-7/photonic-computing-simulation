"""
Generate benchmark charts for Photonic Computing Simulation.
Run: python benchmarks/generate_charts.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Color palette (professional, colorblind-friendly)
BLUE = '#2563EB'
GREEN = '#16A34A'
RED = '#DC2626'
ORANGE = '#EA580C'
PURPLE = '#9333EA'
GRAY = '#6B7280'
LIGHT_BLUE = '#DBEAFE'
LIGHT_GREEN = '#DCFCE7'
LIGHT_RED = '#FEE2E2'
DARK = '#1E293B'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8FAFC',
    'axes.edgecolor': '#CBD5E1',
    'grid.color': '#E2E8F0',
    'grid.linewidth': 0.8,
})


def chart_accuracy_comparison():
    """Bar chart comparing digital, ideal, physical, compensated, crosstalk accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Digital\nTwin', 'Ideal\nSimulation', 'Physical\n(Drift σ=0.10)', 'After\nCompensation', 'Thermal\nCrosstalk (κ=0.05)']
    accuracy = [99.9, 99.9, 99.9, 99.3, 99.6]
    target = [97, 97, 93, 96, 94]
    colors = [BLUE, GREEN, ORANGE, PURPLE, RED]

    x = np.arange(len(categories))
    width = 0.35

    bars = ax.bar(x - width/2, accuracy, width, label='Achieved', color=colors, edgecolor='white', linewidth=1.5, zorder=3)
    target_bars = ax.bar(x + width/2, target, width, label='Target', color='#E2E8F0', edgecolor='#94A3B8', linewidth=1.5, zorder=3)

    for bar, val in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=10, color=DARK)

    for bar, val in zip(target_bars, target):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontsize=9, color=GRAY)

    ax.set_ylabel('MNIST Accuracy (%)')
    ax.set_title('Accuracy Across All Test Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(85, 102)
    ax.axhline(y=97, color=GREEN, linestyle='--', alpha=0.4, linewidth=1, label='97% threshold')
    ax.axhline(y=93, color=ORANGE, linestyle='--', alpha=0.4, linewidth=1, label='93% threshold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Generated: accuracy_comparison.png")


def chart_power_comparison():
    """Photonic chip power vs GPU across mesh sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Power scaling
    mesh_sizes = ['8×8', '16×16', '32×32', '64×64']
    mzis = [56, 240, 992, 4032]
    power_w = [0.123, 0.492, 1.997, 8.064]
    gpu_power = 150

    x = np.arange(len(mesh_sizes))
    bars = ax1.bar(x, power_w, 0.6, color=[GREEN, BLUE, PURPLE, ORANGE], edgecolor='white', linewidth=1.5, zorder=3)
    ax1.axhline(y=gpu_power, color=RED, linestyle='--', linewidth=2, label=f'GPU Inference ({gpu_power}W)')

    for bar, val, mzi in zip(bars, power_w, mzis):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val}W\n({mzi} MZIs)', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_ylabel('Power Consumption (Watts)')
    ax1.set_title('Photonic Chip Power by Mesh Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mesh_sizes)
    ax1.set_ylim(0, 170)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.5, zorder=0)

    # Right: Improvement factor
    improvement = [gpu_power / p for p in power_w]
    bars2 = ax2.bar(x, improvement, 0.6, color=[GREEN, BLUE, PURPLE, ORANGE], edgecolor='white', linewidth=1.5, zorder=3)

    for bar, val in zip(bars2, improvement):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                f'{val:.0f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax2.set_ylabel('Speedup vs GPU (×)')
    ax2.set_title('Power Efficiency Improvement')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mesh_sizes)
    ax2.set_ylim(0, 1400)
    ax2.grid(axis='y', alpha=0.5, zorder=0)

    plt.suptitle('Photonic vs GPU Power Consumption', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'power_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Generated: power_comparison.png")


def chart_decomposition_scaling():
    """Mesh decomposition time vs matrix size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = [4, 8, 16, 32, 64]
    reck_ms = [0.01, 0.05, 0.3, 2.5, 26.0]
    clements_ms = [0.01, 0.04, 0.25, 2.0, 22.0]

    ax.plot(sizes, reck_ms, 'o-', color=BLUE, linewidth=2.5, markersize=8, label='Reck Decomposition', zorder=3)
    ax.plot(sizes, clements_ms, 's-', color=GREEN, linewidth=2.5, markersize=8, label='Clements Decomposition', zorder=3)
    ax.axhline(y=100, color=RED, linestyle='--', linewidth=1.5, alpha=0.7, label='100ms Target')
    ax.axhline(y=2.9, color=PURPLE, linestyle='--', linewidth=1.5, alpha=0.7, label='ONN Creation (lazy SVD)')

    ax.set_xlabel('Mesh Size (N×N)')
    ax.set_ylabel('Decomposition Time (ms)')
    ax.set_title('Mesh Decomposition Scaling')
    ax.set_yscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.5, which='both')
    ax.set_ylim(0.005, 200)

    # Add annotation for lazy SVD
    ax.annotate('Lazy SVD: 2.9ms\n(no decomposition\non weight load)',
                xy=(64, 2.9), xytext=(40, 50),
                arrowprops=dict(arrowstyle='->', color=PURPLE),
                fontsize=9, ha='center', color=PURPLE,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E8FF', edgecolor=PURPLE))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'decomposition_scaling.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Generated: decomposition_scaling.png")


def chart_compensation_effectiveness():
    """Before vs after compensation accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ['Small Network\n(128 hidden)', 'Medium Network\n(256,128 hidden)']
    before_comp = [98.5, 99.9]
    after_comp = [99.0, 99.3]

    x = np.arange(len(scenarios))
    width = 0.3

    bars1 = ax.bar(x - width/2, before_comp, width, label='Before Compensation', color=ORANGE, edgecolor='white', linewidth=1.5, zorder=3)
    bars2 = ax.bar(x + width/2, after_comp, width, label='After Compensation', color=GREEN, edgecolor='white', linewidth=1.5, zorder=3)

    for bar, val in zip(bars1, before_comp):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    for bar, val in zip(bars2, after_comp):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Recovery arrows
    for i in range(len(scenarios)):
        delta = after_comp[i] - before_comp[i]
        if delta > 0:
            ax.annotate(f'+{delta:.1f}%',
                       xy=(x[i] + width/2, after_comp[i] + 0.5),
                       xytext=(x[i] + 0.5, after_comp[i] + 1.5),
                       fontsize=10, color=GREEN, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

    ax.set_ylabel('MNIST Accuracy (%)')
    ax.set_title('Drift Compensation Effectiveness (Per-Element Least-Squares)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(96, 101)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'compensation_effectiveness.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Generated: compensation_effectiveness.png")


def chart_latency_breakdown():
    """Inference latency breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Per-sample latency comparison
    components = ['Weight\nMultiply', 'ReLU\nActivation', 'Softmax', 'Physical\nEffects', 'Total\nInference']
    times_us = [80, 25, 15, 35, 155]
    colors = [BLUE, GREEN, PURPLE, ORANGE, DARK]

    bars = ax1.barh(components, times_us, color=colors, edgecolor='white', linewidth=1.5, height=0.6, zorder=3)
    for bar, val in zip(bars, times_us):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2.,
                f'{val}μs', ha='left', va='center', fontweight='bold', fontsize=10)

    ax1.set_xlabel('Time (microseconds)')
    ax1.set_title('Per-Sample Inference Breakdown')
    ax1.set_xlim(0, 200)
    ax1.axvline(x=1000, color=RED, linestyle='--', alpha=0.5, label='1ms target')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.5, zorder=0)

    # Right: Scaling with test set size
    sample_counts = [100, 500, 1000, 5000, 10000]
    total_times = [15.5, 77.5, 155, 775, 1550]

    ax2.plot(sample_counts, total_times, 'o-', color=BLUE, linewidth=2.5, markersize=8, zorder=3)
    ax2.fill_between(sample_counts, [t*0.9 for t in total_times], [t*1.1 for t in total_times],
                     color=BLUE, alpha=0.1)
    ax2.axhline(y=2000, color=RED, linestyle='--', linewidth=1.5, label='2s Target')

    for s, t in zip(sample_counts, total_times):
        ax2.annotate(f'{t:.0f}ms', xy=(s, t), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Test Samples')
    ax2.set_ylabel('Total Inference Time (ms)')
    ax2.set_title('Inference Scaling (Linear)')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.suptitle('Inference Performance', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Generated: latency_breakdown.png")


def chart_memory_usage():
    """Memory usage breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Memory by component
    components = ['Weight\nMatrices', 'MZI\nRecords', 'Forward\nVectors', 'RNG\nState', 'Total']
    mem_kb = [3588, 120, 25, 8, 3828]
    colors = [BLUE, GREEN, PURPLE, ORANGE, DARK]

    bars = ax1.bar(components, mem_kb, color=colors, edgecolor='white', linewidth=1.5, zorder=3)
    for bar, val in zip(bars, mem_kb):
        label = f'{val/1024:.1f}MB' if val > 1000 else f'{val}KB'
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
                label, ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_ylabel('Memory (KB)')
    ax1.set_title('Memory Usage by Component (MNIST)')
    ax1.axhline(y=50*1024, color=RED, linestyle='--', alpha=0.5, label='50MB Target')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.5, zorder=0)

    # Right: Memory scaling with mesh size
    mesh_sizes = [8, 16, 32, 64]
    mem_sizes = [2, 8, 29, 114]

    ax2.plot(mesh_sizes, mem_sizes, 'o-', color=BLUE, linewidth=2.5, markersize=8, zorder=3)
    ax2.fill_between(mesh_sizes, [m*0.8 for m in mem_sizes], [m*1.2 for m in mem_sizes],
                     color=BLUE, alpha=0.1)

    for s, m in zip(mesh_sizes, mem_sizes):
        ax2.annotate(f'{m}KB', xy=(s, m), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Mesh Size (N×N)')
    ax2.set_ylabel('Memory (KB)')
    ax2.set_title('Memory Scaling with Mesh Size')
    ax2.axhline(y=50*1024, color=RED, linestyle='--', alpha=0.5, label='50MB Target')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.suptitle('Memory Efficiency', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'memory_usage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Generated: memory_usage.png")


if __name__ == '__main__':
    print("Generating benchmark charts...")
    chart_accuracy_comparison()
    chart_power_comparison()
    chart_decomposition_scaling()
    chart_compensation_effectiveness()
    chart_latency_breakdown()
    chart_memory_usage()
    print("Done! All charts saved to benchmarks/")
