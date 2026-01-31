#!/usr/bin/env python3
"""Visualize training logs from modded-nanogpt runs."""

import matplotlib.pyplot as plt
import re
import sys
from pathlib import Path

def parse_log(log_path):
    """Extract validation loss data from training log."""
    steps = []
    losses = []
    times_min = []

    with open(log_path) as f:
        for line in f:
            # Match lines like: step:250/1555 val_loss:4.5492 train_time:62832ms
            match = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+) train_time:(\d+)ms', line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                time_ms = int(match.group(3))
                steps.append(step)
                losses.append(loss)
                times_min.append(time_ms / 60000)  # Convert to minutes

    return steps, losses, times_min

def plot_training(log_path, output_path=None):
    """Create training visualization."""
    steps, losses, times_min = parse_log(log_path)

    if not steps:
        print("No validation data found in log!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss vs Steps
    ax1.plot(steps, losses, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=3.28, color='r', linestyle='--', label='Target (3.28)')
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Validation Loss vs Training Steps', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotations for first and last points
    ax1.annotate(f'{losses[0]:.2f}', (steps[0], losses[0]), textcoords="offset points",
                xytext=(10,10), fontsize=10)
    ax1.annotate(f'{losses[-1]:.4f}', (steps[-1], losses[-1]), textcoords="offset points",
                xytext=(-50,10), fontsize=10, fontweight='bold')

    # Plot 2: Loss vs Time
    ax2.plot(times_min, losses, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=3.28, color='r', linestyle='--', label='Target (3.28)')
    ax2.set_xlabel('Training Time (minutes)', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss vs Training Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add final time annotation
    ax2.annotate(f'{times_min[-1]:.1f} min\n{losses[-1]:.4f}',
                (times_min[-1], losses[-1]), textcoords="offset points",
                xytext=(-60,10), fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")
    else:
        plt.savefig('training_chart.png', dpi=150, bbox_inches='tight')
        print("Chart saved to: training_chart.png")

    # Print summary
    print(f"\n=== Training Summary ===")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss:   {losses[-1]:.4f}")
    print(f"Target loss:  3.28")
    print(f"Status:       {'SUCCESS' if losses[-1] <= 3.28 else 'NOT YET'}")
    print(f"Total time:   {times_min[-1]:.1f} minutes")
    print(f"Total steps:  {steps[-1]}")

if __name__ == "__main__":
    # Default to the most recent log
    log_dir = Path("/root/modded-nanogpt/logs")
    logs = sorted(log_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)

    if logs:
        log_path = logs[0]
        print(f"Using log: {log_path}")
        plot_training(log_path)
    else:
        print("No log files found in logs/ directory")
