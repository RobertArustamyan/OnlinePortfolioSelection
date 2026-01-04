import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def extract_training_days_from_folder(folder_name):
    """
    Extract training days from folder name.
    Expected format: train{N}_test{M}_stocks{K}_{timestamp}
    """
    match = re.search(r'train(\d+)_test', folder_name)
    if match:
        return int(match.group(1))
    return None


def get_best_test_wealth_from_json(json_path):
    """
    Extract the best (maximum) test final wealth from detailed JSON results.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    max_wealth = -float('inf')
    best_params = None

    # Iterate through all cost models in experiments
    for model_name, experiments in data['experiments'].items():
        for experiment in experiments:
            test_wealth = experiment['testing']['final_wealth']
            if test_wealth > max_wealth:
                max_wealth = test_wealth
                best_params = experiment['parameters']

    return max_wealth, best_params


def analyze_training_effect(base_dir):
    """
    Analyze the effect of training days on test performance.
    Returns:DataFrame with training_days, test_final_wealth, and best_params
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        raise ValueError(f"Directory does not exist: {base_dir}")

    results = []

    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue

        training_days = extract_training_days_from_folder(folder.name)
        if training_days is None:
            print(f"Skipping folder (couldn't parse): {folder.name}")
            continue

        # Look for the detailed JSON results file
        json_path = folder / 'ons_costs_detailed_results.json'
        if not json_path.exists():
            print(f"Skipping folder (no JSON found): {folder.name}")
            continue

        try:
            # Extract best test wealth
            best_wealth, best_params = get_best_test_wealth_from_json(json_path)

            results.append({
                'training_days': training_days,
                'test_final_wealth': best_wealth,
                'cost_penalty': best_params['cost_penalty'],
                'alpha': best_params['alpha'],
                'improvement_threshold': best_params['improvement_threshold'],
                'folder_name': folder.name
            })

            print(f" Processed: {folder.name} | Train days: {training_days} | Best wealth: {best_wealth:.4f}")

        except Exception as e:
            print(f"Error processing {folder.name}: {e}")
            continue

    # Create DataFrame and sort by training days
    df = pd.DataFrame(results)
    df = df.sort_values('training_days')

    return df


def plot_training_effect(df, save_path=None):
    """
    Create a plot showing the relationship between training days and test performance.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Training Days vs Test Final Wealth
    ax1.plot(df['training_days'], df['test_final_wealth'],
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Training Days', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Final Wealth', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of Training Duration on Test Performance',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add value labels on points
    for i, row in df.iterrows():
        ax1.annotate(f'{row["test_final_wealth"]:.3f}',
                     (row['training_days'], row['test_final_wealth']),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8, alpha=0.7)

    # Highlight best result
    best_idx = df['test_final_wealth'].idxmax()
    best_row = df.loc[best_idx]
    ax1.scatter(best_row['training_days'], best_row['test_final_wealth'],
                color='red', s=100, marker='+', zorder=5,
                label=f'Best: {best_row["training_days"]} days')
    ax1.legend()

    # Plot 2: Best Hyperparameters over Training Days
    ax2_cost = ax2.twinx()
    ax2_alpha = ax2.twinx()

    # Offset the right spine of ax2_alpha
    ax2_alpha.spines['right'].set_position(('outward', 60))

    p1 = ax2.plot(df['training_days'], df['cost_penalty'],
                  marker='s', label='Cost Penalty', color='red', linewidth=2)
    p2 = ax2_cost.plot(df['training_days'], df['alpha'],
                       marker='^', label='Alpha', color='green', linewidth=2)
    p3 = ax2_alpha.plot(df['training_days'], df['improvement_threshold'],
                        marker='d', label='Improvement Threshold', color='blue', linewidth=2)

    ax2.set_xlabel('Training Days', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cost Penalty', fontsize=11, fontweight='bold', color='red')
    ax2_cost.set_ylabel('Alpha', fontsize=11, fontweight='bold', color='green')
    ax2_alpha.set_ylabel('Improvement Threshold', fontsize=11, fontweight='bold', color='blue')

    ax2.tick_params(axis='y', labelcolor='red')
    ax2_cost.tick_params(axis='y', labelcolor='green')
    ax2_alpha.tick_params(axis='y', labelcolor='blue')

    ax2.set_title('Best Hyperparameters',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Combine legends
    lines = p1 + p2 + p3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()

    return fig


if __name__ == '__main__':
    results_dir = r'C:\Users\rarustamyan\PycharmProjects\UniversalPortfolios\results\portfolio\WithCosts\ONS\TrainingTimeAnalysis'

    df = analyze_training_effect(results_dir)

    csv_path = Path(results_dir) / 'training_days_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Create and save plot
    plot_path = Path(results_dir) / 'training_days_effect.png'
    plot_training_effect(df, save_path=plot_path)