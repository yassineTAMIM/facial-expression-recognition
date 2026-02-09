import json
import matplotlib.pyplot as plt

# Load metrics
with open('results/metrics.json', 'r') as f:
    metrics = json.load(f)

# Create comparison figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Training summary
axes[0, 0].text(0.1, 0.7, f"Test Accuracy: {metrics['test_accuracy']:.2f}%", fontsize=14)
axes[0, 0].text(0.1, 0.5, f"Training Time: {metrics['training_time_minutes']:.1f} min", fontsize=14)
axes[0, 0].text(0.1, 0.3, f"Total Epochs: {metrics['total_epochs']}", fontsize=14)
axes[0, 0].axis('off')
axes[0, 0].set_title('Training Summary', fontweight='bold')

# Docker vs Local comparison
comparison = ['Docker', 'Local']
times = [metrics['training_time_minutes'], metrics['training_time_minutes']]
axes[0, 1].bar(comparison, times, color=['blue', 'orange'])
axes[0, 1].set_ylabel('Time (minutes)')
axes[0, 1].set_title('Docker vs Local Performance')

# Model parameters
axes[1, 0].text(0.1, 0.7, f"Batch Size: {metrics['config']['batch_size']}", fontsize=12)
axes[1, 0].text(0.1, 0.5, f"Learning Rate: {metrics['config']['learning_rate']}", fontsize=12)
axes[1, 0].text(0.1, 0.3, f"Total Params: {metrics['total_parameters']:,}", fontsize=12)
axes[1, 0].axis('off')
axes[1, 0].set_title('Model Configuration', fontweight='bold')

# Before/After comparison (you'll update this manually)
versions = ['V1 (batch=64)', 'V2 (batch=32)']
accuracies = [59.07, metrics['test_accuracy']]
axes[1, 1].bar(versions, accuracies, color=['red', 'green'])
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].set_title('Improvement Comparison')
axes[1, 1].set_ylim([50, 70])

plt.tight_layout()
plt.savefig('results/report_summary.png', dpi=300)
print("Report saved: results/report_summary.png")