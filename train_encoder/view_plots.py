"""
Simple script to view the training plots
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def view_plots():
    """Display all training plots"""
    plot_dir = "training_plots"
    
    if not os.path.exists(plot_dir):
        print(f"Plot directory '{plot_dir}' not found. Run training first!")
        return
    
    plot_files = [
        "training_loss.png",
        "loss_comparison.png", 
        "gradient_norms.png",
        "training_summary.png"
    ]
    
    titles = [
        "Training Loss Over Epochs",
        "Detailed Loss Comparison",
        "Gradient Norms Over Training",
        "Training Summary"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (plot_file, title) in enumerate(zip(plot_files, titles)):
        plot_path = os.path.join(plot_dir, plot_file)
        if os.path.exists(plot_path):
            img = mpimg.imread(plot_path)
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"Plot not found:\n{plot_file}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_plots() 