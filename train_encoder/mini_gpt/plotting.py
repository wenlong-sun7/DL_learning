"""
Plotting utilities for the Mini GPT Language Model
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any


def plot_training_loss(losses: List[float], save_path: str = None, show_plot: bool = True):
    """
    Plot training loss over epochs
    
    Args:
        losses: List of loss values for each epoch
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(losses) + 1)
    
    plt.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=6)
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    min_loss = min(losses)
    max_loss = max(losses)
    final_loss = losses[-1]
    
    plt.text(0.02, 0.98, f'Final Loss: {final_loss:.4f}\nMin Loss: {min_loss:.4f}\nMax Loss: {max_loss:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def plot_loss_comparison(epoch_losses: List[float], step_losses: List[float] = None, 
                        save_path: str = None, show_plot: bool = True):
    """
    Plot both epoch-level and step-level losses for detailed analysis
    
    Args:
        epoch_losses: List of average loss per epoch
        step_losses: List of loss per training step (optional)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot epoch losses
    epochs = range(1, len(epoch_losses) + 1)
    ax1.plot(epochs, epoch_losses, 'b-', linewidth=2, marker='o', markersize=6)
    ax1.set_title('Training Loss per Epoch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Average Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics to epoch plot
    min_epoch_loss = min(epoch_losses)
    final_epoch_loss = epoch_losses[-1]
    ax1.text(0.02, 0.98, f'Final Loss: {final_epoch_loss:.4f}\nMin Loss: {min_epoch_loss:.4f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot step losses if provided
    if step_losses:
        steps = range(1, len(step_losses) + 1)
        ax2.plot(steps, step_losses, 'r-', linewidth=1, alpha=0.7)
        ax2.set_title('Training Loss per Step', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add moving average
        window_size = min(50, len(step_losses) // 10)
        if window_size > 1:
            moving_avg = np.convolve(step_losses, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size, len(step_losses) + 1), moving_avg, 'g-', linewidth=2, label=f'Moving Average (window={window_size})')
            ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def plot_gradient_norms(gradient_norms: List[float], save_path: str = None, show_plot: bool = True):
    """
    Plot gradient norms over training to monitor gradient flow
    
    Args:
        gradient_norms: List of gradient norms for each epoch
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(gradient_norms) + 1)
    
    plt.plot(epochs, gradient_norms, 'g-', linewidth=2, marker='s', markersize=6)
    plt.title('Gradient Norms Over Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    avg_grad_norm = np.mean(gradient_norms)
    max_grad_norm = max(gradient_norms)
    min_grad_norm = min(gradient_norms)
    
    plt.text(0.02, 0.98, f'Avg Norm: {avg_grad_norm:.2f}\nMax Norm: {max_grad_norm:.2f}\nMin Norm: {min_grad_norm:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def create_training_summary_plot(epoch_losses: List[float], gradient_norms: List[float] = None,
                                save_path: str = None, show_plot: bool = True):
    """
    Create a comprehensive training summary plot
    
    Args:
        epoch_losses: List of loss values for each epoch
        gradient_norms: List of gradient norms for each epoch (optional)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    if gradient_norms:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot losses
    epochs = range(1, len(epoch_losses) + 1)
    ax1.plot(epochs, epoch_losses, 'b-', linewidth=2, marker='o', markersize=6)
    ax1.set_title('Training Progress', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add loss statistics
    min_loss = min(epoch_losses)
    final_loss = epoch_losses[-1]
    improvement = epoch_losses[0] - final_loss
    
    ax1.text(0.02, 0.98, f'Final Loss: {final_loss:.4f}\nMin Loss: {min_loss:.4f}\nTotal Improvement: {improvement:.4f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot gradient norms if provided
    if gradient_norms:
        ax2.plot(epochs, gradient_norms, 'g-', linewidth=2, marker='s', markersize=6)
        ax2.set_title('Gradient Flow', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Gradient Norm', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add gradient statistics
        avg_grad_norm = np.mean(gradient_norms)
        ax2.text(0.02, 0.98, f'Avg Gradient Norm: {avg_grad_norm:.2f}', 
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        ax1.set_xlabel('Epoch', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training summary plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close() 