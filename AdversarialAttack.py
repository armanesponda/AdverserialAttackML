import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import time
from datetime import datetime
import json
import os

class FGSMAttacker:
    def __init__(self, model, device='cpu', log_dir='./logs'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.log_dir = log_dir

        os.makedirs(log_dir, exist_ok=True)

        self.experiment_log = []
        self.experiment_number = 1

    def fgsm_attack(self, inputs, labels, epsilon):
        inputs.requires_grad = True

        outputs = self.model(inputs)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)

        self.model.zero_grad()

        loss.backward()

        data_grad = inputs.grad.data

        sign_data_grad = data_grad.sign()

        perturbed_data = inputs + epsilon * sign_data_grad

        return perturbed_data.detach()
    
    def targeted_fgsm_attack(self, inputs, target_labels, epsilon):
        """
        Targeted FGSM attack - tries to make model predict a specific target class
        
        Algorithm: x_adv = x - ε * sign(∇_x L(θ, x, y_target))
        Note the minus sign - we minimize loss for the target class
        
        Args:
            inputs: Original input samples
            target_labels: Desired target labels
            epsilon: Perturbation magnitude
        
        Returns:
            Adversarial examples
        """
        inputs.requires_grad = True
        outputs = self.model(inputs)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, target_labels)
        
        self.model.zero_grad()
        loss.backward()
        
        data_grad = inputs.grad.data
        sign_data_grad = data_grad.sign()
        
        # Subtract perturbation to minimize loss for target class
        perturbed_data = inputs - epsilon * sign_data_grad
        
        return perturbed_data.detach()
    
def evaluate_attack(self, test_loader, epsilons=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]):
        """
        Comprehensive evaluation of FGSM attack across different epsilon values
        
        Args:
            test_loader: DataLoader for test data
            epsilons: List of perturbation magnitudes to test
        
        Returns:
            Dictionary containing detailed results for each epsilon
        """
        results = {
            'epsilon': [],
            'accuracy': [],
            'total_samples': [],
            'correct_predictions': [],
            'attack_success_rate': [],
            'avg_perturbation_l2': [],
            'avg_perturbation_linf': [],
            'evaluation_time': []
        }
        
        baseline_accuracy = None
        
        for epsilon in epsilons:
            start_time = time.time()
            correct = 0
            total = 0
            successful_attacks = 0
            total_l2_perturbation = 0.0
            total_linf_perturbation = 0.0
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if epsilon == 0:
                    # Baseline evaluation - no attack
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                else:
                    # Get original predictions
                    with torch.no_grad():
                        original_outputs = self.model(inputs)
                        _, original_preds = torch.max(original_outputs.data, 1)
                    
                    # Generate adversarial examples
                    adv_inputs = self.fgsm_attack(inputs.clone(), labels, epsilon)
                    
                    # Calculate perturbation metrics
                    perturbation = (adv_inputs - inputs).cpu().numpy()
                    total_l2_perturbation += np.linalg.norm(perturbation, ord=2, axis=1).sum()
                    total_linf_perturbation += np.abs(perturbation).max(axis=1).sum()
                    
                    # Evaluate on adversarial examples
                    with torch.no_grad():
                        outputs = self.model(adv_inputs)
                        _, predicted = torch.max(outputs.data, 1)
                    
                    # Count successful attacks (predictions changed)
                    successful_attacks += (original_preds != predicted).sum().item()
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            eval_time = time.time() - start_time
            
            # Store baseline accuracy
            if epsilon == 0:
                baseline_accuracy = accuracy
            
            # Calculate attack success rate
            attack_success_rate = 0 if epsilon == 0 else (100 * successful_attacks / total)
            
            # Calculate average perturbations
            avg_l2 = 0 if epsilon == 0 else total_l2_perturbation / total
            avg_linf = 0 if epsilon == 0 else total_linf_perturbation / total
            
            # Store results
            results['epsilon'].append(epsilon)
            results['accuracy'].append(accuracy)
            results['total_samples'].append(total)
            results['correct_predictions'].append(correct)
            results['attack_success_rate'].append(attack_success_rate)
            results['avg_perturbation_l2'].append(avg_l2)
            results['avg_perturbation_linf'].append(avg_linf)
            results['evaluation_time'].append(eval_time)
            
            # Log experiment
            self._log_experiment(epsilon, accuracy, attack_success_rate, 
                               avg_l2, avg_linf, total, eval_time)
            
            print(f'Epsilon: {epsilon:.3f} | Accuracy: {accuracy:.2f}% | '
                  f'Attack Success Rate: {attack_success_rate:.2f}% | '
                  f'Time: {eval_time:.2f}s')
        
        return results
    
def evaluate_targeted_attack(self, test_loader, epsilon=0.1, num_target_classes=10):
        """
        Evaluate targeted FGSM attacks across different target classes
        
        Args:
            test_loader: DataLoader for test data
            epsilon: Perturbation magnitude
            num_target_classes: Number of classes in classification problem
        
        Returns:
            Dictionary with targeted attack results
        """
        results = {
            'target_class': [],
            'success_rate': [],
            'avg_confidence': []
        }
        
        for target_class in range(num_target_classes):
            successful = 0
            total = 0
            total_confidence = 0.0
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Only attack samples not already in target class
                mask = labels != target_class
                if mask.sum() == 0:
                    continue
                
                inputs = inputs[mask]
                labels = labels[mask]
                
                # Create target labels
                target_labels = torch.full_like(labels, target_class)
                
                # Generate targeted adversarial examples
                adv_inputs = self.targeted_fgsm_attack(inputs, target_labels, epsilon)
                
                # Evaluate
                with torch.no_grad():
                    outputs = self.model(adv_inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                
                successful += (predicted == target_class).sum().item()
                total += inputs.size(0)
                total_confidence += probabilities[:, target_class].sum().item()
            
            if total > 0:
                success_rate = 100 * successful / total
                avg_confidence = total_confidence / total
            else:
                success_rate = 0
                avg_confidence = 0
            
            results['target_class'].append(target_class)
            results['success_rate'].append(success_rate)
            results['avg_confidence'].append(avg_confidence)
            
            print(f'Target Class: {target_class} | Success Rate: {success_rate:.2f}% | '
                  f'Avg Confidence: {avg_confidence:.4f}')
        
        return results
    
def analyze_sample_predictions(self, test_loader, epsilon=0.1, num_samples=10):
        """
        Detailed analysis of how FGSM affects individual samples
        
        Args:
            test_loader: DataLoader for test data
            epsilon: Perturbation magnitude
            num_samples: Number of samples to analyze
        
        Returns:
            DataFrame with detailed sample-level analysis
        """
        # Get a batch of test data
        inputs, labels = next(iter(test_loader))
        inputs = inputs[:num_samples].to(self.device)
        labels = labels[:num_samples].to(self.device)
        
        # Get original predictions
        with torch.no_grad():
            original_outputs = self.model(inputs)
            original_probs = torch.softmax(original_outputs, dim=1)
            original_confidence, original_preds = torch.max(original_probs, 1)
        
        # Generate adversarial examples
        adv_inputs = self.fgsm_attack(inputs.clone(), labels, epsilon)
        
        # Get adversarial predictions
        with torch.no_grad():
            adv_outputs = self.model(adv_inputs)
            adv_probs = torch.softmax(adv_outputs, dim=1)
            adv_confidence, adv_preds = torch.max(adv_probs, 1)
        
        # Calculate perturbation statistics
        perturbation = (adv_inputs - inputs).cpu().numpy()
        
        # Create detailed analysis DataFrame
        analysis_data = []
        for i in range(num_samples):
            sample_dict = {
                'Sample_ID': i + 1,
                'True_Label': labels[i].item(),
                'Original_Prediction': original_preds[i].item(),
                'Original_Confidence': original_confidence[i].item(),
                'Adversarial_Prediction': adv_preds[i].item(),
                'Adversarial_Confidence': adv_confidence[i].item(),
                'Attack_Success': (original_preds[i] != adv_preds[i]).item(),
                'Confidence_Drop': (original_confidence[i] - adv_confidence[i]).item(),
                'L2_Perturbation': np.linalg.norm(perturbation[i]),
                'Linf_Perturbation': np.abs(perturbation[i]).max(),
                'Mean_Perturbation': np.mean(np.abs(perturbation[i])),
                'Prediction_Changed_To_Wrong_Class': (adv_preds[i] != labels[i]).item() and 
                                                      (original_preds[i] != adv_preds[i]).item()
            }
            analysis_data.append(sample_dict)
        
        analysis_df = pd.DataFrame(analysis_data)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Sample-Level Analysis (Epsilon={epsilon})")
        print(f"{'='*80}")
        print(analysis_df.to_string(index=False))
        print(f"\nSummary Statistics:")
        print(f"  Attack Success Rate: {analysis_df['Attack_Success'].mean()*100:.2f}%")
        print(f"  Average Confidence Drop: {analysis_df['Confidence_Drop'].mean():.4f}")
        print(f"  Average L2 Perturbation: {analysis_df['L2_Perturbation'].mean():.4f}")
        print(f"  Average L-inf Perturbation: {analysis_df['Linf_Perturbation'].mean():.4f}")
        
        return analysis_df
    
def _log_experiment(self, epsilon, accuracy, attack_success_rate, 
                       avg_l2, avg_linf, total_samples, eval_time):
        """Internal method to log experiment parameters and results"""
        experiment_entry = {
            'Experiment_Number': self.experiment_number,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Attack_Type': 'FGSM',
            'Epsilon': epsilon,
            'Accuracy': f'{accuracy:.2f}%',
            'Attack_Success_Rate': f'{attack_success_rate:.2f}%',
            'Avg_L2_Perturbation': f'{avg_l2:.4f}',
            'Avg_Linf_Perturbation': f'{avg_linf:.4f}',
            'Total_Samples': total_samples,
            'Evaluation_Time': f'{eval_time:.2f}s'
        }
        
        self.experiment_log.append(experiment_entry)
        self.experiment_number += 1
    
def save_experiment_log(self, filename='experiment_log.csv'):
        """Save experiment log to CSV file"""
        if self.experiment_log:
            df = pd.DataFrame(self.experiment_log)
            filepath = os.path.join(self.log_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"\nExperiment log saved to: {filepath}")
            return filepath
        else:
            print("No experiments to log.")
            return None
    
def plot_accuracy_vs_epsilon(self, results, save_path='./results'):
        """
        Generate comprehensive visualization of attack results
        
        Args:
            results: Dictionary from evaluate_attack()
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy vs Epsilon
        axes[0, 0].plot(results['epsilon'], results['accuracy'], 
                       'b-o', linewidth=2, markersize=8, label='Model Accuracy')
        axes[0, 0].set_xlabel('Epsilon (ε)', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 0].set_title('Model Accuracy Under FGSM Attack', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Attack Success Rate vs Epsilon
        axes[0, 1].plot(results['epsilon'], results['attack_success_rate'], 
                       'r-s', linewidth=2, markersize=8, label='Attack Success Rate')
        axes[0, 1].set_xlabel('Epsilon (ε)', fontsize=12)
        axes[0, 1].set_ylabel('Attack Success Rate (%)', fontsize=12)
        axes[0, 1].set_title('FGSM Attack Success Rate', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: L2 and L-inf Perturbation Norms
        axes[1, 0].plot(results['epsilon'], results['avg_perturbation_l2'], 
                       'g-^', linewidth=2, markersize=8, label='L2 Norm')
        axes[1, 0].plot(results['epsilon'], results['avg_perturbation_linf'], 
                       'm-v', linewidth=2, markersize=8, label='L-infinity Norm')
        axes[1, 0].set_xlabel('Epsilon (ε)', fontsize=12)
        axes[1, 0].set_ylabel('Average Perturbation Magnitude', fontsize=12)
        axes[1, 0].set_title('Perturbation Norms vs Epsilon', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Evaluation Time
        axes[1, 1].bar(range(len(results['epsilon'])), results['evaluation_time'], 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        axes[1, 1].set_xlabel('Epsilon Value Index', fontsize=12)
        axes[1, 1].set_ylabel('Evaluation Time (seconds)', fontsize=12)
        axes[1, 1].set_title('Computation Time per Epsilon', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(range(len(results['epsilon'])))
        axes[1, 1].set_xticklabels([f'{e:.2f}' for e in results['epsilon']], rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(save_path, 'fgsm_comprehensive_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nComprehensive results plot saved to: {plot_path}")
        plt.show()
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(save_path, 'fgsm_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Results data saved to: {csv_path}")
    
def plot_perturbation_heatmap(self, test_loader, epsilon=0.1, num_samples=5, 
                                  save_path='./results'):
        """
        Visualize perturbations as heatmaps for individual samples
        
        Args:
            test_loader: DataLoader for test data
            epsilon: Perturbation magnitude
            num_samples: Number of samples to visualize
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Get samples
        inputs, labels = next(iter(test_loader))
        inputs = inputs[:num_samples].to(self.device)
        labels = labels[:num_samples].to(self.device)
        
        # Generate adversarial examples
        adv_inputs = self.fgsm_attack(inputs.clone(), labels, epsilon)
        perturbation = (adv_inputs - inputs).cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original input
            im1 = axes[i, 0].imshow(inputs[i].cpu().numpy().reshape(1, -1), 
                                    cmap='viridis', aspect='auto')
            axes[i, 0].set_title(f'Sample {i+1}: Original Input')
            axes[i, 0].set_ylabel('Features')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Perturbation
            im2 = axes[i, 1].imshow(perturbation[i].reshape(1, -1), 
                                    cmap='RdBu', aspect='auto', 
                                    vmin=-epsilon, vmax=epsilon)
            axes[i, 1].set_title(f'Perturbation (ε={epsilon})')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Adversarial input
            im3 = axes[i, 2].imshow(adv_inputs[i].cpu().numpy().reshape(1, -1), 
                                    cmap='viridis', aspect='auto')
            axes[i, 2].set_title(f'Adversarial Input')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.tight_layout()
        heatmap_path = os.path.join(save_path, 'perturbation_heatmaps.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"\nPerturbation heatmaps saved to: {heatmap_path}")
        plt.show()


def main():
    """
    Main execution function demonstrating complete FGSM attack workflow
    
    This would be connected to the trained poker hand model
    """
    print("="*80)
    print("FGSM Adversarial Attack Module - From Scratch Implementation")
    print("="*80)
    print("\nThis module implements the Fast Gradient Sign Method without")
    print("using any built-in adversarial attack libraries.")
    print("\nTo use this module:")
    print("1. Train your poker hand neural network using poker_nn_trainer.py")
    print("2. Load the trained model")
    print("3. Initialize FGSMAttacker with your model")
    print("4. Run evaluate_attack() to test model robustness")
    print("5. Analyze results and generate visualizations")
    print("\nExample usage is shown in the commented code below.")
    print("="*80)
    
    # Uncomment and modify the following code to run with your trained model:
    """
    from poker_nn_trainer import PokerHandNN, PokerHandDataset, load_and_preprocess_data
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading poker hand dataset...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        'poker-hand-training-true.data', 
        'poker-hand-testing.data'
    )
    
    test_dataset = PokerHandDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Load trained model
    print("Loading trained model...")
    model = PokerHandNN(input_size=10, hidden_sizes=[128, 64, 32], num_classes=10)
    model.load_state_dict(torch.load('best_poker_model.pth'))
    model.to(device)
    
    # Initialize FGSM attacker
    print("\nInitializing FGSM attacker...")
    attacker = FGSMAttacker(model, device=device, log_dir='./logs')
    
    # Experiment 1: Evaluate attack with different epsilon values
    print("\n" + "="*80)
    print("Experiment 1: Untargeted FGSM Attack Evaluation")
    print("="*80)
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results = attacker.evaluate_attack(test_loader, epsilons)
    
    # Experiment 2: Sample-level analysis
    print("\n" + "="*80)
    print("Experiment 2: Sample-Level Prediction Analysis")
    print("="*80)
    analysis_df = attacker.analyze_sample_predictions(test_loader, epsilon=0.15, num_samples=10)
    
    # Experiment 3: Targeted attacks
    print("\n" + "="*80)
    print("Experiment 3: Targeted FGSM Attack Evaluation")
    print("="*80)
    targeted_results = attacker.evaluate_targeted_attack(test_loader, epsilon=0.15, num_target_classes=10)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    attacker.plot_accuracy_vs_epsilon(results, save_path='./results')
    attacker.plot_perturbation_heatmap(test_loader, epsilon=0.15, num_samples=5, save_path='./results')
    
    # Save experiment log
    print("\n" + "="*80)
    print("Saving Experiment Log")
    print("="*80)
    attacker.save_experiment_log('fgsm_experiment_log.csv')
    
    # Save sample analysis
    analysis_df.to_csv('./results/sample_analysis.csv', index=False)
    print(f"Sample analysis saved to: ./results/sample_analysis.csv")
    
    print("\n" + "="*80)
    print("FGSM Attack Analysis Complete!")
    print("="*80)
    """


if __name__ == "__main__":
    main()