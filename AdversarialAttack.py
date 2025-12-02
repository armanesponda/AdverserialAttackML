"""
FGSM Adversarial Attack - Fixed and Complete
Works with sklearn models by converting to PyTorch for gradient computation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
import os
import pickle


class SKLearnToPyTorch(nn.Module):
    """
    Converts sklearn MLPClassifier to PyTorch for FGSM attacks
    This allows us to compute gradients with respect to inputs
    """
    def __init__(self, sklearn_model):
        super(SKLearnToPyTorch, self).__init__()
        
        self.n_layers = sklearn_model.n_layers_
        self.activation = sklearn_model.activation
        
        # Build PyTorch layers matching sklearn architecture
        layers = []
        for i in range(len(sklearn_model.coefs_)):
            in_features = sklearn_model.coefs_[i].shape[0]
            out_features = sklearn_model.coefs_[i].shape[1]
            linear = nn.Linear(in_features, out_features)
            
            # Copy weights from sklearn model
            linear.weight.data = torch.FloatTensor(sklearn_model.coefs_[i].T)
            linear.bias.data = torch.FloatTensor(sklearn_model.intercepts_[i])
            
            layers.append(linear)
            
            # Add activation (except for output layer)
            if i < len(sklearn_model.coefs_) - 1:
                if self.activation == 'relu':
                    layers.append(nn.ReLU())
                elif self.activation == 'tanh':
                    layers.append(nn.Tanh())
                elif self.activation == 'logistic':
                    layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class FGSMAttacker:
    """
    Fast Gradient Sign Method Attack - Implemented from scratch
    No adversarial attack libraries used
    """
    
    def __init__(self, sklearn_model, device='cpu', log_dir='./logs'):
        self.device = device
        self.log_dir = log_dir
        
        # Convert sklearn model to PyTorch
        print("Converting sklearn model to PyTorch for gradient computation...")
        self.model = SKLearnToPyTorch(sklearn_model)
        self.model.to(device)
        self.model.eval()
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.experiment_log = []
        self.experiment_number = 1
        
        print("✓ Model conversion complete")

    def fgsm_attack(self, inputs, labels, epsilon):
        """
        FGSM Attack Algorithm (implemented from scratch):
        1. Forward pass to compute loss
        2. Backward pass to compute ∇_x L (gradient w.r.t. input)
        3. Create perturbation: δ = ε * sign(∇_x L)
        4. Generate adversarial example: x_adv = x + δ
        """
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
    
    def evaluate_attack(self, X_test, y_test, epsilons=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]):
        """
        Comprehensive evaluation of FGSM attack across different epsilon values
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
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.LongTensor(y_test).to(self.device)
        
        baseline_accuracy = None
        
        print("\n" + "="*80)
        print("FGSM ATTACK EVALUATION")
        print("="*80)
        
        for epsilon in epsilons:
            start_time = time.time()
            correct = 0
            total = len(X_test)
            successful_attacks = 0
            total_l2_perturbation = 0.0
            total_linf_perturbation = 0.0
            
            # Process in batches
            batch_size = 256
            for i in range(0, len(X_test), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                if epsilon == 0:
                    # Baseline - no attack
                    with torch.no_grad():
                        outputs = self.model(batch_X)
                        _, predicted = torch.max(outputs, 1)
                else:
                    # Get original predictions
                    with torch.no_grad():
                        original_outputs = self.model(batch_X)
                        _, original_preds = torch.max(original_outputs, 1)
                    
                    # Generate adversarial examples
                    adv_batch = self.fgsm_attack(batch_X.clone(), batch_y, epsilon)
                    
                    # Calculate perturbation metrics
                    perturbation = (adv_batch - batch_X).cpu().numpy()
                    total_l2_perturbation += np.linalg.norm(perturbation, ord=2, axis=1).sum()
                    total_linf_perturbation += np.abs(perturbation).max(axis=1).sum()
                    
                    # Get adversarial predictions
                    with torch.no_grad():
                        outputs = self.model(adv_batch)
                        _, predicted = torch.max(outputs, 1)
                    
                    # Count successful attacks
                    successful_attacks += (original_preds != predicted).sum().item()
                
                correct += (predicted == batch_y).sum().item()
            
            accuracy = 100 * correct / total
            eval_time = time.time() - start_time
            
            if epsilon == 0:
                baseline_accuracy = accuracy
            
            attack_success_rate = 0 if epsilon == 0 else (100 * successful_attacks / total)
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
        
        print("="*80)
        return results
    
    def evaluate_targeted_attack(self, X_test, y_test, epsilon=0.1, num_target_classes=10):
        """
        Evaluate targeted FGSM attacks across different target classes
        """
        results = {
            'target_class': [],
            'success_rate': [],
            'avg_confidence': []
        }
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.LongTensor(y_test).to(self.device)
        
        print("\n" + "="*80)
        print("TARGETED FGSM ATTACK EVALUATION")
        print("="*80)
        
        for target_class in range(num_target_classes):
            successful = 0
            total = 0
            total_confidence = 0.0
            
            # Only attack samples not already in target class
            mask = y_tensor != target_class
            if mask.sum() == 0:
                continue
            
            inputs = X_tensor[mask]
            labels = y_tensor[mask]
            
            # Create target labels
            target_labels = torch.full_like(labels, target_class)
            
            # Process in batches
            batch_size = 256
            for i in range(0, len(inputs), batch_size):
                batch_X = inputs[i:i+batch_size]
                batch_target = target_labels[i:i+batch_size]
                
                # Generate targeted adversarial examples
                adv_inputs = self.targeted_fgsm_attack(batch_X, batch_target, epsilon)
                
                # Evaluate
                with torch.no_grad():
                    outputs = self.model(adv_inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                
                successful += (predicted == target_class).sum().item()
                total += batch_X.size(0)
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
        
        print("="*80)
        return results
    
    def analyze_sample_predictions(self, X_test, y_test, epsilon=0.1, num_samples=10):
        """
        Detailed analysis of how FGSM affects individual samples
        """
        X_tensor = torch.FloatTensor(X_test[:num_samples]).to(self.device)
        y_tensor = torch.LongTensor(y_test[:num_samples]).to(self.device)
        
        # Get original predictions
        with torch.no_grad():
            original_outputs = self.model(X_tensor)
            original_probs = torch.softmax(original_outputs, dim=1)
            original_confidence, original_preds = torch.max(original_probs, 1)
        
        # Generate adversarial examples
        adv_inputs = self.fgsm_attack(X_tensor.clone(), y_tensor, epsilon)
        
        # Get adversarial predictions
        with torch.no_grad():
            adv_outputs = self.model(adv_inputs)
            adv_probs = torch.softmax(adv_outputs, dim=1)
            adv_confidence, adv_preds = torch.max(adv_probs, 1)
        
        # Calculate perturbation statistics
        perturbation = (adv_inputs - X_tensor).cpu().numpy()
        
        # Create detailed analysis DataFrame
        analysis_data = []
        for i in range(num_samples):
            sample_dict = {
                'Sample_ID': i + 1,
                'True_Label': y_tensor[i].item(),
                'Original_Prediction': original_preds[i].item(),
                'Original_Confidence': original_confidence[i].item(),
                'Adversarial_Prediction': adv_preds[i].item(),
                'Adversarial_Confidence': adv_confidence[i].item(),
                'Attack_Success': (original_preds[i] != adv_preds[i]).item(),
                'Confidence_Drop': (original_confidence[i] - adv_confidence[i]).item(),
                'L2_Perturbation': np.linalg.norm(perturbation[i]),
                'Linf_Perturbation': np.abs(perturbation[i]).max(),
                'Mean_Perturbation': np.mean(np.abs(perturbation[i]))
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
            print(f"\n✓ Experiment log saved to: {filepath}")
            return filepath
        else:
            print("No experiments to log.")
            return None
    
    def plot_accuracy_vs_epsilon(self, results, save_path='./results'):
        """
        Generate comprehensive visualization of attack results
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
        print(f"\n✓ Comprehensive results plot saved to: {plot_path}")
        plt.show()
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(save_path, 'fgsm_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Results data saved to: {csv_path}")
    
    def plot_perturbation_heatmap(self, X_test, y_test, epsilon=0.1, num_samples=5, 
                                  save_path='./results'):
        """
        Visualize perturbations as heatmaps for individual samples
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Get samples
        X_tensor = torch.FloatTensor(X_test[:num_samples]).to(self.device)
        y_tensor = torch.LongTensor(y_test[:num_samples]).to(self.device)
        
        # Generate adversarial examples
        adv_inputs = self.fgsm_attack(X_tensor.clone(), y_tensor, epsilon)
        perturbation = (adv_inputs - X_tensor).cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original input
            im1 = axes[i, 0].imshow(X_tensor[i].cpu().numpy().reshape(1, -1), 
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
        print(f"\n✓ Perturbation heatmaps saved to: {heatmap_path}")
        plt.show()


def main():
    """
    Main execution - connects to your trained sklearn model
    """
    print("="*80)
    print("FGSM ADVERSARIAL ATTACK ON POKER HAND CLASSIFIER")
    print("="*80)
    
    # Load trained sklearn model
    print("\nLoading trained sklearn model...")
    try:
        with open('best_sklearn_model.pkl', 'rb') as f:
            sklearn_model = pickle.load(f)
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Error: best_sklearn_model.pkl not found!")
        print("Please run the neural network training script first.")
        return
    
    # Load test data
    print("Loading test data...")
    try:
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        print(f"✓ Test data loaded: {len(X_test)} samples")
    except FileNotFoundError:
        print("❌ Error: Test data files not found!")
        print("Please run the neural network training script first.")
        return
    
    # Initialize attacker
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    attacker = FGSMAttacker(sklearn_model, device=device, log_dir='./logs')
    
    # Run FGSM attack evaluation
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results = attacker.evaluate_attack(X_test, y_test, epsilons)
    
    # Analyze specific samples
    print("\n" + "="*80)
    print("SAMPLE-LEVEL ANALYSIS")
    print("="*80)
    analysis_df = attacker.analyze_sample_predictions(X_test, y_test, epsilon=0.15, num_samples=10)
    
    # Evaluate targeted attacks
    print("\n" + "="*80)
    print("TARGETED ATTACK ANALYSIS")
    print("="*80)
    targeted_results = attacker.evaluate_targeted_attack(X_test, y_test, epsilon=0.15, num_target_classes=10)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    attacker.plot_accuracy_vs_epsilon(results, save_path='./results')
    attacker.plot_perturbation_heatmap(X_test, y_test, epsilon=0.15, num_samples=5, save_path='./results')
    
    # Save all results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save experiment log
    attacker.save_experiment_log('fgsm_experiment_log.csv')
    
    # Save sample analysis
    os.makedirs('./results', exist_ok=True)
    analysis_df.to_csv('./results/sample_analysis.csv', index=False)
    print("✓ Sample analysis saved to: ./results/sample_analysis.csv")
    
    # Save targeted results
    targeted_df = pd.DataFrame(targeted_results)
    targeted_df.to_csv('./results/targeted_attack_results.csv', index=False)
    print("✓ Targeted attack results saved to: ./results/targeted_attack_results.csv")
    
    # Print final summary
    print("\n" + "="*80)
    print("ATTACK SUMMARY")
    print("="*80)
    baseline_acc = results['accuracy'][0]
    final_acc = results['accuracy'][-1]
    print(f"Baseline Accuracy (ε=0):      {baseline_acc:.2f}%")
    print(f"Accuracy at ε={epsilons[-1]}:          {final_acc:.2f}%")
    print(f"Accuracy Drop:                 {baseline_acc - final_acc:.2f}%")
    print(f"Max Attack Success Rate:       {max(results['attack_success_rate']):.2f}%")
    print("="*80)
    print("\n✓ All results ready for your IEEE conference paper!")
    print("="*80)


if __name__ == "__main__":
    main()
