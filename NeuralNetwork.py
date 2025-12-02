import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from itertools import product
import pickle


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile, header=None)

    def preprocess(self):
        self.processed_data = self.raw_input.copy()
    
        column_names = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']

        if self.processed_data.columns[0] == 0:
            self.processed_data.columns = column_names

        target_col = 'CLASS'
        
        numerical_cols = [col for col in column_names if col != 'CLASS']
        
        self.scaler = StandardScaler()
        self.processed_data[numerical_cols] = self.scaler.fit_transform(
            self.processed_data[numerical_cols]
        )
        
        self.processed_data.reset_index(drop=True, inplace=True)
        
        # Optional: Print preprocessing summary
        print(f"\nOriginal dataset shape: {self.raw_input.shape}")
        print(f"Processed dataset shape: {self.processed_data.shape}")
        print(f"Number of features: {len(numerical_cols)}")

        # CHANGED: Show all 10 classes instead of binary
        print(f"\nTarget distribution (Poker Hand Classes 0-9):")
        class_counts = self.processed_data[target_col].value_counts().sort_index()
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count} samples ({count/len(self.processed_data)*100:.2f}%)")
        
        return 0

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        model_configs = [
            {'activation': 'relu', 'learning_rate': 0.01, 'max_iter': 100, 'hidden_layers': 2},
            {'activation': 'relu', 'learning_rate': 0.01, 'max_iter': 200, 'hidden_layers': 2},
            {'activation': 'relu', 'learning_rate': 0.1, 'max_iter': 100, 'hidden_layers': 2},
            {'activation': 'relu', 'learning_rate': 0.01, 'max_iter': 100, 'hidden_layers': 3},
        ]
        neurons_per_layer = 64

        results = []
        all_histories = []

        model_num = 0
        best_model = None
        best_test_acc = 0
        
        for config in model_configs:
            model_num += 1
            
            activation = config['activation']
            lr = config['learning_rate']
            max_iter = config['max_iter']
            n_layers = config['hidden_layers']
            
            hidden_layer_sizes = (neurons_per_layer,) * n_layers

            print(f"\nTraining Model {model_num}...")
            print(f"  Activation: {activation}, LR: {lr}, Epochs: {max_iter}, Layers: {n_layers}")

            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=lr,
                max_iter=max_iter,
                random_state=0
            )

            model.fit(self.X_train, self.y_train)

            y_train_predict = model.predict(self.X_train)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_predict))
            train_mse = mean_squared_error(self.y_train, y_train_predict)
            train_r2 = model.score(self.X_train, self.y_train)

            y_test_predict = model.predict(self.X_test)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_predict))
            test_mse = mean_squared_error(self.y_test, y_test_predict)
            test_r2 = model.score(self.X_test, self.y_test)

            #Loss, what we will plot against epochs
            train_accuracy = accuracy_score(self.y_train, y_train_predict)
            test_accuracy = accuracy_score(self.y_test, y_test_predict)

            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                best_model = model  # Keep the actual sklearn model
                best_activation = activation
                best_lr = lr
                best_epochs = max_iter
                best_layers = n_layers

            results.append({
                'activation': activation,
                'learning_rate': lr,
                'epochs': max_iter,
                'hidden_layers': n_layers,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2
            })

            history = model.loss_curve_
            all_histories.append({
                'model_name': f"{activation}_{lr}_{max_iter}_{n_layers}",
                'loss_curve': history
            })

        #Plotting
        plt.figure(figsize=(16, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, 24)) #One color for each model

        for i, history_info in enumerate(all_histories):
            model_name = history_info['model_name']
            loss_curve = history_info['loss_curve']
            epochs = range(1, len(loss_curve) + 1)

            plt.plot(epochs,loss_curve, label=model_name, color=colors[i], linewidth=1.5, alpha=0.7)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Training/Loss History for Models", fontsize=15, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_history.png')
        plt.close()

        print("\nPlot saved as 'model_history.png'")

        results_df = pd.DataFrame(results)

        results_df = results_df[['activation', 'learning_rate', 'epochs', 'hidden_layers', 'train_accuracy', 'test_accuracy', 'train_mse', 'test_mse']]

        #Print the table
        print("\n" + "="*100)
        print("Model Results Table")
        print("="*100)
        print(results_df.to_string(index=False))
        print("="*100)

        # Print summary statistics
        print("\nSaving best model and data for adversarial attacks...")
        with open('best_sklearn_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)  # Save the actual sklearn model here
        print("\n")
        print(f"Best Test Accuracy: {results_df['test_accuracy'].max():.4f}")
        best_model_idx = results_df['test_accuracy'].idxmax()
        best_model = results_df.iloc[best_model_idx]
        print(f"Best Model Configuration:")
        print(f"  - Activation: {best_model['activation']}")
        print(f"  - Learning Rate: {best_model['learning_rate']}")
        print(f"  - Epochs: {best_model['epochs']}")
        print(f"  - Hidden Layers: {best_model['hidden_layers']}")
        print(f"  - Test Accuracy: {best_model['test_accuracy']:.4f}")
        print(f"  - Test MSE: {best_model['test_mse']:.4f}")
        print("="*100)

        print("\nSaving best model and data for adversarial attacks...")
        with open('best_sklearn_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save test data as numpy arrays
        np.save('X_test.npy', self.X_test.values)
        np.save('y_test.npy', self.y_test.values)

        return 0

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/armanesponda/PokerHandDataset/refs/heads/main/poker-hand-testing.data"
    neural_network = NeuralNet(url) 
    neural_network.preprocess()
    neural_network.train_evaluate()
