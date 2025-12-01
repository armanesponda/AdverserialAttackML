import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from itertools import product


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile, header=None)

    def preprocess(self):
        self.processed_data = self.raw_input.copy()
    
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income' ]

        if self.processed_data.columns[0] == 0:
            self.processed_data.columns = column_names

        #Missing values are noted by ?
        self.processed_data.replace(' ?', np.nan, inplace=True)

        #Drops rows with missing values
        self.processed_data.dropna(inplace=True)

        categorical_cols = self.processed_data.select_dtypes(include=['object']).columns.tolist()

        target_col = 'income'
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        numerical_cols = self.processed_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        
        #Convert to binary: 0 for <=50K, 1 for >50K
        self.processed_data[target_col] = self.processed_data[target_col].str.strip()
        self.processed_data[target_col] = self.processed_data[target_col].apply(
            lambda x: 1 if '>50K' in str(x) else 0
        )

        for col in categorical_cols:
            self.processed_data[col] = self.processed_data[col].str.strip()
        
        #One-hot encoding  for categorical values
        if len(categorical_cols) > 0:
            self.processed_data = pd.get_dummies(
                self.processed_data, 
                columns=categorical_cols, 
                drop_first=True
            )
        
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            self.processed_data[numerical_cols] = scaler.fit_transform(
                self.processed_data[numerical_cols]
            )
        
        self.processed_data.reset_index(drop=True, inplace=True)
        
        # Optional: Print preprocessing summary
        print(f"\nOriginal dataset shape: {self.raw_input.shape}")
        print(f"Processed dataset shape: {self.processed_data.shape}")
        print(f"Number of features after encoding: {len(self.processed_data.columns) - 1}")
        print(f"Target distribution:\n{self.processed_data[target_col].value_counts()}")
        
        return 0

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        model = MLPClassifier()
        
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] 
        num_hidden_layers = [2, 3]

        neurons_per_layer = 64;

        results = []
        all_histories = []

        combinations = product(activations, learning_rate, max_iterations, num_hidden_layers)

        model_num = 0
        #Loop for training data and testing data models with each hyperparameter combination
        for activation, lr, max_iter, n_layers in combinations:
            model_num += 1

            hidden_layer_sizes = (neurons_per_layer,) * n_layers
        
            print(f"\nTraining Model {model_num}...")
            print(f"  Activation: {activation}, LR: {lr}, Epochs: {max_iter}, Layers: {n_layers}")

            model = MLPClassifier(
                hidden_layer_sizes = hidden_layer_sizes,
                activation = activation,
                learning_rate_init = lr,
                max_iter = max_iter,
                random_state = 42
            )

            model.fit(X_train, y_train)

            y_train_predict = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
            train_mse = mean_squared_error(y_train, y_train_predict)
            train_r2 = model.score(X_train, y_train)

            y_test_predict = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
            test_mse = mean_squared_error(y_test, y_test_predict)
            test_r2 = model.score(X_test, y_test)

            #Loss, what we will plot against epochs
            train_accuracy = accuracy_score(y_train, y_train_predict)
            test_accuracy = accuracy_score(y_test, y_test_predict)

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

        return 0

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/armanesponda/AdverserialAttackML/refs/heads/main/poker-hand-testing.data?token=GHSAT0AAAAAADPEMUPCZQDMWP4ZQZ4TIQEE2JN5CGQ"
    neural_network = NeuralNet(url) 
    neural_network.preprocess()
    neural_network.train_evaluate()
