import numpy as np
import pandas as pd
import joblib

# Data Processing Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Handling Class Imbalance
from imblearn.over_sampling import SMOTE

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif

# Multi-class classification
from sklearn.multiclass import OneVsRestClassifier


class PTSDSeverityPredictor:
    def __init__(self, dataset_path):
        """
        Initialize the PTSD severity prediction pipeline.

        Args:
            dataset_path (str): Path to the PTSD dataset
        """
        self.dataset_path = dataset_path
        self.data = None
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        """Load and preprocess the PTSD dataset."""
        try:
            # Read the dataset
            self.data = pd.read_csv(self.dataset_path)

            # Check if data is loaded
            if self.data.empty:
                raise ValueError("The dataset is empty.")
            features = [col for col in self.data.columns if col != 'Severity Level' and col != 'Total Score']

            # Split data into features and target
            self.X = self.data.drop('Severity Level', axis=1)
            self.y = self.data['Severity Level']

            # Convert to numeric and handle errors
            self.X = self.X.apply(pd.to_numeric, errors='coerce')

            print("Dataset loaded successfully.")
            print(f"Total samples: {len(self.X)}")
            print(f"Features: {features}")
            print("Missing values:\n", self.X.isnull().sum())

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def preprocess_data(self):
        """Handle missing values, encode target, and scale features."""
        try:
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(self.X)

            # Label encoding for target
            self.y = self.label_encoder.fit_transform(self.y)

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)

            return X_scaled, scaler

        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise

    def train_models(self):
        """Train and evaluate models with cross-validation."""
        try:
            # Preprocess the data
            X_processed, scaler = self.preprocess_data()

            # Handle class imbalance with SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_processed, self.y)

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )

            # Models to evaluate
            models = {
                'RandomForest': RandomForestClassifier(),
                'LogisticRegression': LogisticRegression(max_iter=1000),
                'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000),
                'XGBoost': XGBClassifier()
            }

            results = {}

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results[model_name] = {
                    'model': model,
                    'accuracy': np.mean(y_pred == y_test),
                    'report': classification_report(y_test, y_pred)
                }

            # Save the best model and scaler
            best_model_name = max(results, key=lambda k: results[k]['accuracy'])
            best_model = results[best_model_name]['model']

            joblib.dump(best_model, 'best_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')

            # Save label encoder
            joblib.dump(self.label_encoder, 'label_encoder.pkl')

            return results

        except Exception as e:
            print(f"Model training error: {e}")
            raise


# Example Usage
if __name__ == "__main__":
    try:
        # Initialize the model pipeline
        predictor = PTSDSeverityPredictor('PTSD_Quiz_Severity_Prediction_Balanced_Dataset.csv')

        # Load and preprocess data
        predictor.load_data()

        # Train and evaluate models
        results = predictor.train_models()

        # Print evaluation results
        for model_name, result in results.items():
            print(f"\n{model_name} Model:")
            print(f"Accuracy: {result['accuracy']}")
            print("Classification Report:")
            print(result['report'])

    except Exception as e:
        print(f"An error occurred: {e}")
