import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.xgboost

def train_model(
    X_train_path="tourism_project/data/Xtrain.csv",
    X_test_path="tourism_project/data/Xtest.csv",
    y_train_path="tourism_project/data/ytrain.csv",
    y_test_path="tourism_project/data/ytest.csv",
    model_output_dir="tourism_project/deployment"
):
    print("Starting model training process...")

    # Check if input files exist
    for path in [X_train_path, X_test_path, y_train_path, y_test_path]:
        if not os.path.exists(path):
            print(f"Error: Required data file not found at {path}")
            return

    try:
        # Load the data
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).squeeze() # .squeeze() to convert DataFrame to Series
        y_test = pd.read_csv(y_test_path).squeeze() # .squeeze() to convert DataFrame to Series
        print("Train and test data loaded successfully.")

        # Identify categorical and numerical columns
        categorical_features = X_train.select_dtypes(include=['object']).columns
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

        # Create preprocessing pipelines for numerical and categorical features
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Create a preprocessor using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create the full pipeline with preprocessor and XGBoost classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
        print("Preprocessing pipeline and XGBoost classifier initialized.")

        # Define hyperparameter grid for GridSearchCV
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5],
            'classifier__subsample': [0.7, 0.9]
        }

        # Perform GridSearchCV
        print("Starting GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print("GridSearchCV completed.")

        best_model = grid_search.best_estimator_
        print(f"Best model found with parameters: {grid_search.best_params_}")

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Evaluate the best model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n--- Model Evaluation on Test Set ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))

    except Exception as e:
        print(f"An error occurred during model training/evaluation: {e}")
        raise

    # --- Save the best model immediately after training/evaluation. ---
    # This step is deliberately OUTSIDE the try/except above (and BEFORE the
    # optional MLflow logging below) so that the model file always gets
    # written even if MLflow logging fails for any reason. This is what the
    # GitHub Actions workflow commits into tourism_project/deployment/, and
    # what the Streamlit app loads at runtime.
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

    # --- MLflow experiment tracking (best-effort, non-fatal). ---
    # Wrapped in its own try/except so any MLflow issue (registry, tracking
    # store, logging flavor, etc.) can never prevent the model artifact
    # itself from being produced - that used to be the actual bug here:
    # `best_model` is a scikit-learn Pipeline (preprocessor + XGBClassifier),
    # not a native XGBoost Booster/XGBClassifier, so mlflow.xgboost.log_model
    # tried to call best_model.save_model() and raised
    # "'Pipeline' object has no attribute 'save_model'". That exception used
    # to happen inside the same try block as joblib.dump(), so the model
    # file was never saved and Streamlit reported "best_model.joblib not
    # found". Logging with mlflow.sklearn instead of mlflow.xgboost matches
    # the actual object type, so this now succeeds instead of just being
    # silently skipped.
    try:
        mlflow.set_experiment("Tourism Package Prediction")
        with mlflow.start_run():
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
            # best_model is a sklearn Pipeline, so use the sklearn flavor
            # (not mlflow.xgboost, which expects a native XGBoost object).
            # No registered_model_name - registering requires a
            # database-backed MLflow tracking store, which the CI runner's
            # local file store does not provide.
            mlflow.sklearn.log_model(best_model, "model")
            print("MLflow experiment logged.")
    except Exception as e:
        print(f"Warning: MLflow logging failed and was skipped ({e}). "
              "This does not affect the saved model file.")

if __name__ == "__main__":
    train_model()
