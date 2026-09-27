import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data(csv_path="tourism_project/data/tourism.csv"):
    print(f"Attempting to load data from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print("Dataset loaded successfully for preparation.")

        # Drop unnecessary columns (CustomerID, Designation, MonthlyIncome, ProductPitched)
        # As per initial data exploration or problem understanding, these might not be relevant for direct modeling
        # CustomerID is an identifier, Designation might be too granular, MonthlyIncome has missing values,
        # ProductPitched is usually a feature in pre-pitch analysis, but for 'purchase decision' after pitch it might be less direct.
        # 'PitchSatisfactionScore' and 'DurationOfPitch' are more direct post-pitch indicators.
        columns_to_drop = ["CustomerID", "Designation", "MonthlyIncome", "ProductPitched"]
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")

        # Handle missing values (example: fill with mode for categorical, median for numerical)
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object': # Categorical
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else: # Numerical
                    df[col].fillna(df[col].median(), inplace=True)
        print("Handled missing values.")

        # Define features (X) and target (y)
        X = df.drop("ProdTaken", axis=1)
        y = df["ProdTaken"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print("Data split into training and testing sets.")

        # Create directory for processed data if it doesn't exist
        output_dir = "tourism_project/data"
        os.makedirs(output_dir, exist_ok=True)

        # Save the split datasets locally
        X_train.to_csv(os.path.join(output_dir, "Xtrain.csv"), index=False)
        X_test.to_csv(os.path.join(output_dir, "Xtest.csv"), index=False)
        y_train.to_csv(os.path.join(output_dir, "ytrain.csv"), index=False)
        y_test.to_csv(os.path.join(output_dir, "ytest.csv"), index=False)
        print(f"Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv to {output_dir}")

    except Exception as e:
        print(f"An error occurred during data preparation: {e}")

if __name__ == "__main__":
    prepare_data()
