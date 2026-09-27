import pandas as pd
import os

def register_dataset(csv_path="tourism_project/data/tourism.csv"):
    """
    Reads the tourism dataset, checks for expected columns, and prints a summary.
    """
    print(f"Attempting to register dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print("Dataset loaded successfully.")

        expected_columns = [
            "CustomerID",
            "ProdTaken",
            "Age",
            "TypeofContact",
            "CityTier",
            "Occupation",
            "Gender",
            "NumberOfPersonVisiting",
            "PreferredPropertyStar",
            "MaritalStatus",
            "NumberOfTrips",
            "Passport",
            "OwnCar",
            "NumberOfChildrenVisiting",
            "Designation",
            "MonthlyIncome",
            "PitchSatisfactionScore",
            "ProductPitched",
            "NumberOfFollowups",
            "DurationOfPitch",
        ]

        # Check if all expected columns are present
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: The following expected columns are missing: {missing_columns}")
        else:
            print("All expected columns are present.")

        print("\n--- Dataset Summary ---")
        print(df.info())
        print("\n--- First 5 rows ---")
        print(df.head())
        print("\n--- Descriptive Statistics ---")
        print(df.describe())

    except Exception as e:
        print(f"An error occurred while processing the dataset: {e}")

if __name__ == "__main__":
    # This assumes the script is run from the project root or tourism.csv is in data/
    register_dataset()
