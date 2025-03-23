import os
import glob
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


DATA_DIR = "data/raw"
features = ['gridposition', 'qualiposition', 'grid_vs_qualipos',
            'teamname', 'status_dnf', 'best_quali_time', 'q1', 'q2', 'q3',
            'airtemp', 'tracktemp', 'humidity', 'pressure', 'windspeed']
essential = ['gridposition', 'qualiposition', 'teamname', 'position']
seed = 3

# =============================================================================
# DecisionTree Class
# =============================================================================
class DecisionTree:
    def __init__(self, max_depth=6, min_samples_split=2):
        # Initialize the maximum tree depth and the minimum number of samples required to split a node.
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        # Build the decision tree using the training data.
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        # Predict the class labels for each sample in X.
        return [self._predict(inputs, self.tree) for inputs in X]

    def _gini(self, y):
        # Calculate the Gini impurity for a set of labels.
        counts = Counter(y)
        probs = [count / len(y) for count in counts.values()]
        return 1 - sum(p ** 2 for p in probs)

    def _best_split(self, X, y):
        # Identify the best feature and threshold to split on based on Gini impurity.
        m, n = X.shape
        best_gini = 1
        best_idx, best_val = None, None
        for idx in range(n):
            # Retrieve all unique values for the feature.
            values = np.unique(X[:, idx])
            for val in values:
                # Create masks for splitting data into left and right nodes.
                left_mask = X[:, idx] <= val
                right_mask = ~left_mask
                # Ensure both splits have at least the minimum required samples.
                if sum(left_mask) < self.min_samples_split or sum(right_mask) < self.min_samples_split:
                    continue
                # Calculate the weighted Gini impurity for this split.
                gini = (
                    len(y[left_mask]) / m * self._gini(y[left_mask]) +
                    len(y[right_mask]) / m * self._gini(y[right_mask])
                )
                # Update best split if a lower impurity is found.
                if gini < best_gini:
                    best_gini = gini
                    best_idx, best_val = idx, val
        return best_idx, best_val

    def _grow_tree(self, X, y, depth=0):
        # Recursively build the decision tree.
        # Terminate if maximum depth is reached or if the node is pure.
        if depth >= self.max_depth or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]
        idx, val = self._best_split(X, y)
        # If no valid split is found, return the majority class.
        if idx is None:
            return Counter(y).most_common(1)[0][0]

        # Create masks for the left and right splits.
        left = X[:, idx] <= val
        right = ~left
        # Return a dictionary representing the decision node.
        return {
            'feature': idx,
            'value': val,
            'left': self._grow_tree(X[left], y[left], depth+1),
            'right': self._grow_tree(X[right], y[right], depth+1)
        }

    def _predict(self, x, node):
        # Recursively traverse the tree to predict the class for a single sample.
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['value']:
            return self._predict(x, node['left'])
        else:
            return self._predict(x, node['right'])


# =============================================================================
# RandomForest Class
# =============================================================================
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2):
        # Initialize the number of trees and parameters for each tree.
        self.n_estimators = n_estimators
        self.trees = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        # Train multiple decision trees on bootstrapped samples of the data.
        self.trees = []
        for _ in range(self.n_estimators):
            # Generate bootstrap sample indices.
            np.random.seed(seed)
            idxs = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        # Aggregate predictions from all trees using majority vote.
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        final_preds = [Counter(row).most_common(1)[0][0] for row in tree_preds.T]
        return final_preds


# =============================================================================
# Helper Functions
# =============================================================================
def time_to_ms(x):
    # Convert time string to milliseconds. Return None for invalid or missing data.
    if pd.isnull(x) or x == "":
        return None
    try:
        return int(x)
    except Exception:
        return None

def group_position(pos):
    # Categorize position into defined groups.
    if pos <= 3:
        return "Top 3"
    elif pos <= 10:
        return "Midfield"
    else:
        return "Backmarker"

def merge_features():
    # Merge race, qualifying, and weather data into a single DataFrame.
    race_files = glob.glob(os.path.join(DATA_DIR, "*_R_results.json"))
    dataset = []

    for race_file in race_files:
        # Skip empty race files.
        if os.stat(race_file).st_size == 0:
            print(f"Skipping empty race file: {race_file}")
            continue

        # Extract year and event id from the file name.
        base = os.path.basename(race_file).replace("_R_results.json", "")
        year, *event_parts = base.split("_")
        event_id = "_".join(event_parts)

        # Construct the corresponding qualifying file path.
        quali_file = os.path.join(DATA_DIR, f"{year}_{event_id}_Q_results.json")
        if not os.path.exists(quali_file) or os.stat(quali_file).st_size == 0:
            print(f"Missing/empty quali file: {quali_file}")
            continue

        try:
            # Load race and qualifying data.
            race_df = pd.read_json(race_file)
            quali_df = pd.read_json(quali_file)
        except Exception as e:
            print(f"Read error: {race_file} / {quali_file}: {e}")
            continue

        # Skip if race data is empty.
        if race_df.empty:
            print(f"Empty race data: {race_file}")
            continue

        # Normalize column names to lowercase.
        race_df.columns = [col.lower() for col in race_df.columns]
        quali_df.columns = [col.lower() for col in quali_df.columns]

        # Define required columns for race and qualifying data.
        required_race_cols = ['drivernumber', 'broadcastname', 'teamname',
                              'gridposition', 'position', 'points', 'status']
        required_quali_cols = ['drivernumber', 'position', 'q1', 'q2', 'q3']

        missing = set(required_race_cols) - set(race_df.columns)
        if missing:
            print(f"Skipping {race_file}, missing cols: {missing}")
            continue

        # Keep only required columns and rename qualifying position column.
        race_df = race_df[required_race_cols]
        quali_df = quali_df[required_quali_cols].rename(columns={'position': 'qualiposition'})

        # Convert qualifying times to milliseconds.
        for q in ['q1', 'q2', 'q3']:
            quali_df[q] = quali_df[q].apply(time_to_ms)

        # Calculate the best qualifying time from q1, q2, and q3.
        quali_df['best_quali_time'] = quali_df[['q1', 'q2', 'q3']].min(axis=1)

        # Merge race and qualifying data on driver number.
        df = pd.merge(race_df, quali_df, on='drivernumber', how='left')
        # Create a binary flag for drivers who did not finish (DNF).
        df['status_dnf'] = df['status'].apply(lambda x: 1 if str(x).lower() in ['retired', 'dnf', 'accident', 'lapped'] else 0)
        # Convert team names to categorical numerical codes.
        df['teamname'] = df['teamname'].astype('category').cat.codes

        try:
            # Convert race positions to integer and filter out DNFs.
            df['position'] = df['position'].astype(int)
            df = df[df['status_dnf'] == 0]
        except Exception as e:
            print(f"Conversion error: {e}")
            continue

        # Create a new feature: the difference between grid and qualifying positions.
        df['grid_vs_qualipos'] = df['gridposition'] - df['qualiposition']

        # Attempt to load weather data if available.
        weather_path = os.path.join(DATA_DIR, f"{year}_{event_id}_R_weather.json")
        if os.path.exists(weather_path):
            try:
                with open(weather_path, 'r') as f:
                    weather = json.load(f)
                # Map weather data to new DataFrame columns.
                for key in ['AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed']:
                    df[key.lower()] = weather.get(key, None)
            except Exception as e:
                print(f"Failed weather read: {weather_path}: {e}")
        else:
            print(f"No weather file: {weather_path}")

        # Add the processed DataFrame to the dataset list.
        dataset.append(df)

    # Concatenate all DataFrames into one. Return an empty DataFrame if no data was processed.
    return pd.concat(dataset, ignore_index=True) if dataset else pd.DataFrame()


# =============================================================================
# Data Merging and Preprocessing
# =============================================================================
df = merge_features()
if not df.empty:
    # Drop rows missing essential information.
    df = df.dropna(subset=essential)
    # Fill remaining missing values with 0 and infer data types.
    df = df.fillna(0).infer_objects(copy=False)
    # Create a new target variable 'position_class' to group positions.
    df['position_class'] = df['position'].apply(lambda x: "Top 3" if x <= 3 else "Midfield" if x <= 10 else "Backmarker")


    # =============================================================================
    # Model Training and Evaluation
    # =============================================================================
    X = df[features].values
    y = df['position_class'].values

    # Split the dataset into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest classifier with specified hyperparameters.
    model = RandomForest(n_estimators=50, max_depth=10, min_samples_split=5)
    # Train the model.
    model.fit(X_train, y_train)
    # Make predictions on the test set.
    preds = model.predict(X_test)

    # Output the classification report and confusion matrix.
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
