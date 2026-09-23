import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# UCI Adult Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ["age", "workclass", "fnlwgt", "education", "education-num", 
           "marital-status", "occupation", "relationship", "race", 
           "sex", "capital-gain", "capital-loss", "hours-per-week", 
           "native-country", "income"]

# Data preprocessing
df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
df = df.dropna().reset_index(drop=True)
print('len(df): ', len(df))

df = df[["age", "education-num", "hours-per-week", "sex", "income"]]
df["sex"] = LabelEncoder().fit_transform(df["sex"])
df["income"] = df["income"].map({">50K": 1, "<=50K": 0})

original_df = df.copy()

# Functions of Mondrian K-Anonymization
quasi_identifiers = ["age", "education-num", "hours-per-week"]

def generalize(series):
    return f"[{series.min()} - {series.max()}]"

def mondrian_partition(df, k):
    partitions = [df]
    finished = []
    while partitions:
        partition = partitions.pop()
        if len(partition) < 2 * k:
            finished.append(partition)
            continue
        max_width = -1
        split_col = None
        for col in quasi_identifiers:
            width = partition[col].max() - partition[col].min()
            if width > max_width:
                max_width = width
                split_col = col
        median = partition[split_col].median()
        left = partition[partition[split_col] <= median]
        right = partition[partition[split_col] > median]
        if len(left) < k or len(right) < k:
            finished.append(partition)
        else:
            partitions.append(left)
            partitions.append(right)
    return finished

def generalize_partition(partitions):
    result = []
    for group in partitions:
        gen_group = group.copy()
        for col in quasi_identifiers:
            gen_group[col] = generalize(group[col])
        result.append(gen_group)
    return pd.concat(result, ignore_index=True)

# Apply K-anonymity
partitioned_2 = mondrian_partition(df, 2)
anonymized_df_2 = generalize_partition(partitioned_2)

partitioned_5 = mondrian_partition(df, 5)
anonymized_df_5 = generalize_partition(partitioned_5)

partitioned_10 = mondrian_partition(df, 10)
anonymized_df_10 = generalize_partition(partitioned_10)

# Preprocessing data for SVM
for col in quasi_identifiers:
    anonymized_df_2[col] = anonymized_df_2[col].apply(
        lambda x: np.mean([float(n) for n in x.strip("[]").split(" - ")])
    )
    anonymized_df_5[col] = anonymized_df_5[col].apply(
        lambda x: np.mean([float(n) for n in x.strip("[]").split(" - ")])
    )
    anonymized_df_10[col] = anonymized_df_10[col].apply(
        lambda x: np.mean([float(n) for n in x.strip("[]").split(" - ")])
    )

anonymized_df_2["sex"] = original_df["sex"]
anonymized_df_2["income"] = original_df["income"]

anonymized_df_5["sex"] = original_df["sex"]
anonymized_df_5["income"] = original_df["income"]

anonymized_df_10["sex"] = original_df["sex"]
anonymized_df_10["income"] = original_df["income"]

# SVM Experiment
# def run_svm_evaluation(df, title):
#     X = df.drop("income", axis=1)
#     y = df["income"]

#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42
#     )

#     clf = SVC(kernel='rbf', probability=True)
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     y_proba = clf.predict_proba(X_test)[:, 1]

#     accuracy = accuracy_score(y_test, y_pred)
#     misclassification_rate = 1 - accuracy
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_proba)

#     print(f"Result for: {title}")
#     print(f"Misclassification Rate: {misclassification_rate:.4f}")
#     print(f"Accuracy:               {accuracy:.4f}")
#     print(f"Precision:              {precision:.4f}")
#     print(f"Recall:                 {recall:.4f}")
#     print(f"AUC:                    {auc:.4f}")

#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:")
#     print(cm)

#     TN, FP, FN, TP = cm.ravel()
#     print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

# run_svm_evaluation(original_df, "Original Data")
# run_svm_evaluation(anonymized_df_2, f"K-Anonymized Data (k=2)")
# run_svm_evaluation(anonymized_df_5, f"K-Anonymized Data (k=5)")
# run_svm_evaluation(anonymized_df_10, f"K-Anonymized Data (k=10)")

# Random Forest Experiment
def run_random_forest_evaluation(df, title):
    X = df.drop("income", axis=1)
    y = df["income"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Use RandomForestClassifier instead of SVM
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    misclassification_rate = 1 - accuracy
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Result for: {title}")
    print(f"Misclassification Rate: {misclassification_rate:.4f}")
    print(f"Accuracy:               {accuracy:.4f}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall:                 {recall:.4f}")
    print(f"AUC:                    {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

run_random_forest_evaluation(original_df, "Original Data")
run_random_forest_evaluation(anonymized_df_2, f"K-Anonymized Data (k=2)")
run_random_forest_evaluation(anonymized_df_5, f"K-Anonymized Data (k=5)")
run_random_forest_evaluation(anonymized_df_10, f"K-Anonymized Data (k=10)")