import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    try:
        if hasattr(model, 'predict_proba'):
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    except AttributeError:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    colors = ['green', 'red']
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=colors, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlGn_r, marker='o')
    plt.xlabel('Feature 1')  # Update with the correct feature names or indices
    plt.ylabel('Feature 2')  # Update with the correct feature names or indices
    plt.title(title)
    plt.show()

# Load the dataset
data = pd.read_csv('C:/Users/pinky/Desktop/cleaned_fraud_data.csv')

# Display a summary of the dataframe
print(data.head(), end='\n\n')
print(data.info())
data.hist(bins=70, figsize=(30, 30))
print(data.describe())
fraud = data[data["fraud"] == 1]["fraud"].count()
n_fraud = data[data["fraud"] == 0]["fraud"].count()
print("Number of fraudulent transactions:", fraud)
print("Number of non-fraudulent transactions:", n_fraud)
print("Fraud percentage:", fraud / (fraud + n_fraud) * 100, "%")

# Data Resampling using SMOTE
X = data.drop('fraud', axis=1).values
y = data['fraud'].values
smote_sampler = SMOTE(random_state=39)
X_resampled, y_resampled = smote_sampler.fit_resample(X, y)
resampled_df = pd.DataFrame(X_resampled, columns=["distance_from_home", "distance_from_last_transaction",
                   "ratio_to_median_purchase_price", "repeat_retailer", "used_chip",
                   "used_pin_number", "online_order"])
resampled_df["fraud"] = y_resampled

print(resampled_df.shape)
print(resampled_df.info())
print(resampled_df.describe())

# Correlation Analysis
corr = resampled_df.corr()
spearman_corr = resampled_df.corr(method='spearman')

fig, ax = plt.subplots(nrows=2, figsize=(10, 15))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt=".2f", cmap="Blues", ax=ax[0])
ax[0].set_title('Pearson Correlation')
sns.heatmap(spearman_corr, xticklabels=spearman_corr.columns, yticklabels=spearman_corr.columns, annot=True, fmt=".2f", cmap="Blues", ax=ax[1])
ax[1].set_title('Spearman Correlation')
plt.tight_layout()
plt.show()

# Data Splitting
features = ["distance_from_home", "distance_from_last_transaction",
                   "ratio_to_median_purchase_price", "repeat_retailer", "used_chip",
                   "used_pin_number", "online_order"]
X_train, X_test, y_train, y_test = train_test_split(resampled_df[features], resampled_df.fraud, test_size=0.2, random_state=42, stratify=resampled_df.fraud)

X_train.shape, X_test.shape

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes Model
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_scaled, y_train)
naive_bayes_predictions = naive_bayes_model.predict(X_test_scaled)

print("\nNaive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, naive_bayes_predictions))
print("AUC:", roc_auc_score(y_test, naive_bayes_predictions))
print(classification_report(y_test, naive_bayes_predictions))

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_predictions = dt_model.predict(X_test_scaled)

print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("AUC:", roc_auc_score(y_test, dt_predictions))
print(classification_report(y_test, dt_predictions))

# K-Nearest Neighbors Model
knn_model = KNeighborsClassifier(n_neighbors=31)
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)

print("\nKNN Results:")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("AUC:", roc_auc_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions))

# Logistic Regression Model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("AUC:", roc_auc_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("AUC:", roc_auc_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

# Feature Importance Analysis for Decision Tree and Random Forest
feature_importances_dt = dt_model.feature_importances_
feature_importances_rf = rf_model.feature_importances_
features = X_train.columns

importance_df_dt = pd.DataFrame({'Feature': features, 'Importance': feature_importances_dt})
importance_df_dt = importance_df_dt.sort_values(by='Importance', ascending=False)

importance_df_rf = pd.DataFrame({'Feature': features, 'Importance': feature_importances_rf})
importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False)

print("Decision Tree Feature Importance:")
print(importance_df_dt)

print("\nRandom Forest Feature Importance:")
print(importance_df_rf)

# Feature Selection
X_train_selected = X_train_scaled[:, [2, 6]]  # As tags das features aqui
X_test_selected = X_test_scaled[:, [2, 6]]

# Train again for the selected features
naive_bayes_model.fit(X_train_selected, y_train)
knn_model.fit(X_train_selected, y_train)
dt_model.fit(X_train_selected, y_train)

naive_bayes_predictions = naive_bayes_model.predict(X_test_selected)

print("\nNaive Bayes Results (Selected Features):")
print("Accuracy:", accuracy_score(y_test, naive_bayes_predictions))
print("AUC:", roc_auc_score(y_test, naive_bayes_predictions))
print(classification_report(y_test, naive_bayes_predictions))

dt_predictions = dt_model.predict(X_test_selected)

print("\nDecision Tree Results (Selected Features):")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("AUC:", roc_auc_score(y_test, dt_predictions))
print(classification_report(y_test, dt_predictions))

knn_predictions = knn_model.predict(X_test_selected)

print("\nKNN Results (Selected Features):")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("AUC:", roc_auc_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions))

# Decision Boundary Plotting for Naive Bayes, Decision Tree, KNN, Logistic Regression, and Random Forest
plot_decision_boundary(naive_bayes_model, X_test_selected, y_test, 'Naive Bayes Decision Boundary (Selected Features)')
plot_decision_boundary(dt_model, X_test_selected, y_test, 'Decision Tree Decision Boundary (Selected Features)')
plot_decision_boundary(knn_model, X_test_selected, y_test, 'KNN Decision Boundary (Selected Features)')