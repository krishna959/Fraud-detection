# FAST CREDIT RISK ML PROJECT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

#LOAD DATASET
data = pd.read_csv("creditcard.csv")
print("Dataset Loaded")

#FEATURES & TARGET
X = data.drop("Class", axis=1)
y = data["Class"]

print("Original Class Distribution:")
print(y.value_counts())

#SCALE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#BALANCE DATA (SMOTE)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

print("After SMOTE:")
print(pd.Series(y_res).value_counts())

#REDUCE DATA SIZE (FOR SPEED)
X_res = X_res[:50000]   # use 50k samples
y_res = y_res[:50000]

#PCA FEATURE SELECTION
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_res)

print("Reduced Feature Shape:", X_pca.shape)

#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_res, test_size=0.2, random_state=42)

# FAST MODELS
# Random Forest
rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, subsample=0.8)
xgb.fit(X_train, y_train)

# Hybrid Ensemble
hybrid = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft'
)
hybrid.fit(X_train, y_train)

#EVALUATION FUNCTION
def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n{name} Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# Evaluate
evaluate(rf, "Random Forest")
evaluate(xgb, "XGBoost")
evaluate(hybrid, "Proposed Hybrid Model")

y_prob = hybrid.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="Hybrid AUC = %0.2f" % roc_auc)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Hybrid Model")
plt.legend()
plt.show()
