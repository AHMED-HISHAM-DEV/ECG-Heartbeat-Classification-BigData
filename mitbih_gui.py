import joblib
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)
from sklearn.preprocessing import label_binarize

# ── 1. Load ───────────────────────────────────────────────────
df = pd.read_csv("mitbih_train.csv", header=None)
print(f"Loaded: {df.shape}")

# ── 2. Split X / y ────────────────────────────────────────────
X = df.iloc[:, :-1].values.astype(float)
le = LabelEncoder()
y = le.fit_transform(df.iloc[:, -1].values)
print(f"Classes: {np.unique(y)} | Distribution: {Counter(y)}")

# ── 3. Fill missing ───────────────────────────────────────────
X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
print(f"Missing values filled.")

# ── 4. Remove zero-variance features ─────────────────────────
sel = VarianceThreshold(0.0)
X = sel.fit_transform(X)
print(f"Features after selection: {X.shape[1]}")

# ── 5. Normalize ──────────────────────────────────────────────
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# ── 6. Train / Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ── 7. Train Random Forest ────────────────────────────────────
print("\nTraining... please wait...")
# تم إضافة class_weight='balanced' لحل مشكلة الـ Imbalance بناءً على شروط المشروع
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── 8. Evaluate ───────────────────────────────────────────────
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))

print(f"\nTrain Accuracy : {train_acc:.2%}")
print(f"Test  Accuracy : {test_acc:.2%}")

if train_acc - test_acc > 0.05:
    print("WARNING: Overfitting detected!")
elif test_acc < 0.75:
    print("WARNING: Underfitting detected!")
else:
    print("Model is well fitted.")

test_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, test_pred,
      target_names=["Normal","Type1","Type2","Type3","Type4"]))

# ── 9. Final evaluation on mitbih_test.csv ────────────────────
test_df   = pd.read_csv("mitbih_test.csv", header=None)
X_final   = test_df.iloc[:, :-1].values.astype(float)
y_final   = le.transform(test_df.iloc[:, -1].values)
X_final   = sel.transform(X_final)
X_final   = scaler.transform(X_final)
final_acc = accuracy_score(y_final, model.predict(X_final))
print(f"\nFinal Test Accuracy (mitbih_test.csv): {final_acc:.2%}")

# ── 10. Save ──────────────────────────────────────────────────
joblib.dump(model,  "ecg_model.pkl")
joblib.dump(scaler, "ecg_scaler.pkl")
joblib.dump(sel,    "ecg_selector.pkl")
joblib.dump(le,     "ecg_label_encoder.pkl")
print("\nModel saved successfully!")

# ── 11. Plotting 1: Exploratory Data Analysis (EDA) ───────────
print("\n[1/2] Generating EDA Plots... Close this window to see Model Evaluation.")

fig_eda = plt.figure(figsize=(14, 8))
gs_eda = fig_eda.add_gridspec(2, 2, height_ratios=[1.2, 0.8])

# 1. Class Distribution (الـ Countplot اللي بيوضح عدم التوازن)
ax_eda1 = fig_eda.add_subplot(gs_eda[0, 0])
sns.countplot(x=y, ax=ax_eda1, palette="viridis")
ax_eda1.set_title("Dataset Class Distribution (Imbalance Check)", fontsize=11, fontweight='bold')
ax_eda1.set_xticklabels(["Normal","Type1","Type2","Type3","Type4"])
ax_eda1.set_xlabel("Heartbeat Category")

# 2. Waveform Visualization (شكل نبضة قلب حقيقية بعد الـ Normalization)
ax_eda2 = fig_eda.add_subplot(gs_eda[0, 1])
ax_eda2.plot(X[0], color="#ef4444", linewidth=1.5)
ax_eda2.set_title("Sample Normalized ECG Signal Waveform", fontsize=11, fontweight='bold')
ax_eda2.set_xlabel("Time Step / Feature Index")
ax_eda2.set_ylabel("Amplitude")
ax_eda2.grid(True, alpha=0.3)

# 3. Feature Correlation Heatmap (مصفوفة الارتباط لأول 30 ميزة)
ax_eda3 = fig_eda.add_subplot(gs_eda[1, :])
sample_corr_df = pd.DataFrame(X[:, :30])
sns.heatmap(sample_corr_df.corr(), cmap="coolwarm", ax=ax_eda3, cbar=True, xticklabels=2, yticklabels=2)
ax_eda3.set_title("Feature Correlation Heatmap (First 30 Features)", fontsize=11, fontweight='bold')

fig_eda.tight_layout()
plt.show()  # الكود هيقف هنا لحد ما تقفل النافذة دي

# ── 12. Plotting 2: Model Performance Evaluation ──────────────
print("\n[2/2] Generating Model Evaluation Plots...")

fig_eval = plt.figure(figsize=(14, 8))
gs_eval = fig_eval.add_gridspec(2, 2, height_ratios=[1.3, 0.7])

# 1. Confusion Matrix
ax_ev1 = fig_eval.add_subplot(gs_eval[0, 0])
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_ev1, cbar=False)
ax_ev1.set_title("Confusion Matrix", fontsize=11, fontweight='bold')
ax_ev1.set_xticklabels(["Normal","Type1","Type2","Type3","Type4"], fontsize=9)
ax_ev1.set_yticklabels(["Normal","Type1","Type2","Type3","Type4"], fontsize=9)
ax_ev1.set_xlabel("Predicted Labels")
ax_ev1.set_ylabel("True Labels")

# 2. Multi-Class ROC Curve
ax_ev2 = fig_eval.add_subplot(gs_eval[0, 1])
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = model.predict_proba(X_test)

for i in range(y_test_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax_ev2.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

ax_ev2.plot([0, 1], [0, 1], linestyle='--', color="gray")
ax_ev2.set_title("Multi-Class ROC Curve", fontsize=11, fontweight='bold')
ax_ev2.set_xlabel("False Positive Rate")
ax_ev2.set_ylabel("True Positive Rate")
ax_ev2.legend(fontsize=8, loc='lower right')
ax_ev2.grid(True, alpha=0.3)

# 3. Feature Importance
ax_ev3 = fig_eval.add_subplot(gs_eval[1, :])
importance = pd.Series(model.feature_importances_).sort_values(ascending=False).head(15)
importance.plot(kind='bar', ax=ax_ev3, color='#3b82f6', width=0.6)
ax_ev3.set_title("Top 15 Important ECG Features Selected by Random Forest", fontsize=11, fontweight='bold')
ax_ev3.set_xlabel("Feature Index")
ax_ev3.set_ylabel("Importance Score")
ax_ev3.tick_params(axis='both', labelsize=8)

fig_eval.tight_layout()
plt.show()  # النافذة الثانية والأخيرة

print("\nPipeline executed perfectly. All files ready for api.py deployment!")