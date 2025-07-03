import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb

# ===========================
# 1. Load Dataset
# ===========================
df = pd.read_csv(r"C:\Users\User\A VS CODE\ann_new\Churn_Modelling.csv")

X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

# ===========================
# 2. Apply Encoders
# ===========================
# LabelEncode Gender
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # Gender

# OneHotEncode Geography
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), [1])
], remainder='passthrough')

X = np.array(ct.fit_transform(X))

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# ✅ SAVE the encoders and scaler
joblib.dump(le, "label_encoder.pkl")
joblib.dump(ct, "column_transformer.pkl")
joblib.dump(sc, "scaler.pkl")

# ===========================
# 3. Split Dataset
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ===========================
# 4. Define & Train ML Models
# ===========================
models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'NaiveBayes': GaussianNB(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    joblib.dump(model, f"{name}_model.pkl")

# ===========================
# 5. Define, Train & Save ANN
# ===========================
ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(6, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # ✅ use sigmoid for binary classification
])

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

y_pred_ann = ann.predict(X_test)
y_pred_ann = (y_pred_ann > 0.5).astype(int).flatten()
acc_ann = accuracy_score(y_test, y_pred_ann)
print(f"ANN Accuracy: {acc_ann:.4f}")

ann.save("ann_model.h5")
