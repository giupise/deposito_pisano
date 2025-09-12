import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# carico il dataset
df = pd.read_csv("creditcard.csv") 

# divido input e target
X = df.drop("Class", axis=1) 
y = df["Class"]

# divido i dati in: train (80%) e test (20%) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# addestro l'albero decisionale con class_weight balanced
dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree")
print(classification_report(y_test, y_pred_dt, digits=4))

# addestro con una Random Forest con class_weight balanced
rf = RandomForestClassifier(class_weight="balanced", random_state=42, max_depth=5, n_estimators=50)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest")
print(classification_report(y_test, y_pred_rf, digits=4))

# per generare nuovi esempi sintetici della classe minoritaria
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

dt_smote = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt_smote.fit(X_train_res, y_train_res)
y_pred_dt_smote = dt_smote.predict(X_test)

print("Decision Tree con SMOTE")
print(classification_report(y_test, y_pred_dt_smote, digits=4))

rf_smote = RandomForestClassifier(class_weight="balanced", random_state=42, max_depth=5, n_estimators=50)
rf_smote.fit(X_train_res, y_train_res)
y_pred_rf_smote = rf_smote.predict(X_test)

print("Random Forest con SMOTE")
print(classification_report(y_test, y_pred_rf_smote, digits=4))
