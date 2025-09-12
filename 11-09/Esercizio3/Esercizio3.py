import pandas as pd
from sklearn.model_selection import train_test_split

# carico il dataset
df = pd.read_csv("creditcard.csv") 

# # divido input e target
X = df.drop("Class", axis=1)
y = df["Class"]

#  split: test 15%, temp 85%
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# dal temp ricavo train 70% e val 15%
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

# print shape
print({"Train": {X_train.shape}}, {"Validation": {X_val.shape}}, {"test": {X_test.shape}})