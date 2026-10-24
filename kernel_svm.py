import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo 
  
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
X = breast_cancer_wisconsin_original.data.features 
y = breast_cancer_wisconsin_original.data.targets 

print(breast_cancer_wisconsin_original.metadata) 
  
print(breast_cancer_wisconsin_original.variables) 

dataset = pd.concat([X, y], axis=1)
print("Before cleaning:", dataset.shape)

dataset = dataset.dropna()
print("After cleaning:", dataset.shape)

dataset.to_csv('breast_cancer_clean.csv', index=False)
dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=29, stratify=y
)
print(X_train.shape, X_test.shape)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf', C=1.0, random_state=29)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {round(accuracy,4)}")

cm = confusion_matrix(y_test, y_pred)
cm




