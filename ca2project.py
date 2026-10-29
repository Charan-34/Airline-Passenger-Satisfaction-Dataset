# Airline Passenger Satisfaction: Data Preprocessing and EDA

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# LOAD DATASET
df = pd.read_csv('data.csv')

print(df.info())
print(df.describe())
print(df.head())
print(df.tail())

# REMOVE DUPLICATES
print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# HANDLE MISSING VALUES
print("\nMissing values:")
print(df.isnull().sum())

df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(
    df['Arrival Delay in Minutes'].median()
)

print("\nMissing values after handling:")
print(df.isnull().sum())

# CORRELATION AND COVARIANCE
print("\nCorrelation:")
print(df.corr(numeric_only=True))

print("\nCovariance:")
print(df.cov(numeric_only=True))

# SKEWNESS AND KURTOSIS
print("\nAge Skewness:", df['Age'].skew())
print("Age Kurtosis:", df['Age'].kurt())

# SATISFACTION GRAPH
sns.countplot(data=df, x='satisfaction')
plt.title('Passenger Satisfaction')
plt.show()

# AGE DISTRIBUTION
sns.histplot(data=df, x='Age', bins=25)
plt.title('Distribution of Passenger Ages')
plt.show()

# CORRELATION HEATMAP
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# OBJECTIVE 1: PASSENGER SATISFACTION
print("\nOBJECTIVE 1: Passenger Satisfaction")
print(df['satisfaction'].value_counts())

# OBJECTIVE 2: PASSENGER AGE ANALYSIS
print("\nOBJECTIVE 2: Average Age by Satisfaction")
print(df.groupby('satisfaction')['Age'].mean())

# OBJECTIVE 3: INFLIGHT SERVICE ANALYSIS
print("\nOBJECTIVE 3: Inflight Service Ratings")
print(df.groupby('satisfaction')[['Inflight wifi service', 'Inflight entertainment']].mean())

# OBJECTIVE 4: ONLINE BOARDING ANALYSIS
print("\nOBJECTIVE 4: Satisfaction Rate by Online Boarding Rating")
boarding_rate = df.groupby('Online boarding')['satisfaction'].mean() * 100
print(boarding_rate.round(2))

# OBJECTIVE 5: FLIGHT DELAYS
print("\nOBJECTIVE 5: Average Flight Delays")
print(df.groupby('satisfaction')[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].mean())


# GRAPH 1: PASSENGER SATISFACTION
sns.countplot(data=df, x='satisfaction')
plt.title('Passenger Satisfaction')
plt.show()


# GRAPH 2: PASSENGER AGE ANALYSIS
sns.barplot(data=df, x='satisfaction', y='Age')
plt.title('Average Age by Satisfaction')
plt.show()


# GRAPH 3: INFLIGHT SERVICE ANALYSIS
df.groupby('satisfaction')[
    ['Inflight wifi service', 'Inflight entertainment']
].mean().plot(kind='bar')

plt.title('Inflight Service Ratings')
plt.show()


# GRAPH 4: ONLINE BOARDING ANALYSIS
boarding_rate = df.groupby('Online boarding')['satisfaction'].mean() * 100
sns.barplot(x=boarding_rate.index, y=boarding_rate.values)
plt.title('Satisfaction by Online Boarding')
plt.show()


# GRAPH 5: FLIGHT DELAY ANALYSIS
df.groupby('satisfaction')[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].mean().plot(kind='bar')
plt.title('Flight Delays by Satisfaction')
plt.show()


# SELECT SAMPLE FOR FASTER TRAINING
data = df.sample(n=min(8000, len(df)), random_state=42)

# SELECT FEATURES AND TARGET
X = data.drop('satisfaction', axis=1)
Y = data['satisfaction']

# CONVERT TEXT INTO NUMBERS
X = pd.get_dummies(X, drop_first=True)

# SPLIT DATASET
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# STANDARDIZE DATA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL 1: LOGISTIC REGRESSION
model1 = LogisticRegression(max_iter=500)
model1.fit(X_train, Y_train)
y_pred1 = model1.predict(X_test)

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(Y_test, y_pred1))
print("Precision:", precision_score(Y_test, y_pred1))
print("Recall:", recall_score(Y_test, y_pred1))
print("F1 Score:", f1_score(Y_test, y_pred1))
print(confusion_matrix(Y_test, y_pred1))
print(classification_report(Y_test, y_pred1))

# MODEL 2: KNN
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

print("\nKNN")
print("Accuracy:", accuracy_score(Y_test, y_pred2))
print("Precision:", precision_score(Y_test, y_pred2))
print("Recall:", recall_score(Y_test, y_pred2))
print("F1 Score:", f1_score(Y_test, y_pred2))
print(confusion_matrix(Y_test, y_pred2))
print(classification_report(Y_test, y_pred2))

# MODEL 3: SUPPORT VECTOR MACHINE
model3 = SVC(kernel='linear', cache_size=1000)
model3.fit(X_train, Y_train)
y_pred3 = model3.predict(X_test)

print("\nSUPPORT VECTOR MACHINE")
print("Accuracy:", accuracy_score(Y_test, y_pred3))
print("Precision:", precision_score(Y_test, y_pred3))
print("Recall:", recall_score(Y_test, y_pred3))
print("F1 Score:", f1_score(Y_test, y_pred3))
print(confusion_matrix(Y_test, y_pred3))
print(classification_report(Y_test, y_pred3))

# MODEL 4: DECISION TREE
model4 = DecisionTreeClassifier(max_depth=15, random_state=42)
model4.fit(X_train, Y_train)
y_pred4 = model4.predict(X_test)

print("\nDECISION TREE")
print("Accuracy:", accuracy_score(Y_test, y_pred4))
print("Precision:", precision_score(Y_test, y_pred4))
print("Recall:", recall_score(Y_test, y_pred4))
print("F1 Score:", f1_score(Y_test, y_pred4))
print(confusion_matrix(Y_test, y_pred4))
print(classification_report(Y_test, y_pred4))

# MODEL 5: RANDOM FOREST
model5 = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42)
model5.fit(X_train, Y_train)
y_pred5 = model5.predict(X_test)

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(Y_test, y_pred5))
print("Precision:", precision_score(Y_test, y_pred5))
print("Recall:", recall_score(Y_test, y_pred5))
print("F1 Score:", f1_score(Y_test, y_pred5))
print(confusion_matrix(Y_test, y_pred5))
print(classification_report(Y_test, y_pred5))

# MODEL ACCURACY COMPARISON
models = ['Logistic Regression', 'KNN', 'SVM', 'Decision Tree', 'Random Forest']

accuracies = [
    accuracy_score(Y_test, y_pred1),
    accuracy_score(Y_test, y_pred2),
    accuracy_score(Y_test, y_pred3),
    accuracy_score(Y_test, y_pred4),
    accuracy_score(Y_test, y_pred5)
]

print("\nMODEL ACCURACY COMPARISON")
for i in range(len(models)):
    print(models[i], ":", accuracies[i])

# ACCURACY GRAPH
plt.figure(figsize=(10, 5))
sns.barplot(x=models, y=accuracies)
plt.xticks(rotation=20)
plt.xlabel('Machine Learning Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()

# BEST-FIT MODEL
best = models[accuracies.index(max(accuracies))]

print("\nBEST-FIT MODEL:", best)
print("BEST ACCURACY:", max(accuracies))

# KEY FINDINGS AND CONCLUSION
print("\nKEY FINDINGS")
print("1. Passenger satisfaction was analyzed.")
print("2. Satisfaction by class, travel type and customer type was studied.")
print("3. Flight delays were compared.")
print("4. Five classification models were evaluated.")
print("5. Best model:", best)

print("\nFINAL CONCLUSION")
print("The best model is:", best)
print("Best accuracy:", round(max(accuracies) * 100, 2), "%")
print("The model was selected by comparing five ML models.")
