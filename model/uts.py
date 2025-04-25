#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#%%
df = pd.read_csv('dataset_buys_comp.csv')
print(df.head())

#%%
X = df.drop('Buys_Computer', axis=1)
y = df['Buys_Computer']

#%%
# Inisialisasi LabelEncoder untuk setiap kolom kategorikal
le_age = LabelEncoder()
le_income = LabelEncoder()
le_student = LabelEncoder()
le_credit = LabelEncoder()

X['Age'] = le_age.fit_transform(X['Age'])
X['Income'] = le_income.fit_transform(X['Income'])
X['Student'] = le_student.fit_transform(X['Student'])
X['Credit_Rating'] = le_credit.fit_transform(X['Credit_Rating'])
print("\nData setelah Label Encoding:")
print(X.head())

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
model = GaussianNB()
model.fit(X_train, y_train)

#%%
y_pred = model.predict(X_test)

#%%
# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy:.2f}")

# Buat confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Buat classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)