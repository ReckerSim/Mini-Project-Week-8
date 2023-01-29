import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report

def find_best_k(X_train, Y_train, X_test, Y_test, min_k=1, max_k=100):
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(Y_test, preds, average='micro')
        if f1 > best_score:
            best_k = k
            best_score = f1
    
    print("Best Value for k: {}".format(best_k))
    print("F1-Score: {}".format(best_score))

def print_metrics(labels, preds):
    print("Precision Score: ", precision_score(labels, preds, average='micro'))
    print("Recall Score: ", recall_score(labels, preds, average='micro'))
    print("Accuracy Score: ", accuracy_score(labels, preds))
    print("F1 Score: ", f1_score(labels, preds, average='micro'))

music_three_sec_data = pd.read_csv('Data/features_3_sec.csv')
print(music_three_sec_data.head(10))
print(music_three_sec_data.info())
print(music_three_sec_data.describe)

X = music_three_sec_data.iloc[:,1:-1]
Y = music_three_sec_data.label


scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

scaled_df = pd.DataFrame(X, columns=X.columns)
scaled_df.head()

X_train, X_test, Y_train, Y_test = train_test_split(scaled_df, Y, stratify=Y, test_size=0.2, random_state=42)

best_k = find_best_k(X_train, Y_train, X_test, Y_test, min_k=1, max_k=100)

knn = KNeighborsClassifier(n_neighbors=int(best_k))
knn.fit(X_train, Y_train)
preds = knn.predict(X_test)

print_metrics(Y_test, preds)


