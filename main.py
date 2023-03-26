import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

train_data, test_data = tf.keras.datasets.mnist.load_data()
X_tr,y_tr = train_data
X_test,y_test = test_data
print(f'{X_tr.shape}')
print(f'{y_tr.shape}')
print(f'{X_test.shape}')
print(f'{y_test.shape}')
fig, axs = plt.subplots(1, 9, figsize=(15, 3))
for index, ax in enumerate(axs):
  ax.imshow(X_tr[index],cmap='gray')
plt.show()

CATEGORIES = 10
# Normalizace <0,255> -> <0,1>
X_tr = X_tr/255.0
X_test = X_test/255.0
# Prevod matic 28*28 na vektory
X_tr_vec = X_tr.reshape(-1,28*28)
X_test_vec = X_test.reshape(-1,28*28)
print(X_tr_vec.shape)
print(X_test_vec.shape)

N_tr = 30000 # velikost treninkovych dat
X_train = X_tr_vec[:N_tr]
y_train = y_tr[:N_tr]
print(f"Velikost X trenikovych dat:{X_train.shape}")
print(f"Velikost y trenikovych dat:{y_train.shape}")

from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(random_state=0,solver="newton-cg",max_iter=200, verbose=1)
model_lr.fit(X_train,y_train)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay

y_hat_train = model_lr.predict(X_train)
y_hat_test = model_lr.predict(X_test_vec)

cm = confusion_matrix(y_test,y_hat_test)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [ i for i in range(10)])# vykresleni
disp.plot()
print(f"Report klasifikace na trenovacich datech LogREG :\n{classification_report(y_train, y_hat_train)}")
print(f"Report klasifikace na testovacich datech LogREG :\n{classification_report(y_test, y_hat_test)}")



from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train,y_train)
y_hat_train = model_knn.predict(X_train)
y_hat_test = model_knn.predict(X_test_vec)

cm = confusion_matrix(y_test,y_hat_test)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [ i for i in range(10)])# vykresleni
disp.plot()
print(f"Report klasifikace na trenovacich datech KNN :\n{classification_report(y_train, y_hat_train)}")
print(f"Report klasifikace na testovacich datech KNN :\n{classification_report(y_test, y_hat_test)}")



from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0)
model_rf.fit(X_train,y_train)
y_hat_train = model_rf.predict(X_train)
y_hat_test = model_rf.predict(X_test_vec)

cm = confusion_matrix(y_test,y_hat_test)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [ i for i in range(10)])# vykresleni
disp.plot()
print(f"Report klasifikace na trenovacich datech Random forest :\n{classification_report(y_train, y_hat_train)}")
print(f"Report klasifikace na testovacich datech Random forest :\n{classification_report(y_test, y_hat_test)}")

from sklearn.ensemble import GradientBoostingClassifier
model_btre = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model_btre.fit(X_train,y_train)
y_hat_train = model_btre.predict(X_train)
y_hat_test = model_btre.predict(X_test_vec)

cm = confusion_matrix(y_test,y_hat_test)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [ i for i in range(10)])# vykresleni
disp.plot()
print(f"Report klasifikace na trenovacich datech Random forest :\n{classification_report(y_train, y_hat_train)}")
print(f"Report klasifikace na testovacich datech Random forest :\n{classification_report(y_test, y_hat_test)}")