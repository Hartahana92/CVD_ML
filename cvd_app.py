import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.title("Предсказательная модель оценки развития сердечно-сосудистых заболеваний")
st.write("Вероятность развития ССЗ")
data = st.file_uploader("Загрузите файл")
classifier_name = st.sidebar.selectbox("Select classifier", ("KNN", "SVM", "Random Forest"))


def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K

    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


params = add_parameter_ui(classifier_name)


def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                     random_state=1234)
    return clf


clf = get_classifier(classifier_name, params)

# classification
if data is not None:
    data = pd.read_excel(data)
    y = data['Group'].values
    del data["Group"]
    X = data.values
    X[np.isnan(X)] = np.median(X[~np.isnan(X)])
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    X = preprocessing.normalize(X, norm='l2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    undersample = RandomOverSampler()
    X_train, y_train = undersample.fit_resample(X_train, y_train)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)