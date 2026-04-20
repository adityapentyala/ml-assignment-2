from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def train_svc(X_train, y_train, C=1.0, kernel='rbf', save_path='saved_models/svc.pkl'):
    model = SVC(random_state=42, C=C, kernel=kernel)
    model.fit(X_train, y_train)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
    return model

def evaluate_svc(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return cm, report
