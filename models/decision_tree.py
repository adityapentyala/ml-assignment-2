from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def train_decision_tree(X_train, y_train, max_depth=7, save_path='saved_models/decision_tree.pkl'):  
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    model.fit(X_train, y_train)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
    return model

def evaluate_decision_tree(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return cm, report
