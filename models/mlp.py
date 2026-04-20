from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def train_mlp(X_train, y_train, hidden_layer_sizes=(100,), activation='relu', solver='adam', save_path='saved_models/mlp.pkl'):
    model = MLPClassifier(random_state=42, hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver)
    model.fit(X_train, y_train)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
    return model

def evaluate_mlp(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return cm, report