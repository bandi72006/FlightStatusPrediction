from sklearn.metrics import classification_report, confusion_matrix

def evaluateModel(model, XTest, yTest):
    preds = model.predict(XTest)
    print("Classification Report:")
    print(classification_report(yTest, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(yTest, preds))
