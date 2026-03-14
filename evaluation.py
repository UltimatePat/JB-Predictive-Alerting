from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score,  f1_score



def evaluate(model, X_test, y_test, threshold):

    # Predict probability of label of test point
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Depending on threshold assign it to the anomolay or normal class
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    # Print results
    print(f"Accuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"AUROC Score: {auroc:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report")
    print(classification_report(y_test, y_pred, zero_division=0))