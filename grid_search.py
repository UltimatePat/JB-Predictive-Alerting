from sklearn.metrics import roc_auc_score, f1_score
import data_processing as dp
import model as md

def grid_search_on_W_H(threshold):

    # Values to perform grid search on 
    W_values = [2, 4, 12, 24, 48]
    H_values = [1, 2, 4]
    labels_path = "combined_windows.json"
    
    # Storing best found values to date
    best_auroc = 0
    best_params_auroc = {}
    best_model_info_auroc = ()
    best_f1 = 0
    best_params_f1 = {}
    best_model_info_f1 = ()
    
    # Loop through all combinations of W and H and save best found results
    for W in W_values:
        for H in H_values:
            
            dp.create_windowed_dataset(W, H, labels_path)
            model, X_test, y_test = md.train_xgboost()
            
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
            auroc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            print(f"W = {W}, H = {H}")
            print(f"AUROC: {auroc:.4f}")
            print(f"F1: {f1}\n")
            
            if auroc > best_auroc:
                best_auroc = auroc
                best_params_auroc = {'W': W, 'H': H}
                best_model_info_auroc = (model, X_test, y_test)

            if f1 > best_f1:
                best_f1 = f1
                best_params_f1 = {'W': W, 'H': H}
                best_model_info_f1 = (model, X_test, y_test)
            

    # Print and return results
    print(f"Best W and H for AUROC: W = {best_params_auroc['W']}, H = {best_params_auroc['H']}")
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Best W and H for F1: W = {best_params_f1['W']}, H = {best_params_f1['H']}")
    print(f"Best F1: {best_f1:.4f}")
    
    return best_auroc, best_params_auroc, best_model_info_auroc, best_f1, best_params_f1, best_model_info_f1
