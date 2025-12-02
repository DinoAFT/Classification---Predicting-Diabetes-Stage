import numpy as np
import pandas as pd
import time as time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

X_train_std = pd.read_csv("X_train_std.csv")
Y_train = pd.read_csv("Y_train.csv")
X_test_std = pd.read_csv("X_test_std.csv")
Y_test = pd.read_csv("Y_test.csv")

# Identify categorical columns (dtype == object)
categorical_cols = X_train_std.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

categorical_cols = X_train_std.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

# Apply one-hot encoding only to categorical columns
X_train_std_encoded = pd.get_dummies(X_train_std, columns=categorical_cols, drop_first=True, dtype=int)
X_test_std_encoded = pd.get_dummies(X_test_std, columns=categorical_cols, drop_first=True, dtype=int)

le = LabelEncoder()

# Fit on training labels and transform both train/test
Y_train_encoded = le.fit_transform(Y_train.values.ravel())
Y_test_encoded = le.transform(Y_test.values.ravel())

CLASS_LABELS = le.classes_

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """Plots a confusion matrix as a heatmap with class labels."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels, 
        yticklabels=labels,
        cbar=False,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.show()

def plot_multiclass_roc(y_test, y_pred_proba, labels):
    # Binarize the output labels for ROC calculation (required for OvR)
    # The classes argument ensures all potential classes are included.
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown'] # Use diverse colors
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
            label=f'ROC curve of {labels[i]} (AUC = {roc_auc[i]:.2f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - QDA', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def forward_selection_qda(X, y, cv, scoring='f1_weighted'):
    current_features = [] # Stores feature NAMES
    remaining_features = list(X.columns) # Start with all column NAMES
    
    # Track the best performing subset found so far
    best_score_overall = -np.inf
    best_features_overall = [] # Stores feature NAMES
    
    print(f"Starting Forward Selection for QDA (Criterion: {cv.n_splits}-fold CV {scoring})...")
    print("-" * 50)
    
    # 

    # Loop continues as long as there are features left to potentially add
    while remaining_features:
        best_feature_to_add = None # Stores the NAME of the feature
        best_score_this_step = -np.inf
        
        # Iterate over all remaining features to find the best one to add
        for feature_candidate in remaining_features:
            # Create a candidate feature set by adding the new feature name
            candidate_features = current_features + [feature_candidate]
            
            # Use DataFrame slicing with feature NAMES
            X_candidate = X[candidate_features]

            # Initialize and evaluate the QDA model using cross-validation
            # FIX: Added reg_param for numerical stability
            qda = QuadraticDiscriminantAnalysis(reg_param=1e-6) 
            
            # Pass the DataFrame to cross_val_score
            scores = cross_val_score(qda, X_candidate, y, cv=cv, scoring=scoring)
            mean_score = np.mean(scores)

            # Check if this candidate feature provides the best score in this step
            if mean_score > best_score_this_step:
                best_score_this_step = mean_score
                best_feature_to_add = feature_candidate
        
        # --- Stopping Criterion Check ---
        if best_score_this_step > best_score_overall:
            best_score_overall = best_score_this_step
            current_features.append(best_feature_to_add)
            remaining_features.remove(best_feature_to_add)
            best_features_overall = list(current_features)
            
            print(f"Step {len(current_features)}: Added feature '{best_feature_to_add}'. Current features: {current_features} | New CV {scoring}: {best_score_overall:.4f}")
        else:
            print("-" * 50)
            print(f"Stopping criterion met at {len(current_features)} features.")
            print(f"Best score achieved at previous step ({len(current_features)} features) was {best_score_overall:.4f}.")
            break
            
    return best_features_overall, best_score_overall

start_time = time.time()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
selected_feature_indices, final_score = forward_selection_qda(X_train_std_encoded, Y_train_encoded, cv=cv)

end_time = time.time()
# Get runtime of lasso cross-validation
runtime = end_time - start_time
print(f"The code block took: {runtime} seconds")

# Final Results and Model Training

print("\n--- Final Model Summary ---")
print(f"Optimized Feature Indices: {selected_feature_indices}")
print(f"Number of Selected Features: {len(selected_feature_indices)}")
print(f"Highest 5-fold CV Weighted F1 Score Achieved: {final_score:.6f}")

# Train the final QDA model on the selected features using the full training set
if selected_feature_indices:
    X_train_final = X_train_std_encoded.loc[:, selected_feature_indices]
    X_test_final = X_test_std_encoded.loc[:, selected_feature_indices]
    
    final_qda_model = QuadraticDiscriminantAnalysis(reg_param=1e-6)
    final_qda_model.fit(X_train_final, Y_train_encoded)
    
    # Predict class labels and probabilities on the held-out TEST set
    Y_pred = final_qda_model.predict(X_test_final)
    Y_pred_proba = final_qda_model.predict_proba(X_test_final)
    
    # Evaluation on Test Set
    test_accuracy = accuracy_score(Y_test_encoded, Y_pred)
    test_f1 = f1_score(Y_test_encoded, Y_pred, average='weighted')
    
    # Calculate ROC AUC for multi-class using 'one-vs-rest' (ovr) strategy
    try:
        test_roc_auc = roc_auc_score(Y_test_encoded, Y_pred_proba, multi_class='ovr', average='weighted')
    except ValueError as e:
        # This can happen if a class is missing in the test set due to small data size
        test_roc_auc = f"N/A (Error calculating ROC AUC: {e})"
        
    conf_matrix = confusion_matrix(Y_test_encoded, Y_pred)

    print("\n--------------------------------------------------")
    print("--- Test Set Evaluation (Performance on Unseen Data) ---")
    print("--------------------------------------------------")
    print(f"Test Accuracy: {test_accuracy:.6f}")
    print(f"Test Weighted F1 Score: {test_f1:.6f}")
    print(f"Test ROC AUC Score (OvR, Weighted): {test_roc_auc:.6f}")
    print("\nConfusion Matrix (Rows=True Class, Columns=Predicted Class):")
    print(conf_matrix)
    
    plot_confusion_matrix(conf_matrix, CLASS_LABELS, title='QDA Test Set Confusion Matrix')
    plot_multiclass_roc(Y_test_encoded, Y_pred_proba, CLASS_LABELS)
    
    print("\nFinal QDA model trained and evaluated successfully.")
else:
    print("\nNo features were selected, no final model was trained or evaluated.")
