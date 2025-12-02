"""
@author: gagui
"""

import zipfile
import json
import time
import pandas as panda_object
import numpy as numpy_object
import matplotlib.pyplot as plot_object
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, auc, roc_curve, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint 


def load_data():
    """
    Load dataframe for analysis
    Returns:
        A dataframe from the passed in data set.
    """
    zip_path = "../data_processed.zip"

    with zipfile.ZipFile(zip_path, 'r') as z:
        X_train_std = panda_object.read_csv(z.open('X_train_std.csv'))
        Y_train = panda_object.read_csv(z.open('Y_train.csv'))
        X_test_std  = panda_object.read_csv(z.open('X_test_std.csv'))
        Y_test  = panda_object.read_csv(z.open('Y_test.csv'))
    return X_train_std, Y_train, X_test_std, Y_test

def ex_load_data():
    """
    Load dataframe for analysis
    Returns:
        A dataframe from the passed in data set.
    """
    zip_path = "../ex_data_processed.zip"

    with zipfile.ZipFile(zip_path, 'r') as z:
        X_train_std = panda_object.read_csv(z.open('ex_X_train_std.csv'))
        Y_train = panda_object.read_csv(z.open('ex_Y_train.csv'))
        X_test_std  = panda_object.read_csv(z.open('ex_x_test_std.csv'))
        Y_test  = panda_object.read_csv(z.open('ex_Y_test.csv'))
    return X_train_std, Y_train, X_test_std, Y_test

def get_ovr_feature_importances(ovr):
    return numpy_object.sum(numpy_object.abs(numpy_object.stack([e.coef_ for e in ovr.estimator_], axis=0)), axis=0).flatten()


def preprocess_knn_model_with_lasso(X_train,):
    '''
    Preprocessing and outputting a feature pipeline to reduce x space
    '''
    X_train_cat = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    X_train_num = X_train.select_dtypes(include=['number']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), X_train_cat), 
            ('num', 'passthrough', X_train_num) 
        ],
        remainder='passthrough'
    )
    
    logreg_estimator = LogisticRegression(
        penalty='l1', 
        C=0.1,
        solver='liblinear', 
        random_state=42,
        max_iter=5000 
    )

    l1_selector = SelectFromModel(
        logreg_estimator,
        threshold='mean',
        importance_getter='coef_',
        prefit=False
    )

    feature_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', l1_selector)
    ])

    base_knn_estimator = KNeighborsClassifier(n_neighbors=5)
    final_knn_estimator = OneVsRestClassifier(base_knn_estimator, n_jobs=-1)


    return feature_pipeline, final_knn_estimator

def feature_selection_process(feature_pipeline, X_train, y_train, X_test):
    '''
    Finds best features before fitting final model, returns a reduced x space from it.
    '''
    sfs_start_time = time.time()
    
    X_train_reduced = feature_pipeline.fit_transform(X_train, y_train)

    sfs_end_time = time.time()
    time_to_sfs = sfs_end_time - sfs_start_time

    X_test_reduced = feature_pipeline.transform(X_test)
    
    feature_selection_result = {
        'time_to_lasso': time_to_sfs,
        'X_train_reduced': X_train_reduced,
        'X_test_reduced': X_test_reduced
    }
    
    return feature_selection_result

def knn_model_training(ovn_knn, feature_selection_result, y_train):
    n_iter_search=5
    param_dist = {'estimator__n_neighbors': randint(1, 26)}
       
    random_search_start_time = time.time()

    rs_model = RandomizedSearchCV(
        estimator=ovn_knn,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring='f1_weighted',
        cv=5,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )

    rs_model.fit(feature_selection_result['X_train_reduced'], y_train)

    random_search_end_time = time.time()
    time_to_tune = random_search_end_time - random_search_start_time
    
    knn_model_training_result = {
        'rs_model': rs_model,
        'time_to_tune': time_to_tune
    }
    
    return knn_model_training_result

def get_cv_metrics(knn_model_training_result):
    cv_results = {
        'best_cv_f1_weighted': knn_model_training_result['rs_model'].best_score_,
        'tried_k_values': knn_model_training_result['rs_model'].cv_results_['param_estimator__n_neighbors'].tolist(),
        'mean_cv_scores': knn_model_training_result['rs_model'].cv_results_['mean_test_score'].tolist(),
        'best_k_neighbors': knn_model_training_result['rs_model'].best_params_['estimator__n_neighbors'],
    }

    return cv_results

def save_confusion_matrix(y_true,y_pred,class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plot_object.subplots(figsize=(8, 8))
    disp.plot(cmap=plot_object.cm.Blues, ax=ax, xticks_rotation='vertical')
    ax.set_title("Confusion Matrix (Raw Counts)")
    plot_object.tight_layout()
    plot_object.savefig('knn_multiclass_confusion_matrix.png', dpi=300)
    plot_object.close()
    
def save_roc_curve(y_train,y_test,  y_proba):
    class_labels = numpy_object.unique(y_train)
    # Binarize the output for multi-class ROC curve
    y_test_bin = label_binarize(y_test, classes=class_labels)

    plot_object.figure(figsize=(10, 8))
    
    # Compute ROC curve and ROC area for each class
    for i, class_label in enumerate(class_labels):
        # We need the probability of the positive class (i) vs all others
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plot_object.plot(
            fpr, tpr, 
            label=f'Class {class_label} (AUC = {roc_auc:.2f})',
            linewidth=2
        )

    plot_object.plot([0, 1], [0, 1], 'k--', label='Chance')
    plot_object.xlim([0.0, 1.0])
    plot_object.ylim([0.0, 1.05])
    plot_object.xlabel('False Positive Rate')
    plot_object.ylabel('True Positive Rate')
    plot_object.title('Per-Label Receiver Operating Characteristic (ROC) Curve')
    plot_object.legend(loc="lower right")
    plot_object.grid(True)
    plot_object.savefig('knn_roc_curve.png')
    plot_object.close()
    
def run_knn_multiclass_classification(X_train,y_train,X_test,y_test):
    # --- Preprocess ---
    feature_pipeline, ovn_knn = preprocess_knn_model_with_lasso(X_train)
    
    feature_selection_result = feature_selection_process(feature_pipeline, X_train, y_train, X_test)
    
    # --- Model Training ---
    
    knn_model_training_result = knn_model_training(ovn_knn, feature_selection_result, y_train)
         
    best_model = knn_model_training_result['rs_model'].best_estimator_
    
    # --- Evaluate ---

    y_pred = best_model.predict(feature_selection_result['X_test_reduced']) 
    
    # --- Metrics and Plots ---
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    test_accuracy = accuracy_score(y_test, y_pred)
    y_proba = knn_model_training_result['rs_model'].predict_proba(feature_selection_result['X_test_reduced'])
            
    # Confusion Matrix
    class_labels = numpy_object.unique(y_train)
    save_confusion_matrix(y_test, y_pred, class_labels)
    
    # ROC

    save_roc_curve(y_train,y_test, y_proba)
    
    # Metrics Extraction
    
    feature_names_processed = feature_pipeline['preprocessor'].get_feature_names_out(X_train.columns.tolist())
    selected_indices = feature_pipeline['selector'].get_support(indices=True)
    selected_features = feature_names_processed[selected_indices].tolist()
    cv_metrics = get_cv_metrics(knn_model_training_result)
    total_time_to_train = feature_selection_result['time_to_lasso'] + knn_model_training_result['time_to_tune']
    # Save to JSON
    report_data = {
        "best_cv_f1_weighted": cv_metrics['best_cv_f1_weighted'],
        "f1_weighted_score": f1_weighted,
        "overall_test_accuracy": test_accuracy,
        "overall_error_rate": 1-test_accuracy,
        "tried_k_neighbors": cv_metrics['tried_k_values'],
        "mean_cv_scores": cv_metrics['mean_cv_scores'],
        "best_n_neighbors": cv_metrics['best_k_neighbors'],
        "total_training_tuning_time_seconds": total_time_to_train,
        "number_of_features": len(selected_features),
        "selected_features": selected_features
    }
    
    report_filename = "knn_classification_report.json"
    with open(report_filename, 'w') as f:
        json.dump(report_data, f, indent=4)
        

# Main Function of Script
if __name__ == "__main__":
    '''
    Every result file will be dumped into the directory its in.
    '''
    try:
        
        X_train,y_train,X_test,y_test = load_data()
        
        run_knn_multiclass_classification(X_train,y_train,X_test,y_test )

        

    except FileNotFoundError:
        print(f"Error: The file was not found.")

        