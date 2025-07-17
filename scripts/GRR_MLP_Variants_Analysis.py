# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 17:55:35 2025

@author: H.A.R
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import os
import time
import shutil
from scipy.stats import ttest_rel, friedmanchisquare, wilcoxon

# Configuration - USER CAN MODIFY THESE VALUES
DATA_PATH = 'E:/GRR/Complete_data_Set_4_Species.csv'
BASE_DIR = 'E:/GRR/MLP_RBH_all3'
LEARNING_RATES = [0.01, 0.001, 0.0001]  # Can be modified to any list of values
BATCH_SIZES = [32, 64, 128, 256]       # Can be modified to any list of values

METHODS = {
    'NovelGRR': {'dir': 'results_Novel', 'builder': 'build_novel'},
    'AdaptiveL1L2': {'dir': 'results_Adaptive', 'builder': 'build_adaptive'},
    'FixedL1': {'dir': 'results_FixedL1', 'builder': 'build_fixed_l1'},
    'FixedL2': {'dir': 'results_FixedL2', 'builder': 'build_fixed_l2'},
    'ElasticNet': {'dir': 'results_ElasticNet', 'builder': 'build_elastic'},
    'MLP': {'dir': 'results_MLP', 'builder': 'build_mlp'}
}

# 1. Novel Gradient-Responsive Regularizer
class GRRegularizer(regularizers.Regularizer):
    def __init__(self, base_l1=0.01, base_l2=0.01, adapt_rate=0.1, grad_window=10):
        super().__init__()
        self.base_l1 = tf.Variable(base_l1, trainable=False)
        self.base_l2 = tf.Variable(base_l2, trainable=False)
        self.adapt_rate = adapt_rate
        self.grad_window = grad_window
        
        self.grad_buffer = tf.Variable(tf.zeros(grad_window), trainable=False)
        self.buffer_index = tf.Variable(0, dtype=tf.int32)
        self.l1_history = tf.Variable([], shape=[None], dtype=tf.float32)
        self.l2_history = tf.Variable([], shape=[None], dtype=tf.float32)

    def __call__(self, weights):
        current_grad = tf.reduce_mean(tf.abs(weights))
        index = self.buffer_index % self.grad_window
        self.grad_buffer.scatter_nd_update([[index]], [current_grad])
        self.buffer_index.assign_add(1)
        
        valid_grads = self.grad_buffer[:tf.minimum(self.buffer_index, self.grad_window)]
        grad_variance = tf.math.reduce_variance(valid_grads)
        stability_factor = tf.math.exp(-grad_variance)
        
        activation_mean = tf.reduce_mean(tf.abs(weights))
        feature_modulation = tf.math.sigmoid(activation_mean * 10)
        
        effective_l1 = self.base_l1 * stability_factor * feature_modulation
        effective_l2 = self.base_l2 * stability_factor * (1 - feature_modulation)
        
        self.l1_history.assign(tf.concat([self.l1_history, [effective_l1]], axis=0))
        self.l2_history.assign(tf.concat([self.l2_history, [effective_l2]], axis=0))
        
        return effective_l1 * tf.reduce_sum(tf.abs(weights)) + effective_l2 * tf.reduce_sum(tf.square(weights))

    def get_config(self):
        return {
            'base_l1': self.base_l1.numpy(),
            'base_l2': self.base_l2.numpy(),
            'adapt_rate': self.adapt_rate,
            'grad_window': self.grad_window
        }

# 2. Model Builders
def build_model(method, input_shape, num_classes, learning_rate):
    input_shape_tuple = (input_shape,)
    
    if method == 'build_novel':
        input_layer = layers.Input(shape=input_shape_tuple)
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=GRRegularizer(0.01, 0.01))(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        reg_path = layers.Dense(64, activation='linear')(input_layer)
        reg_weights = layers.Dense(128, activation='sigmoid')(reg_path)
        x = layers.Multiply()([x, reg_weights])
        
        x = layers.Dense(64, activation='relu',
                        kernel_regularizer=GRRegularizer(0.005, 0.005))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=input_layer, outputs=output)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    elif method == 'build_adaptive':
        model = models.Sequential([
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.L1L2(0.01, 0.01),
                        input_shape=input_shape_tuple),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.L1L2(0.005, 0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    elif method == 'build_fixed_l1':
        model = models.Sequential([
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.L1(0.01),
                        input_shape=input_shape_tuple),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.L1(0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    elif method == 'build_fixed_l2':
        model = models.Sequential([
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.L2(0.01),
                        input_shape=input_shape_tuple),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.L2(0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    elif method == 'build_elastic':
        model = models.Sequential([
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.L1L2(0.01, 0.01),
                        input_shape=input_shape_tuple),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.L1L2(0.005, 0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    elif method == 'build_mlp':
        model = models.Sequential([
            layers.Dense(256, kernel_regularizer=regularizers.l2(0.01), input_shape=input_shape_tuple),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.3),
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.3),
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.01)),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.3),
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.01)),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    raise ValueError(f"Unknown method: {method}")

# 3. Training Pipeline
def run_experiment(method_name, config, learning_rate, batch_size):
    # Create subfolder for this learning rate and batch size
    lr_str = f"{learning_rate:.2f}".replace('.', '_')  # Changed to use decimal format
    batch_str = str(batch_size)
    output_dir = os.path.join(BASE_DIR, f"lr_{lr_str}_bs_{batch_str}", config['dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        data = pd.read_csv(DATA_PATH)
        X = data.iloc[:, :-1].values
        y = data['Species'].values - 1
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        num_classes = len(np.unique(y))
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # Initialize metrics storage for all folds
        fold_metrics = {
            'train': {
                'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []
            },
            'val': {
                'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []
            },
            'test': {
                'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []
            },
            'training_time': [],
            'val_accuracy': []
        }
        
        best_model_path = os.path.join(output_dir, f"Best_{method_name}_Model.h5")
        best_val_acc = -np.inf
        best_fold = -1
        best_train_index = None
        best_val_index = None
        
        # Start timer for the full model
        full_model_start_time = time.time()

        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            print(f"\n{method_name} (LR={learning_rate}, BS={batch_size}) - Fold {fold+1}/10")
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train_cat[train_index], y_train_cat[val_index]
            
            model = build_model(config['builder'], X_train.shape[1], num_classes, learning_rate)
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                min_delta=0.001,
                mode='max',
                restore_best_weights=True
            )
            
            start_time = time.time()
            history = model.fit(
                X_train_fold, y_train_fold,
                epochs=50,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                verbose=0,
                callbacks=[early_stop]
            )
            
            # Calculate metrics for training data
            y_train_pred = model.predict(X_train_fold)
            y_train_pred_classes = np.argmax(y_train_pred, axis=1)
            y_train_true = np.argmax(y_train_fold, axis=1)
            
            fold_metrics['train']['accuracy'].append(accuracy_score(y_train_true, y_train_pred_classes))
            fold_metrics['train']['precision'].append(precision_score(y_train_true, y_train_pred_classes, average='macro'))
            fold_metrics['train']['recall'].append(recall_score(y_train_true, y_train_pred_classes, average='macro'))
            fold_metrics['train']['f1'].append(f1_score(y_train_true, y_train_pred_classes, average='macro'))
            fold_metrics['train']['mcc'].append(matthews_corrcoef(y_train_true, y_train_pred_classes))
            
            # Calculate metrics for validation data
            y_val_pred = model.predict(X_val_fold)
            y_val_pred_classes = np.argmax(y_val_pred, axis=1)
            y_val_true = np.argmax(y_val_fold, axis=1)
            
            fold_metrics['val']['accuracy'].append(accuracy_score(y_val_true, y_val_pred_classes))
            fold_metrics['val']['precision'].append(precision_score(y_val_true, y_val_pred_classes, average='macro'))
            fold_metrics['val']['recall'].append(recall_score(y_val_true, y_val_pred_classes, average='macro'))
            fold_metrics['val']['f1'].append(f1_score(y_val_true, y_val_pred_classes, average='macro'))
            fold_metrics['val']['mcc'].append(matthews_corrcoef(y_val_true, y_val_pred_classes))
            
            # Calculate metrics for test data
            y_test_pred = model.predict(X_test)
            y_test_pred_classes = np.argmax(y_test_pred, axis=1)
            
            fold_metrics['test']['accuracy'].append(accuracy_score(y_test, y_test_pred_classes))
            fold_metrics['test']['precision'].append(precision_score(y_test, y_test_pred_classes, average='macro'))
            fold_metrics['test']['recall'].append(recall_score(y_test, y_test_pred_classes, average='macro'))
            fold_metrics['test']['f1'].append(f1_score(y_test, y_test_pred_classes, average='macro'))
            fold_metrics['test']['mcc'].append(matthews_corrcoef(y_test, y_test_pred_classes))
            
            # Store training time and validation accuracy
            fold_metrics['training_time'].append(time.time() - start_time)
            fold_metrics['val_accuracy'].append(np.max(history.history['val_accuracy']))
            
            # Save history
            fold_num = fold + 1
            for metric in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                np.save(os.path.join(output_dir, f"fold{fold_num}_{metric}.npy"), 
                        np.array(history.history[metric]))
            
            # Check for best model
            val_acc = np.max(history.history['val_accuracy'])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_fold = fold
                best_train_index = train_index
                best_val_index = val_index
                model.save(best_model_path)

        # Calculate total training time for the full model
        full_model_time_minutes = (time.time() - full_model_start_time) / 60
        
        # Save training time to a CSV file
        time_df = pd.DataFrame({
            'Method': [method_name],
            'Learning_Rate': [learning_rate],
            'Batch_Size': [batch_size],
            'Total_Training_Time_Minutes': [full_model_time_minutes]
        })
        time_df.to_csv(os.path.join(output_dir, "Training_Time.csv"), index=False)

        # Save fold metrics to CSV
        fold_metrics_df = pd.DataFrame({
            'Fold': range(1, 11),
            'Train_Accuracy': fold_metrics['train']['accuracy'],
            'Train_Precision': fold_metrics['train']['precision'],
            'Train_Recall': fold_metrics['train']['recall'],
            'Train_F1': fold_metrics['train']['f1'],
            'Train_MCC': fold_metrics['train']['mcc'],
            'Val_Accuracy': fold_metrics['val']['accuracy'],
            'Val_Precision': fold_metrics['val']['precision'],
            'Val_Recall': fold_metrics['val']['recall'],
            'Val_F1': fold_metrics['val']['f1'],
            'Val_MCC': fold_metrics['val']['mcc'],
            'Test_Accuracy': fold_metrics['test']['accuracy'],
            'Test_Precision': fold_metrics['test']['precision'],
            'Test_Recall': fold_metrics['test']['recall'],
            'Test_F1': fold_metrics['test']['f1'],
            'Test_MCC': fold_metrics['test']['mcc'],
            'Training_Time': fold_metrics['training_time'],
            'Val_Accuracy_Epoch': fold_metrics['val_accuracy']
        })
        fold_metrics_df.to_csv(os.path.join(output_dir, "Fold_Metrics_Detailed.csv"), index=False)
        
        # Calculate and save average metrics
        avg_metrics = {
            'Dataset': ['Train', 'Validation', 'Test'],
            'Accuracy': [
                np.mean(fold_metrics['train']['accuracy']),
                np.mean(fold_metrics['val']['accuracy']),
                np.mean(fold_metrics['test']['accuracy'])
            ],
            'Precision': [
                np.mean(fold_metrics['train']['precision']),
                np.mean(fold_metrics['val']['precision']),
                np.mean(fold_metrics['test']['precision'])
            ],
            'Recall': [
                np.mean(fold_metrics['train']['recall']),
                np.mean(fold_metrics['val']['recall']),
                np.mean(fold_metrics['test']['recall'])
            ],
            'F1': [
                np.mean(fold_metrics['train']['f1']),
                np.mean(fold_metrics['val']['f1']),
                np.mean(fold_metrics['test']['f1'])
            ],
            'MCC': [
                np.mean(fold_metrics['train']['mcc']),
                np.mean(fold_metrics['val']['mcc']),
                np.mean(fold_metrics['test']['mcc'])
            ]
        }
        avg_metrics_df = pd.DataFrame(avg_metrics)
        avg_metrics_df.to_csv(os.path.join(output_dir, "Average_Metrics.csv"), index=False)
        
        # Process best model
        if best_fold >= 0:
            custom_objects = {'GRRegularizer': GRRegularizer} if 'Novel' in method_name else {}
            best_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)
            
            # Calculate metrics for best model
            metrics_sets = {}
            
            # Training set metrics
            X_best_train = X_train[best_train_index]
            y_best_train = y_train_cat[best_train_index]
            y_train_pred = best_model.predict(X_best_train)
            y_train_pred_classes = np.argmax(y_train_pred, axis=1)
            y_train_true = np.argmax(y_best_train, axis=1)
            
            metrics_sets['train'] = {
                'accuracy': accuracy_score(y_train_true, y_train_pred_classes),
                'precision': precision_score(y_train_true, y_train_pred_classes, average='macro'),
                'recall': recall_score(y_train_true, y_train_pred_classes, average='macro'),
                'f1': f1_score(y_train_true, y_train_pred_classes, average='macro'),
                'mcc': matthews_corrcoef(y_train_true, y_train_pred_classes),
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
            
            # Validation set metrics
            X_best_val = X_train[best_val_index]
            y_best_val = y_train_cat[best_val_index]
            y_val_pred = best_model.predict(X_best_val)
            y_val_pred_classes = np.argmax(y_val_pred, axis=1)
            y_val_true = np.argmax(y_best_val, axis=1)
            
            metrics_sets['validation'] = {
                'accuracy': accuracy_score(y_val_true, y_val_pred_classes),
                'precision': precision_score(y_val_true, y_val_pred_classes, average='macro'),
                'recall': recall_score(y_val_true, y_val_pred_classes, average='macro'),
                'f1': f1_score(y_val_true, y_val_pred_classes, average='macro'),
                'mcc': matthews_corrcoef(y_val_true, y_val_pred_classes),
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
            
            # Test set metrics
            y_test_pred = best_model.predict(X_test)
            y_test_pred_classes = np.argmax(y_test_pred, axis=1)
            
            metrics_sets['test'] = {
                'accuracy': accuracy_score(y_test, y_test_pred_classes),
                'precision': precision_score(y_test, y_test_pred_classes, average='macro'),
                'recall': recall_score(y_test, y_test_pred_classes, average='macro'),
                'f1': f1_score(y_test, y_test_pred_classes, average='macro'),
                'mcc': matthews_corrcoef(y_test, y_test_pred_classes),
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
            
            # Save best model metrics to CSV
            best_metrics_df = pd.DataFrame(metrics_sets).T
            best_metrics_df.to_csv(os.path.join(output_dir, "Best_Model_Metrics.csv"))
            
            # Confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix(y_test, y_test_pred_classes), 
                        annot=True, fmt='d', cmap='Blues',
                        xticklabels=[f"Species {i+1}" for i in range(num_classes)],
                        yticklabels=[f"Species {i+1}" for i in range(num_classes)])
            plt.title(f'{method_name} (LR={learning_rate}, BS={batch_size}) Confusion Matrix')
            plt.savefig(os.path.join(output_dir, "Confusion_Matrix.png"))
            plt.close()
        
        # Clean up - remove unwanted CSV files (keeping only specified ones)
        files_to_keep = [
            "Training_Time.csv", "Fold_Metrics_Detailed.csv", 
            "Average_Metrics.csv", "Best_Model_Metrics.csv"
        ]
        for file in os.listdir(output_dir):
            if file.endswith('.csv') and file not in files_to_keep:
                os.remove(os.path.join(output_dir, file))
        
        return metrics_sets
    
    except Exception as e:
        print(f"Error in {method_name} (LR={learning_rate}, BS={batch_size}): {str(e)}")
        return None

# 4. Analysis Functions - Simplified version without removed files
def analyze_results(base_dir):
    # This function is kept for compatibility but doesn't generate the removed files
    print("Analysis function running (simplified version)")
    return None

# 5. Main Execution
if __name__ == "__main__":
    # Clean previous results if they exist
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Run all experiments
    for learning_rate in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            for method_name, config in METHODS.items():
                print(f"\n{'='*40}\nRunning {method_name} with LR={learning_rate}, BS={batch_size}\n{'='*40}")
                run_experiment(method_name, config, learning_rate, batch_size)
    
    # Run simplified analysis
    print("\nAnalyzing results...")
    analyze_results(BASE_DIR)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Key outputs generated for each method configuration:")
    print("- Training_Time.csv: Training time information")
    print("- Fold_Metrics_Detailed.csv: Detailed metrics for all 10 folds")
    print("- Average_Metrics.csv: Average metrics across folds")
    print("- Best_Model_Metrics.csv: Metrics for the best model")
    print("- Confusion_Matrix.png: Confusion matrix for test data")
    print("\nResults saved to:", BASE_DIR)