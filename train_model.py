"""
EduPulse AI — Student Academic Performance & Pass/Fail Prediction System
Segment 1: Dataset Generation, Machine Learning Pipeline, High-DPI Visualizations & JSON Export
"""

import os
import sys

# Set writable cache directory for Matplotlib before importing matplotlib
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.mplconfig')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import json
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive Agg backend for robust headless rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Set writable cache directory for Matplotlib
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.mplconfig')

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
PROJECT_NAME = "EduPulse AI — Student Academic Performance & Pass/Fail Prediction System"
OUTPUT_DIR = "outputs"
DATASET_PATH = "student_data.csv"
JSON_EXPORT_PATH = "model_data.json"

COLOR_INDIGO = "#4F46E5"
COLOR_EMERALD = "#10B981"
COLOR_SLATE_DARK = "#1E293B"
COLOR_SLATE_MUTED = "#64748B"
COLOR_BORDER = "#E2E8F0"
COLOR_BG_LIGHT = "#F8FAFC"

FEATURE_NAMES = [
    "Hours_Studied",
    "Attendance_Percentage",
    "Previous_Score",
    "Assignments_Completed"
]

FEATURE_LABELS = [
    "Hours Studied",
    "Attendance (%)",
    "Previous Score",
    "Assignments Completed"
]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(".mplconfig", exist_ok=True)

# Set base matplotlib styling parameters
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial', 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.0


# ==============================================================================
# 2. DATASET GENERATION
# ==============================================================================
def generate_dataset(n_samples=500, random_state=19):
    """
    Generates a realistic student academic dataset (500 rows) with 5 columns:
    Hours_Studied, Attendance_Percentage, Previous_Score, Assignments_Completed, Result.
    Uses a weighted realistic academic formula with slight Gaussian noise tuned
    to achieve ~92% machine learning prediction accuracy.
    """
    np.random.seed(random_state)
    
    # 1. Feature distributions within specified ranges
    hours_studied = np.random.uniform(1.0, 10.0, n_samples)
    attendance_pct = np.random.uniform(50.0, 100.0, n_samples)
    previous_score = np.random.uniform(30.0, 100.0, n_samples)
    assignments_completed = np.random.randint(0, 11, n_samples).astype(float)
    
    # 2. Normalized feature values for realistic composite academic score [0, 1]
    h_norm = (hours_studied - 1.0) / 9.0
    att_norm = (attendance_pct - 50.0) / 50.0
    score_norm = (previous_score - 30.0) / 70.0
    assign_norm = assignments_completed / 10.0
    
    # Realistic weights: Previous Score (32%), Hours Studied (28%), Attendance (24%), Assignments (16%)
    composite_score = (
        0.28 * h_norm +
        0.24 * att_norm +
        0.32 * score_norm +
        0.16 * assign_norm
    )
    
    # 3. Add calibrated Gaussian noise to simulate real-world human performance variability
    noise = np.random.normal(0.0, 0.040, n_samples)
    total_score = composite_score + noise
    
    # 4. Result threshold (1 = Pass, 0 = Fail)
    result = (total_score >= 0.48).astype(int)
    
    # Construct clean DataFrame with rounded realistic values
    df = pd.DataFrame({
        'Hours_Studied': np.round(hours_studied, 1),
        'Attendance_Percentage': np.round(attendance_pct, 1),
        'Previous_Score': np.round(previous_score, 1),
        'Assignments_Completed': assignments_completed.astype(int),
        'Result': result
    })
    
    return df


# ==============================================================================
# 3. MODEL TRAINING & EVALUATION
# ==============================================================================
def train_and_evaluate(df):
    """
    Splits data 80/20 (stratify=y, random_state=42), fits StandardScaler,
    trains Logistic Regression & Random Forest Classifier, and computes full metrics.
    """
    X = df[FEATURE_NAMES]
    y = df['Result']
    
    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Fit StandardScaler on train set and transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Model 1: Logistic Regression ---
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    # --- Model 2: Random Forest Classifier ---
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]
    
    # Compute evaluation metrics
    metrics = {
        'logistic_regression': {
            'accuracy': float(accuracy_score(y_test, y_pred_lr)),
            'precision': float(precision_score(y_test, y_pred_lr)),
            'recall': float(recall_score(y_test, y_pred_lr)),
            'f1_score': float(f1_score(y_test, y_pred_lr)),
            'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist()
        },
        'random_forest': {
            'accuracy': float(accuracy_score(y_test, y_pred_rf)),
            'precision': float(precision_score(y_test, y_pred_rf)),
            'recall': float(recall_score(y_test, y_pred_rf)),
            'f1_score': float(f1_score(y_test, y_pred_rf)),
            'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist(),
            'feature_importances': {
                feat: float(imp)
                for feat, imp in zip(FEATURE_NAMES, rf_clf.feature_importances_)
            }
        }
    }
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'log_reg': log_reg,
        'rf_clf': rf_clf,
        'y_pred_lr': y_pred_lr,
        'y_pred_rf': y_pred_rf,
        'metrics': metrics
    }


# ==============================================================================
# 4. HIGH-DPI VISUALIZATIONS (300 DPI, White Background, Despined)
# ==============================================================================

def remove_spines(ax):
    """Helper to remove top and right spines from axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_confusion_matrices(metrics, save_path):
    """
    Generates side-by-side high-DPI confusion matrices for Logistic Regression
    and Random Forest using custom Indigo & Emerald gradient palettes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='white', dpi=300)
    
    cm_lr = np.array(metrics['logistic_regression']['confusion_matrix'])
    cm_rf = np.array(metrics['random_forest']['confusion_matrix'])
    
    models_info = [
        ("Logistic Regression", cm_lr, metrics['logistic_regression']['accuracy'], COLOR_INDIGO, axes[0]),
        ("Random Forest Classifier", cm_rf, metrics['random_forest']['accuracy'], COLOR_EMERALD, axes[1])
    ]
    
    for title, cm, acc, primary_color, ax in models_info:
        # Create subtle custom colormap from white/light-tint to theme color
        cmap = LinearSegmentedColormap.from_list(
            'custom_theme',
            ['#FFFFFF', '#F1F5F9', primary_color],
            N=256
        )
        
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
        
        ax.set_title(f"{title}\nAccuracy: {acc:.1%}", fontsize=14, fontweight='bold', pad=16, color=COLOR_SLATE_DARK)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Fail (0)', 'Pass (1)'], fontsize=11, fontweight='600', color=COLOR_SLATE_DARK)
        ax.set_yticklabels(['Fail (0)', 'Pass (1)'], fontsize=11, fontweight='600', color=COLOR_SLATE_DARK)
        ax.set_xlabel('Predicted Academic Outcome', fontsize=12, fontweight='600', labelpad=10, color=COLOR_SLATE_DARK)
        ax.set_ylabel('Actual Academic Outcome', fontsize=12, fontweight='600', labelpad=10, color=COLOR_SLATE_DARK)
        
        # Annotate cells with count and percentage
        thresh = cm.max() / 2.0
        total_samples = cm.sum()
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total_samples * 100
                cell_label = f"{val}\n({pct:.1f}%)"
                text_color = "white" if val > thresh else COLOR_SLATE_DARK
                ax.text(
                    j, i, cell_label,
                    ha="center", va="center",
                    color=text_color, fontsize=12, fontweight='bold'
                )
                
        remove_spines(ax)
        ax.tick_params(top=False, right=False, bottom=False, left=False)
        
    plt.tight_layout(w_pad=4.0)
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {save_path}")


def plot_model_comparison(metrics, save_path):
    """
    Generates a grouped bar chart comparing Accuracy, Precision, Recall,
    and F1-Score between Logistic Regression and Random Forest.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='white', dpi=300)
    
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    
    lr_vals = [metrics['logistic_regression'][k] * 100 for k in metric_keys]
    rf_vals = [metrics['random_forest'][k] * 100 for k in metric_keys]
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    rects1 = ax.bar(
        x - width/2, lr_vals, width,
        label='Logistic Regression', color=COLOR_INDIGO,
        edgecolor='none', zorder=3, alpha=0.95
    )
    rects2 = ax.bar(
        x + width/2, rf_vals, width,
        label='Random Forest', color=COLOR_EMERALD,
        edgecolor='none', zorder=3, alpha=0.95
    )
    
    # Value annotations on top of bars
    for rects, color in [(rects1, COLOR_INDIGO), (rects2, COLOR_EMERALD)]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10.5, fontweight='bold', color=COLOR_SLATE_DARK
            )
            
    ax.set_ylabel('Metric Score (%)', fontsize=12, fontweight='600', color=COLOR_SLATE_DARK, labelpad=10)
    ax.set_title(
        'EduPulse AI — Model Performance Benchmark\nLogistic Regression vs. Random Forest Classifier',
        fontsize=14, fontweight='bold', pad=20, color=COLOR_SLATE_DARK
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='600', color=COLOR_SLATE_DARK)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#94A3B8', zorder=0)
    
    ax.legend(
        loc='upper right', frameon=True, facecolor='#F8FAFC',
        edgecolor=COLOR_BORDER, fontsize=10.5, fancybox=True
    )
    
    remove_spines(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {save_path}")


def plot_feature_importance(model_results, save_path):
    """
    Generates a horizontal comparison chart displaying Random Forest feature
    importances alongside standardized Logistic Regression coefficient weights.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='white', dpi=300)
    
    rf = model_results['rf_clf']
    lr = model_results['log_reg']
    
    rf_importances = rf.feature_importances_ * 100
    lr_weights = np.abs(lr.coef_[0])
    lr_normalized = (lr_weights / lr_weights.sum()) * 100
    
    y = np.arange(len(FEATURE_NAMES))
    height = 0.36
    
    # Sort indices by Random Forest importance
    sorted_idx = np.argsort(rf_importances)
    
    ax.barh(
        y - height/2, [lr_normalized[i] for i in sorted_idx], height,
        label='Logistic Regression (|Std. Coef| %)', color=COLOR_INDIGO,
        alpha=0.92, zorder=3
    )
    ax.barh(
        y + height/2, [rf_importances[i] for i in sorted_idx], height,
        label='Random Forest (Gini Importance %)', color=COLOR_EMERALD,
        alpha=0.92, zorder=3
    )
    
    # Annotate bar values
    for i, idx in enumerate(sorted_idx):
        lr_val = lr_normalized[idx]
        rf_val = rf_importances[idx]
        ax.text(lr_val + 0.8, i - height/2, f"{lr_val:.1f}%", va='center', fontsize=9.5, fontweight='bold', color=COLOR_INDIGO)
        ax.text(rf_val + 0.8, i + height/2, f"{rf_val:.1f}%", va='center', fontsize=9.5, fontweight='bold', color=COLOR_EMERALD)
    
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS[i] for i in sorted_idx], fontsize=11, fontweight='600', color=COLOR_SLATE_DARK)
    ax.set_xlabel('Relative Feature Contribution / Importance (%)', fontsize=12, fontweight='600', color=COLOR_SLATE_DARK, labelpad=10)
    ax.set_title(
        'EduPulse AI — Feature Importance & Predictive Contribution Analysis',
        fontsize=14, fontweight='bold', pad=18, color=COLOR_SLATE_DARK
    )
    ax.set_xlim(0, max(rf_importances.max(), lr_normalized.max()) + 10)
    ax.grid(axis='x', linestyle='--', alpha=0.4, color='#94A3B8', zorder=0)
    
    ax.legend(
        loc='lower right', frameon=True, facecolor='#F8FAFC',
        edgecolor=COLOR_BORDER, fontsize=10.5, fancybox=True
    )
    
    remove_spines(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {save_path}")


def plot_study_vs_score_scatter(df, save_path):
    """
    Generates a scatter plot of Hours Studied vs. Previous Score,
    colored by Result (Pass: Emerald #10B981, Fail: Indigo #4F46E5),
    with clean decision boundary approximation and marginal trends.
    """
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor='white', dpi=300)
    
    pass_mask = df['Result'] == 1
    fail_mask = df['Result'] == 0
    
    # Scatter points for Fail
    ax.scatter(
        df.loc[fail_mask, 'Hours_Studied'],
        df.loc[fail_mask, 'Previous_Score'],
        color=COLOR_INDIGO,
        label=f'Fail (0) — {fail_mask.sum()} Students',
        alpha=0.75,
        s=55,
        edgecolor='white',
        linewidth=0.8,
        zorder=3
    )
    
    # Scatter points for Pass
    ax.scatter(
        df.loc[pass_mask, 'Hours_Studied'],
        df.loc[pass_mask, 'Previous_Score'],
        color=COLOR_EMERALD,
        label=f'Pass (1) — {pass_mask.sum()} Students',
        alpha=0.75,
        s=55,
        edgecolor='white',
        linewidth=0.8,
        zorder=4
    )
    
    # Visual decision boundary trendline (approximate linear frontier)
    # y = mx + c separating pass from fail
    x_vals = np.linspace(1.0, 10.0, 200)
    # Empirically derived trend boundary for visualization
    y_trend = 94.0 - 5.5 * x_vals
    ax.plot(
        x_vals, y_trend,
        color=COLOR_SLATE_MUTED, linestyle='--', linewidth=1.8,
        label='Empirical Passing Frontier', zorder=2, alpha=0.8
    )
    
    # Fill subtle background shading to emphasize passing / at-risk zones
    ax.fill_between(x_vals, y_trend, 105, color=COLOR_EMERALD, alpha=0.06, label='Passing Zone')
    ax.fill_between(x_vals, 25, y_trend, color=COLOR_INDIGO, alpha=0.05, label='At-Risk Zone')
    
    ax.set_title(
        'EduPulse AI — Academic Distribution: Study Hours vs. Previous Score\nClassified by Pass/Fail Outcome',
        fontsize=14, fontweight='bold', pad=18, color=COLOR_SLATE_DARK
    )
    ax.set_xlabel('Study Time (Hours / Week)', fontsize=12, fontweight='600', color=COLOR_SLATE_DARK, labelpad=10)
    ax.set_ylabel('Previous Academic Score (Points)', fontsize=12, fontweight='600', color=COLOR_SLATE_DARK, labelpad=10)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(25, 105)
    
    ax.grid(True, linestyle='--', alpha=0.35, color='#94A3B8', zorder=0)
    
    ax.legend(
        loc='upper right', frameon=True, facecolor='#F8FAFC',
        edgecolor=COLOR_BORDER, fontsize=10, fancybox=True
    )
    
    remove_spines(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {save_path}")


# ==============================================================================
# 5. EXPORT JSON FOR BROWSER / CLIENT
# ==============================================================================
def export_model_json(df, model_results, save_path):
    """
    Exports all project metadata, weights, scaler parameters, metrics,
    dataset summary statistics, and the first 15 rows for browser runtime consumption.
    """
    scaler = model_results['scaler']
    lr = model_results['log_reg']
    metrics = model_results['metrics']
    
    # Logistic Regression parameters
    coefficients = lr.coef_[0].tolist()
    intercept = float(lr.intercept_[0])
    
    # StandardScaler parameters
    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()
    
    # Dataset summary statistics
    numeric_stats = {}
    for col in FEATURE_NAMES:
        numeric_stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'q25': float(df[col].quantile(0.25)),
            'median': float(df[col].median()),
            'q75': float(df[col].quantile(0.75)),
            'max': float(df[col].max())
        }
        
    pass_count = int((df['Result'] == 1).sum())
    fail_count = int((df['Result'] == 0).sum())
    total_count = len(df)
    
    # First 15 rows of student_data.csv as list of dictionaries
    first_15_rows = df.head(15).to_dict(orient='records')
    
    payload = {
        "project_name": PROJECT_NAME,
        "features": FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
        "target": "Result",
        
        # Browser predictor weights & scaler parameters (both flat and nested for easy access)
        "logistic_regression": {
            "coefficients": coefficients,
            "intercept": intercept,
            "feature_coefficients": dict(zip(FEATURE_NAMES, coefficients))
        },
        "coefficients": coefficients,
        "intercept": intercept,
        
        "scaler": {
            "mean": scaler_mean,
            "scale": scaler_scale,
            "feature_means": dict(zip(FEATURE_NAMES, scaler_mean)),
            "feature_scales": dict(zip(FEATURE_NAMES, scaler_scale))
        },
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        
        # Model evaluation metrics
        "evaluation_metrics": metrics,
        
        # Dataset summary statistics
        "dataset_summary": {
            "total_records": total_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_percentage": round(pass_count / total_count * 100, 2),
            "fail_percentage": round(fail_count / total_count * 100, 2),
            "features_summary": numeric_stats
        },
        
        # First 15 records
        "first_15_rows": first_15_rows
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        
    print(f"  [+] Saved: {save_path}")


# ==============================================================================
# 6. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    print("=" * 76)
    print(f"  {PROJECT_NAME}")
    print("  Segment 1: Dataset & Machine Learning Pipeline")
    print("=" * 76)
    
    # 1. Generate clean realistic dataset (500 rows)
    print("\n[1/5] Generating student dataset (500 rows)...")
    df = generate_dataset(n_samples=500, random_state=19)
    df.to_csv(DATASET_PATH, index=False)
    print(f"  [+] Saved dataset to '{DATASET_PATH}' ({len(df)} rows, {len(df.columns)} columns)")
    print(f"      Pass rate: {(df['Result'] == 1).mean():.1%} (Pass: {(df['Result'] == 1).sum()}, Fail: {(df['Result'] == 0).sum()})")
    
    # 2. Train and evaluate models
    print("\n[2/5] Training Logistic Regression & Random Forest models...")
    model_results = train_and_evaluate(df)
    metrics = model_results['metrics']
    
    lr_m = metrics['logistic_regression']
    rf_m = metrics['random_forest']
    
    print("\n  ================================================================")
    print("  MODEL EVALUATION RESULTS (80/20 Stratified Split, Test Set N=100)")
    print("  ================================================================")
    print(f"  {'Metric':<18} | {'Logistic Regression':<22} | {'Random Forest':<20}")
    print("  " + "-" * 64)
    print(f"  {'Accuracy':<18} | {lr_m['accuracy']:<22.2%} | {rf_m['accuracy']:<20.2%}")
    print(f"  {'Precision':<18} | {lr_m['precision']:<22.2%} | {rf_m['precision']:<20.2%}")
    print(f"  {'Recall':<18} | {lr_m['recall']:<22.2%} | {rf_m['recall']:<20.2%}")
    print(f"  {'F1-Score':<18} | {lr_m['f1_score']:<22.3f} | {rf_m['f1_score']:<20.3f}")
    print("  " + "-" * 64)
    print(f"  Confusion Matrix (LR): TN={lr_m['confusion_matrix'][0][0]}, FP={lr_m['confusion_matrix'][0][1]}, FN={lr_m['confusion_matrix'][1][0]}, TP={lr_m['confusion_matrix'][1][1]}")
    print(f"  Confusion Matrix (RF): TN={rf_m['confusion_matrix'][0][0]}, FP={rf_m['confusion_matrix'][0][1]}, FN={rf_m['confusion_matrix'][1][0]}, TP={rf_m['confusion_matrix'][1][1]}")
    
    # 3. Generate 4 High-DPI Charts
    print("\n[3/5] Generating 4 High-Resolution (300 DPI) Plots in 'outputs/'...")
    plot_confusion_matrices(metrics, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_model_comparison(metrics, os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plot_feature_importance(model_results, os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plot_study_vs_score_scatter(df, os.path.join(OUTPUT_DIR, "study_vs_score_scatter.png"))
    
    # 4. Export JSON for Browser
    print("\n[4/5] Exporting model weights, scaler parameters, and metadata to JSON...")
    export_model_json(df, model_results, JSON_EXPORT_PATH)
    
    # 5. Verification
    print("\n[5/5] Verifying generated artifacts...")
    required_files = [
        DATASET_PATH,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        os.path.join(OUTPUT_DIR, "model_comparison.png"),
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
        os.path.join(OUTPUT_DIR, "study_vs_score_scatter.png"),
        JSON_EXPORT_PATH
    ]
    
    all_ok = True
    for fpath in required_files:
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  [OK] {fpath} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {fpath}")
            all_ok = False
            
    if all_ok:
        print("\n--> Pipeline executed successfully! All files generated and verified.")
    else:
        print("\n--> Warning: Some files are missing!")
        sys.exit(1)


if __name__ == "__main__":
    main()
