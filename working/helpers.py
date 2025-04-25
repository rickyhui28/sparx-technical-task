import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scipy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro,ttest_ind, mannwhitneyu



# EDA Functions

def clean_data(df):
    """Drop duplicates and fill missing numeric values with median."""
    df = df.copy()
    df.drop_duplicates(inplace=True)
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def plot_distributions(df, columns):
    """Plot histogram and KDE (line) separately for more control."""
    for col in columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], bins=20, color='purple', edgecolor='white', alpha=0.6)
        sns.kdeplot(df[col], color='darkblue', linewidth=2)
        plt.title(f"{col.title()} Distribution with KDE Line")
        plt.xlabel(col.title())
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



def plot_correlation_matrix(df):
    """Plot a correlation heatmap for numeric variables."""
    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()



# Statistic functions


def summary_statistics(df):
    """Print basic info and descriptive statistics."""
    print("Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescriptive Statistics:\n", df.describe())


def check_normality(series, plot=True, alpha=0.05):
    """
    Performs Shapiro-Wilk test for normality.

    Parameters:
    - series: pandas Series
    - plot: whether to plot the distribution (default: True)
    - alpha: significance level for the test (default: 0.05)

    Returns:
    - Dictionary with W-statistic, p-value, and decision
    """

    stat, p = shapiro(series.dropna())

    result = {
        "statistic": stat,
        "p_value": p,
        "conclusion": "Data is normal" if p > alpha else "Data is NOT normal"
    }

    if plot:
        plt.figure(figsize=(6, 4))
        sns.histplot(series.dropna(), kde=True)
        plt.title(f"Distribution (Shapiro p = {p:.3f})")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return result


def run_hypothesis_test(df, group_col, target_col, threshold=3, test_type="t-test"):
    """
    Runs a hypothesis test comparing two groups on a numeric outcome.

    Parameters:
    - df: DataFrame
    - group_col: column to split groups by (e.g. 'sessions_per_week')
    - target_col: outcome variable (e.g. 'score')
    - threshold: value to split high vs. low engagement
    - test_type: "t-test" (default) or "mannwhitney"

    Returns:
    - Dictionary with p-value, test name, and conclusion
    """

    high_group = df[df[group_col] >= threshold][target_col].dropna()
    low_group = df[df[group_col] < threshold][target_col].dropna()

    if test_type == "t-test":
        stat, p_val = ttest_ind(high_group, low_group, equal_var=False)
        test_name = "Welch’s T-test"
    elif test_type == "mannwhitney":
        stat, p_val = mannwhitneyu(high_group, low_group, alternative="two-sided")
        test_name = "Mann-Whitney U"
    else:
        raise ValueError("Invalid test_type. Use 't-test' or 'mannwhitney'.")

    result = {
        "test": test_name,
        "statistic": stat,
        "p_value": p_val,
        "conclusion": "Reject H₀" if p_val < 0.05 else "Fail to reject H₀"
    }

    return result



# Bootstrap
def bootstrap_ci(series, stat='mean', n_iterations=1000, ci=95, plot=True, random_state=42):

    # Perform bootstrap resampling to estimate confidence interval of a statistic.

    # Parameters:
    # - series: pandas Series or 1D array-like
    # - stat: 'mean', 'median', or any numpy statistical function
    # - n_iterations: number of bootstrap samples
    # - ci: confidence interval percentage (default: 95)
    # - plot: whether to show a KDE plot of the bootstrapped values
    # - random_state: seed for reproducibility

    # Returns:
    # - Tuple of (lower_bound, upper_bound)

    np.random.seed(random_state)
    data = np.array(series.dropna())
    n_size = len(data)

    if stat == 'mean':
        func = np.mean
    elif stat == 'median':
        func = np.median
    else:
        raise ValueError("Only 'mean' or 'median' supported")

    boot_stats = [func(np.random.choice(data, size=n_size, replace=True)) for _ in range(n_iterations)]
    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)

    if plot:
        sns.histplot(boot_stats, kde=True, color="skyblue")
        plt.axvline(lower, color='red', linestyle='--', label=f"{(100 - ci) / 2:.1f}% CI")
        plt.axvline(upper, color='red', linestyle='--', label=f"{100 - (100 - ci) / 2:.1f}% CI")
        plt.title(f"Bootstrap Distribution of {stat.title()}")
        plt.xlabel(f"{stat.title()} Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return lower, upper

# model split

def split_features_target(df, target_column):
    """Split into X (features) and y (target)"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def split_train_test(X, y, test_size=0.2, random_state=42):
    """Train/test split"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(y_true, y_pred):
    """Prints evaluation metrics"""
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    

# Feature Selection Method
def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importances for tree-based models"""
    importances = model.feature_importances_
    indices = importances.argsort()[-top_n:][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh([feature_names[i] for i in indices], importances[indices])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()
    plt.show()

