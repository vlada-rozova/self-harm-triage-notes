from self_harm_triage_notes.config import results_dir
from self_harm_triage_notes.text import count_tokens, count_vocab_tokens_in_data
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import *
from sklearn.calibration import calibration_curve

# Pretty plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 12

def calculate_auc(y, y_proba, method, return_all=False):
    """
    Calculate AUC under ROC or PR curves.
    """
    if method=='roc':
        x1, x2, _ = roc_curve(y, y_proba)
    elif method=='pr':
        x2, x1, _ = precision_recall_curve(y, y_proba)
    
    if return_all:
        return x1, x2, auc(x1, x2)
    else:
        return auc(x1, x2)

def plot_curves(y, y_proba, filename=None):
    """
    Plot ROC and PR curves.
    """
    # Plot a histogram of predicted probabilities
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.figure();
    
    sns.histplot(x=y_proba, hue=y, bins=25);
    plt.xlabel("Predicted probabilities");
    
    if filename:
        plt.savefig(results_dir / (filename + " histogram.jpeg"), bbox_inches='tight', dpi=300);
    
    # Plot ROC curves for each fold
    plt.figure();
    
    fpr, tpr, roc_auc = calculate_auc(y, y_proba, 'roc', return_all=True)
    sns.lineplot(x=[0,1], y=[0,1], lw=0.5, color=sns.color_palette()[0], linestyle='--')
    sns.lineplot(x=fpr, y=tpr, estimator=None, sort=False,
                 lw=3, color=sns.color_palette()[2], label="AUC = %0.3f" % roc_auc)
    
    plt.xlim([-0.02, 1.01])
    plt.ylim([-0.01, 1.02])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC curve")
    plt.legend(loc="lower right", title="ROC AUC");
    
    if filename:
        plt.savefig(results_dir / (filename + " ROC.jpeg"), bbox_inches='tight', dpi=300);
        
    # Plot ROC curves for each fold
    plt.figure();
    
    rec, prec, pr_auc = calculate_auc(y, y_proba, 'pr', return_all=True)
    sns.lineplot(x=rec, y=prec, estimator=None, sort=False,
                 lw=3, color=sns.color_palette()[3], label="AUC = %0.3f" % pr_auc)
    
    plt.xlim([-0.02, 1.01])
    plt.ylim([-0.01, 1.02])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower right", title="PR AUC");
    
    if filename:
        plt.savefig(results_dir / (filename + " PR.jpeg"), bbox_inches='tight', dpi=300);

def plot_curves_cv(y, y_proba, cv_generator):
    """
    Plot ROC and PR curves for each CV fold.
    """    
    _, (ax1, ax2) = plt.subplots(2, figsize=(5, 10))
    plt.subplots_adjust(hspace=0.4)
    sns.lineplot(x=[0,1], y=[0,1], lw=0.5, color=sns.color_palette()[0], linestyle='--', ax=ax1)
    
    for _, val_idx in cv_generator:
        
        # Plot ROC curves for each fold
        fpr, tpr, roc_auc = calculate_auc(y.loc[val_idx], y_proba[val_idx], 'roc', return_all=True)
        sns.lineplot(x=fpr, y=tpr, estimator=None, sort=False, label="AUC = %0.2f" % roc_auc, ax=ax1)
        
        # Plot ROC curves for each fold
        rec, prec, pr_auc = calculate_auc(y.loc[val_idx], y_proba[val_idx], 'pr', return_all=True)
        sns.lineplot(x=rec, y=prec, estimator=None, sort=False, label="AUC = %0.2f" % pr_auc, ax=ax2)
        
    ax1.set(xlim=[-0.02, 1.01], ylim=[-0.01, 1.02], 
            xlabel="1 - Specificity", ylabel="Sensitivity",
            title="ROC curve")
    ax1.legend(loc="upper left", title="ROC AUC", bbox_to_anchor=(1.0, 1.0))
    
    ax2.set(xlim=[-0.02, 1.01], ylim=[-0.01, 1.02], 
            xlabel="Recall", ylabel="Precision",
            title="Precision-Recall curve")
    ax2.legend(loc="upper left", title="PR AUC", bbox_to_anchor=(1.0, 1.0))

def plot_calibration_curve(y, y_proba, y_proba_calibrated, filename=None):
    sns.lineplot(x=[0, 1], y=[0, 1], 
             color=sns.color_palette()[0], lw=1, linestyle='--', 
             label="Perfectly calibrated")
    fop, mpv = calibration_curve(y, y_proba, n_bins=30)
    sns.lineplot(x=mpv, y=fop, 
             lw=2, marker='.', markersize=15, 
             color=sns.color_palette()[1],
             label="Uncalibrated");
    fop, mpv = calibration_curve(y, y_proba_calibrated, n_bins=30)
    sns.lineplot(x=mpv, y=fop, 
             lw=2, marker='.', markersize=15, 
             color=sns.color_palette()[2],
             label="Calibrated");
    plt.legend(loc="upper left");
    plt.xlabel("Mean predicted value");
    plt.ylabel("Fraction of positives");
    if filename:
        plt.savefig(results_dir / (filename + " calibration.jpeg"), bbox_inches='tight', dpi=300);

def evaluate_classification(y, y_pred, filename=None):
    """
    Evaluate model performance: print classification report and plot confusion matrix.
    """ 
    # Proportion of instances predcited as positive 
    print("Proportion of labels predicted as positive: %.1f%%" % (y_pred.sum() / y_pred.shape[0] * 100))
    
    # Print classification report
    print("Classification report:")
    print(classification_report(y, y_pred))
    
    # Print PPV, Sensitivity, Specificity
    print("PPV: %.2f, Sensitivity: %.2f, Specificity: %.2f" % (precision_score(y, y_pred), 
                                                               recall_score(y, y_pred), 
                                                               recall_score(y, y_pred, pos_label=0)))
    # Plot confusion matrix
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.figure();
    
    sns.heatmap(confusion_matrix(y, y_pred), 
                annot=True, fmt='d', annot_kws={'size': 16},
                cmap='Blues', cbar=False, 
                xticklabels=("Negative", "Positive"), 
                yticklabels=("Negative", "Positive"))

    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix");
    
    if filename:
        plt.savefig(results_dir / (filename + " conf mat.jpeg"), bbox_inches='tight', dpi=300);

def format_quarter(x):
    """
    Format quarter to replace number with the first month of the quarter
    """
    # The first month of each quarter
    q_start_month = {'1': 'Jan', '2': 'Apr', '3': 'Jul', '4': 'Oct'}
    # Slit into year and quarter number
    year, q = x.split('Q')
    # Map and swap order
    return q_start_month[q] + ' ' + year

def plot_dim_over_time(df, title):
    """Plot the percentage of vocabulary overlap per quarter."""
    def calculate_dimensionality(notes):
        """Calculate the number of unique valid tokens."""
        return len(count_tokens(notes, valid=True))
    
    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 3)

    palette = sns.color_palette('tab10')

    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_dimensionality(x.preprocessed_triage_note)),
                                                        color=palette[0], lw=2, 
                                                        label="Before text normalisation");
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_dimensionality(x.entities)),
                                                        color=palette[2], lw=2, 
                                                        label="After text normalisation");
    # Axes limits, ticks, and labels
    plt.ylim([100, 31000]);
    plt.xticks(rotation=45, 
               ticks=range(0, df.quarter.nunique(), 2), 
               labels=[format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                       if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("# unique tokens");
    plt.legend(loc='lower right');

    # Title
    plt.title(title);

    # Save plot
    plt.savefig(results_dir / (title + " unique tokens per quarter.jpeg"), bbox_inches='tight', dpi=300);

def plot_token_overlap_over_time(df, vocab, title):
    """
    Plot the percentage of vocabulary overlap per quarter.
    """
    def calculate_token_overlap(notes):
        """
        Calculate the proportion of tokens that are in the vocab.
        """
        counts = count_tokens(notes, valid=True)

        return sum([counts[t]>0 for t in vocab]) / len(counts) * 100
    
    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 3)

    palette = sns.color_palette('tab10')

    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_token_overlap(x.preprocessed_triage_note)),
                                                        color=palette[0], lw=2, 
                                                        label="Before text normalisation");
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_token_overlap(x.entities)),
                                                        color=palette[2], lw=2, 
                                                        label="After text normalisation");
    # Axes limits, ticks, and labels
    plt.ylim([-5, 105]);
    plt.xticks(rotation=45, 
               ticks=range(0, df.quarter.nunique(), 2), 
               labels=[format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                       if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("% token overlap");
    plt.legend(loc='lower right');

    # Title
    plt.title(title);

    # Save plot
    plt.savefig(results_dir / (title + " token overlap per quarter.jpeg"), bbox_inches='tight', dpi=300);

def plot_dim_reduction_over_time(df, col1, col2, title):
    """
    Plot the percentage of vocabulary overlap per quarter.
    """
    def calculate_dim_reducton(before, after):
        """
        Calculate the reduction in dimensionality.
        """
        tokens_before = count_tokens(before, valid=True)
        tokens_after = count_tokens(after, valid=True)
        return 100 - len(tokens_after) * 100 / len(tokens_before)
    
    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 2)

    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_dim_reducton(x[col1], x[col2])),
                                                        color=sns.color_palette('tab10')[4], lw=2);
    # Axes limits, ticks, and labels
    plt.ylim([0, 100]);
    plt.xticks(rotation=45, 
               ticks=range(0, df.quarter.nunique(), 2), 
               labels=[format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                       if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("% reduction");

    # Title
    plt.title(title);

    # Save plot
    plt.savefig(results_dir / (title + " dim reduction per quarter.jpeg"), bbox_inches='tight', dpi=300);

def plot_selected_fts_over_time(df, fts, title):
    """
    Plot the number of available features used as input for the classifier.
    """
    def calculate_available_fts(notes):
        """Calculate the number of available features."""
        counts = count_vocab_tokens_in_data(notes, fts)
        return len([k for k, v in counts.items() if v > 0])
    
    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 3)
    
    color = sns.color_palette('tab10')[1]

    sns.lineplot(x=range(0, df.quarter.nunique()), 
                 y=len(fts), 
                 color=color, ls='--', 
                 label="Selected")
    
    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_available_fts(x.entities)),
                                                        color=color, lw=2, 
                                                        label="Available");
    # Axes limits, ticks, and labels
    plt.ylim([400, 655]);
    plt.xticks(rotation=45, 
               ticks=range(0, df.quarter.nunique(), 2), 
               labels=[format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                       if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("# features");
    plt.legend(loc='lower right')

    # Title
    plt.title(title);

    # Save plot
    plt.savefig(results_dir / (title + " selected features per quarter.jpeg"), bbox_inches='tight', dpi=300);

def plot_divergence_over_time(df, dev_counts, selected_features, title):
    """
    Plot JS divergence in the frequency of selected features from the development set.
    """
    def calculate_divergence(x):
        """Calculate divergence in the counts of selected features."""
        counts = count_vocab_tokens_in_data(x, selected_features)
        return jensenshannon([v for _,v in sorted(dev_counts.items())], [v for _,v in sorted(counts.items())])

    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 3)

    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_divergence(x.entities)),
                                                        color=sns.color_palette('tab10')[4], lw=2);
    # Axes limits, ticks, and labels
    plt.ylim([0.05, 0.31]);
    plt.xticks(rotation=45, 
                ticks=range(0, df.quarter.nunique(), 2), 
                labels=[format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                        if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("JS divergence");

    # Title
    plt.title(title);

    # Save plot
    plt.savefig(results_dir / (title + " js divergence per quarter.jpeg"), bbox_inches='tight', dpi=300);
