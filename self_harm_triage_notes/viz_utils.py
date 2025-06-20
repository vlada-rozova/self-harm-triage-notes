from self_harm_triage_notes.config import results_dir
from self_harm_triage_notes.text_utils import count_tokens, count_vocab_tokens_in_data
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import *

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

def plt_predicted_proba(df, title):
    """Plot distributions of predicted probabilities"""
    # Upper half
    # Histogram: predicted probabilities
    ax = sns.histplot(x='probability', hue=df.SH.map({0: 'Negative', 1: 'Positive'}), data=df,
                    binwidth=0.035, palette={'Negative': sns.color_palette('tab10')[7], 
                                            'Positive': sns.color_palette('tab10')[3]});

    # Vertical line: proability threshold
    plt.axvline(0.36, 0, 1, color=sns.color_palette('tab20c')[-4], ls='--');
    ax.get_legend().set_title("Self-harm")
    ax.set_title(title);
    # Save plot
    plt.savefig(results_dir / (title + " predicted proba upper.jpeg"), bbox_inches='tight', dpi=300);

    plt.figure();
    sns.histplot(x='probability', hue='SH', data=df,
                    binwidth=0.035, palette={0: sns.color_palette('tab10')[7], 
                                            1: sns.color_palette('tab10')[3]}, legend=False);
    plt.axvline(0.36, 0, 1, color=sns.color_palette('tab20c')[-4], ls='--');
    plt.ylim([0, 1200]);
    plt.xlabel("Predicted probability");  
    # Save plot
    plt.savefig(results_dir / (title + " predicted proba lower.jpeg"), bbox_inches='tight', dpi=300);

def format_quarter(x):
    """Format quarter to replace number with the first month of the quarter"""
    # The first month of each quarter
    q_start_month = {'1': 'Jan', '2': 'Apr', '3': 'Jul', '4': 'Oct'}
    # Slit into year and quarter number
    year, q = x.split('Q')
    # Map and swap order
    return q_start_month[q] + ' ' + year

def plot_length_over_time(df, title, annotate_dev=True):
    """Plot character length of triage notes over time"""
    
    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 3)

    sns.lineplot(x='quarter', y='length', hue=df.SH.map({0: 'Negative', 1: 'Positive'}), data=df, 
                 estimator='mean', errorbar='sd', 
                 lw=2, palette={'Negative': sns.color_palette('tab10')[7], 
                                'Positive': sns.color_palette('tab10')[3]})
    
    if annotate_dev:
        # Horisontal line: dev and test sets
        plt.plot([0, 23], [300, 300], marker='s', markevery=True, color=sns.color_palette('tab20c')[-4]);
    
    # Axes limits, ticks, and labels
    plt.ylim([50, 600]);
    plt.xticks(rotation=45, 
               ticks=range(0, df.quarter.nunique(), 2), 
               labels=[format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                       if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("Character length");
    plt.legend(title='Self-harm');

    # Title
    plt.title(title);

    # Save plot
    plt.savefig(results_dir / (title + " char length per quarter.jpeg"), bbox_inches='tight', dpi=300);

def plot_scores_over_time(df, title):
    """Plot PR AUC, PPV, Sensitivity, and Specificity per quarter"""
    
    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 6)
    
    palette = sns.color_palette("tab10")
    
    # PR AUC
    sns.lineplot(x=range(0, df.quarter.nunique()), 
                 y=calculate_auc(df.SH, df.probability, method='pr'), 
                 color=palette[4], ls='--', 
                 label="PR AUC (overall)")
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_auc(x.SH, x.probability, method='pr')),
                                                        color=palette[4], lw=2, 
                                                        label="PR AUC (per quarter)");
    # PPV
    sns.lineplot(x=range(0, df.quarter.nunique()), 
                 y=precision_score(df.SH, df.prediction, zero_division=1), 
                 color=palette[0], ls='--', 
                 label="PPV (overall)")
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        precision_score(x.SH, x.prediction, zero_division=1)),
                                                        color=palette[0], lw=2, 
                                                        label="PPV (per quarter)");
    # Sensitivity
    sns.lineplot(x=range(0, df.quarter.nunique()), 
                 y=recall_score(df.SH, df.prediction), 
                color=palette[2], ls='--', 
                label="Sensitivity (overall)")
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        recall_score(x.SH, x.prediction)),
                                                        color=palette[2], lw=2, 
                                                        label="Sensitivity (per quarter)");
    # Specificity
    sns.lineplot(x=range(0, df.quarter.nunique()), 
                y=recall_score(df.SH, df.prediction, pos_label=0), 
                color=palette[1], ls='--', 
                label="Specificity (overall)")
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        precision_score(x.SH, x.prediction, pos_label=0)),
                                                        color=palette[1], lw=2, 
                                                        label="Specificity (per quarter)");
    # Axes limits, ticks, and labels
    plt.ylim([0, 1.01]);
    plt.xticks(rotation=45, 
                ticks=range(0, df.quarter.nunique(), 2), 
                labels=[format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                        if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("Performance metric");
    plt.legend(loc="lower right");

    # Title
    plt.title(title);

    # Save plot
    plt.savefig(results_dir / (title + " merics per quarter.jpeg"), bbox_inches='tight', dpi=300);

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
