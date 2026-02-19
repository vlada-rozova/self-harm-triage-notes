import numpy as np
import pandas as pd
from self_harm_triage_notes.text_utils import count_tokens, count_vocab_tokens_in_data
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer 
from self_harm_triage_notes.dev_utils import get_stopwords

# Pretty plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Configuration constants
DEFAULT_FIGURE_SIZE = (6, 4)
DEFAULT_POS_LABEL = 1
DEFAULT_CONFIDENCE_LEVEL = 0.95

def _format_quarter(x):
    """Format quarter to replace number with the first month of the quarter"""
    # The first month of each quarter
    q_start_month = {'1': 'Jan', '2': 'Apr', '3': 'Jul', '4': 'Oct'}
    # Slit into year and quarter number
    year, q = x.split('Q')
    # Map and swap order
    return q_start_month[q] + ' ' + year

def plot_presentations_over_time(df, title, palette, annotate_dev=True, results_dir=None):
    """
    Plot the number of presentations and SH/SI rates per quarter.
    """
    # Create subplots
    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 6)
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Barplot: Numper of ED presentations
    sns.countplot(x='quarter', data=df, 
                    color=palette[0], alpha=0.4,
                    ax=ax1, legend=False);
    
    # Axes limits, ticks, and labels
    ax1.set_xticks(rotation=45, 
                   ticks=range(0, df.quarter.nunique(), 2), 
                   labels=[_format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                           if q.endswith('1') or q.endswith('3')]);
    ax1.set_xlabel("Arrival date");
    ax1.set_ylabel("# ED presentations");

    # Lineplot: SI rate per quarter
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                    x.SI.cat.codes.sum() / x.shape[0] * 100), 
                                                    color=palette[-2], lw=2, 
                                                    label="Suicidal ideation", ax=ax2);
    # Lineplot: SH rate per quarter
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                    x.SH.cat.codes.sum() / x.shape[0] * 100), 
                                                    color=palette[-1], lw=2, 
                                                    label="Self-harm", ax=ax2);
    
    # Vertical line: start of COVID-19 pandemic
    plt.axvline(32.5, 0, 1, color='#53585F', ls='--', label="Start of COVID-19 pandemic");

    if annotate_dev:
        # Horisontal line: dev and test sets
        plt.plot([-0.5, 23.5], [2.2, 2.2], marker='s', markevery=True, color='#53585F', label="Development and test sets");

    # Axes limits, ticks, and labels
    ax2.set_ylim([0, 2.8]);
    ax2.set_ylabel("% cases");
    ax2.legend(loc='lower left');

    # Title
    plt.title(title);

    # Save plot
    if results_dir is not None:
        plt.savefig(results_dir / (title + " presentations and cases per quarter.png"), bbox_inches='tight', dpi=300);

def plot_length_over_time(df, title, palette, annotate_dev=True, results_dir=None):
    """Plot character length of triage notes over time"""

    plt.figure(figsize=(df.quarter.nunique() * 12 / 40, 3))

    sns.lineplot(x=df.quarter.cat.codes, y=df.length, hue=df.SH, 
                 estimator='mean', errorbar='sd', 
                 lw=2, palette=palette);
    
    if annotate_dev:
        # Horisontal line: dev and test sets
        plt.plot([0, 23], [300, 300], marker='s', markevery=True, color='#53585F', label="Development and test sets");
    
    # Axes limits, ticks, and labels
    plt.ylim([50, 600]);
    plt.xticks(rotation=45, 
               ticks=range(0, df.quarter.nunique(), 2), 
               labels=[_format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                       if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("Character length");
    plt.legend(title="True class", loc='upper left');

    # Title
    plt.title(title);

    # Save plot
    if results_dir is not None:
        plt.savefig(results_dir / (title + " char length per quarter.png"), bbox_inches='tight', dpi=300);

def plot_probabilities(y, y_proba, palette, threshold=None, results_dir=None, filename=None):

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)

    ax = sns.histplot(x=y_proba.iloc[:, DEFAULT_POS_LABEL], hue=y, binwidth=0.035, palette=palette);
    ax.get_legend().set_title("True class")

    if threshold is not None:
        plt.axvline(threshold, 0, 1, ls='--', color='#53585F');
    
    if results_dir is not None and filename is not None:
        plt.savefig(results_dir / ("Predicted probability upper half " + filename + ".png"), bbox_inches='tight', dpi=300)

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)

    sns.histplot(x=y_proba.iloc[:, DEFAULT_POS_LABEL], hue=y, binwidth=0.035, palette=palette, legend=False)

    if threshold is not None:
        plt.axvline(threshold, 0, 1, ls='--', color='#53585F');
    
    plt.ylim([0, 1200])    
    plt.xlabel("Predicted probability")

    if results_dir is not None and filename is not None:
        plt.savefig(results_dir / ("Predicted probability lower half " + filename + ".png"), bbox_inches='tight', dpi=300)

def plot_score_over_time(scores_overall, scores_per_quarter, palette, dataset_name, score_name, title, estimated_overall=None, results_dir=None):
    
        n_quarters = scores_per_quarter.quarter.nunique()

        plt.figure(figsize=(n_quarters * 12 / 28, 4))

        # Estimated from the development set
        if estimated_overall is not None:
            sns.lineplot(x=range(0, n_quarters), y=estimated_overall, 
                        estimator=None,
                        color='dimgrey', ls='-.', lw=2, label="Test set (overall estimate)")

        # Overall score
        sns.lineplot(x=range(0, n_quarters), y=scores_overall.value.item(), 
                     estimator=None,
                     color=palette[score_name], ls='--', lw=2, label=f"{dataset_name} set (overall estimate)")
        # Temporal scores
        sns.lineplot(x=scores_per_quarter.quarter.cat.codes, y=scores_per_quarter.value, 
                     estimator='mean', errorbar=('pi', DEFAULT_CONFIDENCE_LEVEL*100),
                     color=palette[score_name], lw=2, label=f"{dataset_name} set (per quarter)")
        
        # Axes limits, ticks, and labels
        plt.ylim([0, 1.01]);
        plt.xticks(rotation=45, 
                   ticks=range(0, n_quarters, 2), 
                   labels=[_format_quarter(q) for q in scores_per_quarter.quarter.cat.categories.astype(str) 
                           if q.endswith('1') or q.endswith('3')]);
        plt.xlabel("Arrival date");
        plt.ylabel("Score");
        plt.legend(title=score_name, loc="lower right");
        plt.title(title);

        if results_dir is not None:
            plt.savefig(results_dir / ("Score over time " + dataset_name + " " +  score_name + ".png"), bbox_inches='tight', dpi=300)

def plot_confusion_matrix_all_labels(y, y_pred, color=None, results_dir=None, filename=None):

    if color:
        cmap = sns.light_palette(color, as_cmap=True)

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)

    M = pd.DataFrame(index = y.cat.categories, columns=y_pred.cat.categories)
    for i in y_pred.cat.categories:
        for j in y.cat.categories:
            M.loc[j, i] = len(y[(y_pred == i) & (y == j)])

    sns.heatmap(M.astype(int), 
                annot=True, fmt='d', annot_kws={'size': 16},
                yticklabels=['Not self-harm\nor suicidal\nideation', 'Suicidal\nideation', 'Self-harm'],
                cmap=cmap, cbar=False)

    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix");

    if results_dir is not None and filename is not None:
        plt.savefig(results_dir / ("Confusion matrix for all labels " + filename + ".png"), bbox_inches='tight', dpi=300)

def plot_dim_over_time(df, title, palette, results_dir=None):
    """Plot the percentage of vocabulary overlap per quarter."""
    def calculate_dimensionality(notes):
        """Calculate the number of unique valid tokens."""
        return len(count_tokens(notes, valid=True))
    
    plt.figure(figsize=(df.quarter.nunique() * 12 / 40, 3))

    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_dimensionality(x.preprocessed_triage_note)),
                                                        color=palette[1], lw=2, 
                                                        label="Before text normalisation");
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_dimensionality(x.entities)),
                                                        color=palette[3], lw=2, 
                                                        label="After text normalisation");
    # Axes limits, ticks, and labels
    plt.ylim([100, 31000]);
    plt.xticks(rotation=45, 
               ticks=range(0, df.quarter.nunique(), 2), 
               labels=[_format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                       if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("# unique tokens");
    plt.legend(loc='lower right');

    # Title
    plt.title(title);

    # Save plot
    if results_dir is not None:
        plt.savefig(results_dir / ("Unique tokens per quarter " + title + ".png"), bbox_inches='tight', dpi=300)

def plot_token_overlap_over_time(df, vocab, title, palette, results_dir=None):
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

    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_token_overlap(x.preprocessed_triage_note)),
                                                        color=palette[1], lw=2, 
                                                        label="Before text normalisation");
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_token_overlap(x.entities)),
                                                        color=palette[3], lw=2, 
                                                        label="After text normalisation");
    # Axes limits, ticks, and labels
    plt.ylim([-5, 105]);
    plt.xticks(rotation=45, 
               ticks=range(0, df.quarter.nunique(), 2), 
               labels=[_format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                       if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("% token overlap");
    plt.legend(loc='lower right');

    # Title
    plt.title(title);

    # Save plot
    if results_dir is not None:
        plt.savefig(results_dir / ("Token overlap per quarter " + title + ".png"), bbox_inches='tight', dpi=300);

def plot_dim_reduction_over_time(df, col1, col2, title, palette, results_dir=None):
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
    
    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 3)

    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_dim_reducton(x[col1], x[col2])),
                                                        color=palette[2], lw=2);
    # Axes limits, ticks, and labels
    plt.ylim([0, 100]);
    plt.xticks(rotation=45, 
               ticks=range(0, df.quarter.nunique(), 2), 
               labels=[_format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                       if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("% reduction");

    # Title
    plt.title(title);

    # Save plot
    if results_dir is not None:
        plt.savefig(results_dir / ("Dimensionality reduction per quarter" + title + ".png"), bbox_inches='tight', dpi=300);

def plot_selected_fts_over_time(df, selected_features, title, palette, results_dir=None):
    """
    Plot the number of available features used as input for the classifier.
    """
    def calculate_available_fts(x):
        ft_counts = vectorizer.fit_transform(x)
        return (ft_counts.sum(axis=0) > 0).sum()
    
    # Initialise the tokenizer
    vectorizer = TfidfVectorizer(stop_words=get_stopwords(), token_pattern=r'\S+', ngram_range=(1,3), vocabulary=selected_features)
    # Fit the vecotriser
    vectorizer.fit(df.entities)
    
    n_quarters = df.quarter.nunique()
    plt.rcParams['figure.figsize'] = (n_quarters * 12 / 40, 3)
    
    sns.lineplot(x=range(0, n_quarters), 
                 y=len(selected_features), 
                 color=palette[0], ls='--', 
                 label="Selected features")
    
    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_available_fts(x.entities)),
                                                        color=palette[0], lw=2, 
                                                        label="Available features");
    # Axes limits, ticks, and labels
    plt.ylim([0, 2000]);
    plt.xticks(rotation=45, 
               ticks=range(0, n_quarters, 2), 
               labels=[_format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                       if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("# features");
    plt.legend(loc='lower right')

    # Title
    plt.title(title);

    # Save plot
    if results_dir is not None:
        plt.savefig(results_dir / ("Selected features per quarter" + title + ".png"), bbox_inches='tight', dpi=300);

def plot_divergence_over_time(df_dev, df, selected_features, title, palette, results_dir=None):
    """
    Plot JS divergence in the frequency of selected features from the development set.
    """
    def calculate_divergence(x):
        ft_counts = vectorizer.fit_transform(x).toarray().sum(axis=0)
        return jensenshannon(dev_counts, ft_counts)
    
    # Initialise the tokenizer
    vectorizer = TfidfVectorizer(stop_words=get_stopwords(), token_pattern=r'\S+', ngram_range=(1,3), vocabulary=selected_features)
    # Fit the vecotriser
    dev_counts = vectorizer.fit_transform(df_dev.entities).toarray().sum(axis=0)

    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 3)

    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_divergence(x.entities)),
                                                        color=palette[0], lw=2);
    # Axes limits, ticks, and labels
    plt.ylim([0.05, 0.31]);
    plt.xticks(rotation=45, 
                ticks=range(0, df.quarter.nunique(), 2), 
                labels=[_format_quarter(q) for q in df.quarter.cat.categories.astype(str) 
                        if q.endswith('1') or q.endswith('3')]);
    plt.xlabel("Arrival date");
    plt.ylabel("JS divergence");

    # Title
    plt.title(title);

    # Save plot
    if results_dir is not None:
        plt.savefig(results_dir / (title + " js divergence per quarter.png"), bbox_inches='tight', dpi=300);
