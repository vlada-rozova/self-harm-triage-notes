from self_harm_triage_notes.config import results_dir
from self_harm_triage_notes.text import count_tokens, count_vocab_tokens_in_data
from scipy.spatial.distance import jensenshannon

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

    plt.rcParams['figure.figsize'] = (df.quarter.nunique() * 12 / 40, 2)

    # Lineplot: Overlap with development set
    sns.lineplot(df.groupby(df.quarter.cat.codes).apply(lambda x: 
                                                        calculate_divergence(x.entities)),
                                                        color=sns.color_palette('tab10')[4], lw=2);
    # Axes limits, ticks, and labels
    # plt.ylim([0, 0.35]);
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
