def get_mapping():
    """Map integer responses to strings for interpretability."""
    return {0: 'Negative', 1: 'Positive'}

def print_stats(df):
    """
    Print the number of presentations and the proportion of self-harm, suicidal ideation and AOD OD cases.
    v2.sh from 18.12.23
    """
    print("The dataset contains %d presentations.\n" % df.shape[0])
    print("SELF-HARM")
    print("Number of presentations:")
    print(df.SH.value_counts(dropna=False).sort_index())
    print()
    print("Proportion of presentations:")
    print(df.SH.value_counts(dropna=False, normalize=True).sort_index()*100)
    print()
    print("_"*80)
    print("SUICIDAL IDEATION")
    print("Number of presentations:")
    print(df.SI.value_counts(dropna=False).sort_index())
    print()
    print("Proportion of presentations:")
    print(df.SI.value_counts(dropna=False, normalize=True).sort_index()*100)
    print("_"*80)
    print("AOD overdose")
    print("Number of presentations:")
    print(df.AOD_OD.value_counts(dropna=False).sort_index())
    print()
    print("Proportion of presentations:")
    print(df.AOD_OD.value_counts(dropna=False, normalize=True).sort_index()*100)
    print()  