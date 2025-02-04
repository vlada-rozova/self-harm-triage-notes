def print_stats(df):
    """
    Print the number of presentations and the proportion of self-harm, suicidal ideation and AOD OD cases.
    v1 from 13.12.23
    """
    print("The dataset contains %d presentations.\n" % df.shape[0])
    print("SELF-HARM")
    print("Number of presentations:")
    print(df.SH.value_counts(dropna=False))
    print()
    print("Proportion of presentations:")
    print(df.SH.value_counts(dropna=False, normalize=True)*100)
    print()
    print("_"*80)
    print("SUICIDAL IDEATION")
    print("Number of presentations:")
    print(df.SI.value_counts(dropna=False))
    print()
    print("Proportion of presentations:")
    print(df.SI.value_counts(dropna=False, normalize=True)*100)
    print("_"*80)
    print("AOD overdose")
    print("Number of presentations:")
    print(df.AOD_OD.value_counts(dropna=False))
    print()
    print("Proportion of presentations:")
    print(df.AOD_OD.value_counts(dropna=False, normalize=True)*100)
    print()  