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