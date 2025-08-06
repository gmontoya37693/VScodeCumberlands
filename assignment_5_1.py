"""
Assignment 5 Problem 1: use successive approximation to find
the value of an investment given salary, savings, rate, and
duration
August 6, 2025
Germán Montoya      
"""
def fixedInvestor(salary, p_rate, f_rate, years):
    """
    This function returns a dictionary where the key is the year
    and the value is the balance at the end of that year.

    Arguments:
        salary: annual salary
        p_rate: percentage of salary saved each year
        f_rate: fixed interest rate for the investment
        years: number of years to calculate

    Returns:
        dict: {year: balance}
    """
    results = {}
    balance = 0
    for year in range(1, years + 1):
        match_rate = p_rate
        employer_rate = 0.05
        
        balance *= (1 + f_rate)  # Apply interest rate
        results[year] = balance  # Save year and balance in the dictionary
        
    return results