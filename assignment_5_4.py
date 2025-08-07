from assignment_5_2 import variableInvestor
from assignment_5_3 import finallyRetired

"""
Assignment 5 Problem 4: Estimate maximum yearly expense from 
a retirement account to use all of the money for life expectancy.
August 7, 2025
Germán Montoya
"""

# Imported functions:
# variableInvestor(salary, p_rate, workRate):
#   - Calculates investment balance year by year during working years.
#   - Inputs: salary (float), p_rate (decimal), workRate (list of decimals)
#   - Output: dict {year: balance}
#
# finallyRetired(saved, retiredRate, expensed):
#   - Calculates retirement account balance year by year after retirement.
#   - Inputs: saved (float), retiredRate (list of decimals), expensed (float)
#   - Output: dict {year: balance}

def maximumExpensed(salary, p_rate, workRate, retiredRate, epsilon):
    """
    Uses binary search to estimate the maximum yearly expense from a retirement account
    so the balance is nearly zero at the end of retirement.
    Inputs:
        salary (float): annual salary during working years
        p_rate (float): percentage of salary saved each year (as decimal)
        workRate (list of float): annual rates of return during working years
        retiredRate (list of float): annual rates of return during retirement
        epsilon (float): acceptable margin for ending balance
    Returns:
        float: maximum yearly expense
    """