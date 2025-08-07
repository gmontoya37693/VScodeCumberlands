"""
Assignment 5 Problem 3: Track how the account balance changes 
due to an initial balance, afixed annual withdrawals, and a rate
of return.
August 6, 2025
Germán Montoya
"""

def finallyRetired(saved, rate_of_return, expensed, years):
    """
    Calculates the balance of a retirement account over a number of years,
    given:
        saved: initial amount in the account (float)
        rate_of_return: fixed annual rate of return (as decimal, e.g., 0.05 for 5%)
        expensed: amount withdrawn each year (float)
        years: number of years (int)

    Returns:
        dict: {year: balance}
    """
    results = {}
    balance = saved
    for year in range(1, years + 1):
        balance *= (1 + rate_of_return)
        balance -= expensed
        results[year] = balance
    return results

# Example usage
if __name__ == "__main__":
    saved = float(input("Enter the initial account balance: "))
    rate_of_return = float(input("Enter the fixed annual rate of return (e.g., 5 for 5%): ")) / 100
    expensed = float(input("Enter the annual withdrawal amount: "))
    years = int(input("How many years do you want to track? "))
    results = finallyRetired(saved, rate_of_return, expensed, years)
    for year, balance in results.items():
        print(f"Year {year}: ${balance:,.2f} (Rate: {rate_of_return*100:.2f}%)")

# Testing:
# Black-box: Input saved=23643.75, rate_of_return=0.05, expensed=7500, years=2
# Output:
# Year 1: $17,325.94 (Rate: 5.00%)
# Year 2: $10,692.23 (Rate: 5.00%)

# White-box: Input saved=130000, rate_of_return=0.05, expensed=16835.59475, years=10
# Output:
# Year 1: $119,664.41 (Rate: 5.00%)
# Year 2: $108,812.03 (Rate: 5.00%)
# Year 3: $97,417.04 (Rate: 5.00%)
# Year 4: $85,452.29 (Rate: 5.00%)
# Year 5: $72,889.31 (Rate: 5.00%)
# Year 6: $59,698.19 (Rate: 5.00%)
# Year 7: $45,847.50 (Rate: 5.00%)
# Year 8: $31,304.28 (Rate: 5.00%)
# Year 9: $16,033.90 (Rate: 5.00%)
# Year 10: $-0.00 (Rate: 5.00%)

# Summary: Outputs match expected balances for both typical and edge-case

