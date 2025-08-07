"""
Assignment 5 Problem 3: Track how the account balance changes 
due to an initial balance, afixed annual withdrawals, and a rate
of return.
August 6, 2025
Germán Montoya
"""

def finallyRetired(initial_balance, rate_of_return, annual_withdrawal, n, years):
    """
    Calculates the balance of an investment account over a number of years,
    given:
        initial_balance: initial amount in the account (float)
        annual_withdrawal: amount withdrawn each year (float)
        rate_of_return: annual rate of return (as a decimal, e.g., 0.05 for 5%)
        years: number of years to track the balance

    Returns:
        dict: {year: balance}
    """
    results = {}
    balance = initial_balance

    for year in range(1, years + 1):
        balance *= (1 + rate_of_return)  # Apply rate of return
        balance -= annual_withdrawal       # Withdraw fixed amount
        results[year] = balance

    return results

# Example usage
if __name__ == "__main__":
    initial_balance = float(input("Enter the initial account balance: "))
    rate_of_return = float(input("Enter the annual rate of return (e.g., 5 for 5%): ")) / 100
    annual_withdrawal = float(input("Enter the annual withdrawal amount: "))
    years = int(input("Enter the number of years to track: "))
    results = finallyRetired(initial_balance, rate_of_return, annual_withdrawal, None, years)
    for year, balance in results.items():
        print(f"Year {year}: ${balance:,.2f}")

# Testing:
# Black-box: Input initial_balance=23643.75, rate_of_return=5%, annual_withdrawal=7500, years=2
# Output:
# Year 1: $17,325.94
# Year 2: $10,692.23

# White-box: Input initial_balance=130000, rate_of_return=5%, annual_withdrawal=16835.59475, years=10
# Output:
# Year 1: $119,664.41
# Year 2: $108,812.03
# Year 3: $97,417.04
# Year 4: $85,452.29
# Year 5: $72,889.31
# Year 6: $59,698.19
# Year 7: $45,847.50
# Year 8: $31,304.28
# Year 9: $16,033.90
# Year 10: $-0.00

# Summary: Outputs match expected balances for both typical and edge-case

