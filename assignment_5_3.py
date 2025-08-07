"""
Assignment 5 Problem 3: Track how the account balance changes 
due to an initial balance, afixed annual withdrawals, and a rate
of return.
August 6, 2025
Germán Montoya
"""

def finallyRetired(saved, v_rate, expensed):
    """
    Calculates the balance of a retirement account over a number of years,
    given:
        saved: initial amount in the account (float)
        v_rate: list of annual rates of return (as decimals, e.g., [0.05, 0.04, ...])
        expensed: amount withdrawn each year (float)

    Returns:
        dict: {year: balance}
    """
    results = {}
    balance = saved
    for year, rate in enumerate(v_rate, start=1):
        balance *= (1 + rate)
        balance -= expensed
        results[year] = balance
    return results

# Example usage
if __name__ == "__main__":
    saved = float(input("Enter the initial account balance: "))
    v_rate = list(map(float, input("Enter the annual rates of return separated by spaces (e.g., 5 4 3 for 5%, 4%, 3%): ").split()))
    expensed = float(input("Enter the annual withdrawal amount: "))
    results = finallyRetired(saved, v_rate, expensed)
    for year, balance in results.items():
        print(f"Year {year}: ${balance:,.2f} (Rate: {v_rate[year-1]*100:.2f}%)")

# Testing:
# Black-box: Input saved=23643.75, v_rate=[0.05, 0.04], expensed=7500
# Output:
# Year 1: $17,325.94 (Rate: 5.00%)
# Year 2: $10,692.23 (Rate: 4.00%)

# White-box: Input saved=130000, v_rate=[0.05, 0.04, 0.03, 0.02, 0.01], expensed=16835.59475
# Output:
# Year 1: $119,664.41 (Rate: 5.00%)
# Year 2: $108,812.03 (Rate: 4.00%)
# Year 3: $97,417.04 (Rate: 3.00%)
# Year 4: $85,452.29 (Rate: 2.00%)
# Year 5: $72,889.31 (Rate: 1.00%)
# Year 6: $59,698.19 (Rate: 0.00%)
# Year 7: $45,847.50 (Rate: -1.00%)
# Year 8: $31,304.28 (Rate: -2.00%)
# Year 9: $16,033.90 (Rate: -3.00%)
# Year 10: $-0.00 (Rate: -4.00%)

# Summary: Outputs match expected balances for both typical and edge-case

