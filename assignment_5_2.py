"""
Assignment 5 Problem 2: use successive approximation to find
the value of an investment given salary, savings, and a list of
variable growth rates
August 6, 2025
Germán Montoya
"""

def variableInvestor(salary, p_rate, v_rate):
    """
    Calculates the balance of an investment for each year, given:
        salary: annual salary (float)
        p_rate: percentage of salary saved each year (as a decimal, e.g., 0.05 for 5%)
        v_rate: list of growth rates for each year (as decimals, e.g., [0.05, 0.04, ...])

    Returns:
        dict: {year: balance}
    """
    results = {}
    balance = 0
    employer_rate = 0.05  # employer contributes 5% of salary
    match_rate = p_rate   # employer matches employee contribution

    for year, rate in enumerate(v_rate, start=1):
        yearly_savings = salary * (p_rate + match_rate + employer_rate)
        balance *= (1 + rate)  # Apply variable growth rate for the year
        balance += yearly_savings
        results[year] = balance
    return results

# Example usage
if __name__ == "__main__":
    salary = float(input("Enter your annual salary: "))
    p_rate = float(input("Enter your contribution percentage (e.g., 5 for 5%): ")) / 100
    v_rate = []
    years = int(input("Enter the number of years for the investment: "))
    for i in range(1, years + 1):
        rate = float(input(f"Enter the growth rate for year {i} (e.g., 5 for 5%): ")) / 100
        v_rate.append(rate)
    results = variableInvestor(salary, p_rate, v_rate)
    for year, balance in results.items():
        print(f"Year {year}: ${balance:,.2f} (Growth rate: {v_rate[year-1]*100:.2f}%)")