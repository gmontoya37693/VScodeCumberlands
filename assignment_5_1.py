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
        match_rate = p_rate         # match_rate is the same as p_rate 
        employer_rate = 0.05        # employer contributes 5% of salary
        # Calculate total savings for the year
        yearly_savings = salary * (f_rate + employer_rate + match_rate) 
        balance *= (1 + f_rate)     # Apply the match rate to the balance
        balance += yearly_savings   # Add the yearly savings to the balance
        results[year] = balance     # Save year and balance in the dictionary
        
    return results

# Example usage
while True:
    test = input("Do you want to estimate your investment? (yes/no): ").lower()
    if test == "no":
        print("Thank you for coming.")
        break
    elif test == "yes":
        salary = float(input("Enter your annual salary: "))
        p_rate = float(input("Enter your contribution percentage (as a decimal): "))
        f_rate = float(input("Enter the fixed interest rate (as a decimal): "))
        years = int(input("Enter the number of years for the investment: "))
        results = fixedInvestor(salary, p_rate, f_rate, years)
        for year, balance in results.items():
            print(f"Year {year}: ${balance:,.2f}")