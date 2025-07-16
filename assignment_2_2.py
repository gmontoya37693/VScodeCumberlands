"""
Assignment 2: Basic input and output for finding sum of log(prime) between 2 to n.
July 16, 2025
German Montoya
"""

import math     # Importing math module for logarithm function
number = 0           # Variable to store the upper limit

while True:
    test = input("Do you want to test a value? (yes/no): ").lower() # Convert input to lowercase for consistency
    # Check if the input is either "yes" or "no"
    if test == "no":
        print("Thank you for using the program.")
        break
    elif test == "yes":
        import math
        primes = []
        n = 2
        number = int(input("Enter a positive integer greater than or equal to 2: "))  # Input for the upper limit
        while n < number + 1:  # Find prime numbers up to n
            for i in range(2, n):
                if n % i == 0:
                    break
            else:
                primes.append(n)  # Append the prime number to the list
            n += 1  # Increment n to check the next number
    
    #create a list of tuples with prime numbers and their logarithm
    prime_log = [(p, math.log(p)) for p in primes if p >= 2]  # Calculate log for each prime number  
    # Calculate the sum of logarithms of prime numbers
    log_sum = sum(log for _, log in prime_log)  # Sum of logarithms
    # Calculate the average of the logarithms
    avg_log = log_sum / len(prime_log) if prime_log else 0
      
    print(f"Total prime numbers found [primes]: {len(primes)}")  # Print the total number of prime numbers found
    print("Last Prime number in [primes] is:", primes[-1])  # Print the last prime number found
    print("Last Prime number in [primelog] is:", primes[-1])  # Print the last prime in prime_log
    print("Sum of log(prime) between 2 to n:", log_sum)
    print("Size of prime vector: ", len(primes))  # Print the total number of prime numbers found






