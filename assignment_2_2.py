"""
Assignment 2: Basic input and output for finding sum of log(prime) between 2 to n.
July 16, 2025
German Montoya
"""

import math     # Importing math module for logarithm function
primes = [2]      # List to store prime numbers
n = 2           # Start with the first prime number


while True:
    test = input("Do you want to test a value? (yes/no): ").lower() # Convert input to lowercase for consistency
    # Check if the input is either "yes" or "no"
    if test == "no":
        print("Thank you for using the program.")
        break
    elif test == "yes":
        n = int(input("Enter a positive integer greater than or equal to 2: "))  # Input for the upper limit
        while len(primes) < n:  # Find prime numbers up to n
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
    
      
    print(f"Total prime numbers found [primes]: {len(primes)}")  # Print the total number of prime numbers found
    print("Last Prime number in [primes] is:", primes[-1])  # Print the last prime number found




        
