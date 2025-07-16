"""
Assignment 2: Basic input and output for finding sum of log(prime) between 2 to n.
July 16, 2025
German Montoya
"""

import math     # Importing math module for logarithm function

while True:
    test = input("Do you want to test a value? (yes/no): ").lower()
    if test == "no":
        print("Thank you for using the program.")
        break
    elif test == "yes":
        primes = []
        n = 2
        number = int(input("Enter a positive integer greater than or equal to 2: "))
        while n < number + 1:
            for i in range(2, n):
                if n % i == 0:
                    break
            else:
                primes.append(n)
            n += 1

        # Calculations and output inside the 'yes' block
        prime_log = [(p, math.log(p)) for p in primes if p >= 2]
        log_sum = sum(log for _, log in prime_log)
        avg_log = log_sum / len(prime_log) if prime_log else 0

        print(f"Total prime numbers found [primes]: {len(primes)}")
        print("Last Prime number in [primes] is:", primes[-1])
        print("Last (prime, log(prime)) in [prime_log] is:", prime_log[-1])
        print("Sum of log(prime) between 2 to n:", log_sum)
        print("Size of prime vector: ", len(primes))






