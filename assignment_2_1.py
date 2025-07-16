"""
Assignment 1: Basic input and output for finding prime numbers.
July 16, 2025
German Montoya
"""

prime_count = 0  # Prime counter
prime = []       # List to store prime numbers
n = 2            # Start with the first prime number

# Function to find prime numbers in order
while len(prime) < 450:     # Find 450 prime numbers
    for i in range(2, n):
        if n % i == 0:
            break
    else:
        prime.append(n)     # Append the prime number to the list
        prime_count += 1    # Increment the prime counter
        if prime_count % 150 == 0:
            print(f"Found {prime_count} prime numbers so far")  # Print progress every 150 primes

    n += 1  
    
print(f"Total prime numbers found: {prime_count}")
print("Primer number 450:", prime[-1])  # Print the 450th prime number
print(prime)  # Print the list of prime numbers
print(len(prime))  # Print the total number of prime numbers found  