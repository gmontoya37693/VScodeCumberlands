"""
Assignment 1: this is a basic input and output file
June 30, 2025
German Montoya
"""

name = input("What is your name: ")
print(name)
while True:
    test = input("Do you know Python? (yes/no): ").lower() # Convert input to lowercase for consistency
    # Check if the input is either "yes" or "no"
    if test == "yes" or test == "no":
        break
    print("Please answer with 'yes' or 'no'.")
print("Thanks for answering my questions, {}.".format(name))
