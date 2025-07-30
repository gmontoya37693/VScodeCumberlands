"""
Assignment 4 Problem 2: Find all start indices of occurrences 
of a substring in a string
July 30, 2025
Germán Montoya
"""

# This function find and counts the number of times a substring 
# appears in a string iteratively.
def allMatchesIndices(srch_str, sub_str):
    """
    allMatchesIndices will find all start indices of occurrences of sub_str 
    in srch_str. It takes two arguments: srch_str (the string to search within),
    and sub_str (the substring to search for). It returns a tuple of the start
    indices of all matches.
    """
    indices = []
    start = 0
    while True:
        pos = srch_str.find(sub_str, start)
        if pos == -1:
            break
        indices.append(pos) # Append the found start index to the list
        start = pos + 1  # Move start by 1 to allow overlapping matches
    return tuple(indices)

# Example usage
# The following code allows the user to input a string and a substring
# Cycles through the process until the user decides to stop.
while True:
    test = input("Do you want to find all substring indices? (yes/no): ").lower()
    if test == "no":
        print("Thank you for using the program.")
        break
    elif test == "yes":
        srch_str = input("Enter the string to search in: ")
        sub_str = input("Enter the substring to find: ")
        indices = allMatchesIndices(srch_str, sub_str)
        print(f"allMatchesIndices({srch_str!r}, {sub_str!r})")
        print(indices)
        print()  # Add a blank line after each run
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
