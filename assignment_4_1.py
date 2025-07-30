"""
Assignment 4 Problem 1: find and count matching strings
July 30, 2025
Germán Montoya
"""

# This function find and counts the number of times a substring appears in a string iteratively.
def countSubstrMatches(srch_str, sub_str):
    """
    Describe what the function does, what type of information goes
    into the arguments, and what is returned.
    """
    count = 0   # Initialize count of matches to zero
    start = 0   # Start position for searching in the string at the beginning
    for i in range(len(srch_str)):      # Loop through the string to find all occurrences
        pos = srch_str.find(sub_str, start)
        if pos == -1:
            break
        count += 1
        start = pos + len(sub_str)     # Move start position to the end of the found substring
    return count                       # Return the count of matches

# This function can be used to find and count the number of times a substring appears in a string recursively.
def countSubstrRecursive(srch_str, sub_str):
    """
    Recursively counts the number of times sub_str appears in srch_str (non-overlapping).
    """
    pos = srch_str.find(sub_str)
    if pos == -1:
        return 0
    return 1 + countSubstrRecursive(srch_str[pos + len(sub_str):], sub_str)


# Example usage
while True:
    test = input("Do you want to find a substring? (yes/no): ").lower() #Loop to ask user if they want to find a substring
    if test == "no":
        print("Thank you for using the program.")
        break
    elif test == "yes":
        srch_str = input("Enter the string to search in: ")
        sub_str = input("Enter the substring to find: ")
        count = countSubstrMatches(srch_str, sub_str)  # Call the function
        print(f"The substring '{sub_str}' appears {count} times in the string.")
        print()  # Add a blank line after each run
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
    print()  # Add a blank line after each run