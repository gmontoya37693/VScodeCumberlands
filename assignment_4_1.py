"""
Assignment 4 Problem 1: find and count matching strings
July 30, 2025
Germán Montoya

Note: The string library is not required for the find method 
because find is built into Python’s str type.
"""

# This function find and counts the number of times a substring appears in a string iteratively.
def countSubstrMatches(srch_str, sub_str):
    """
    countSubstrMatches function finds number of matches
    of sub_str in srch_str iteratively. It asks the user for a string
    and a substring, and returns the count of matches.
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
    countSubstrRecursive function finds number of matches
    of sub_str in srch_str recursively. It asks the user for a string
    and a substring, and returns the count of matches.
    """
    pos = srch_str.find(sub_str)
    if pos == -1:
        return 0
    return 1 + countSubstrRecursive(srch_str[pos + len(sub_str):], sub_str)


# Example usage
while True:
    test = input("Do you want to find a substring? (yes/no): ").lower()
    if test == "no":
        print("Thank you for using the program.")
        break
    elif test == "yes":
        srch_str = input("Enter the string to search in: ")
        sub_str = input("Enter the substring to find: ")
        while True:
            method = input("Which function do you want to use? (iterative/recursive): ").lower()
            if method == "iterative":
                count = countSubstrMatches(srch_str, sub_str)
                print(f"The substring '{sub_str}' appears {count} times in the string (iterative).")
                break
            elif method == "recursive":
                count = countSubstrRecursive(srch_str, sub_str)
                print(f"The substring '{sub_str}' appears {count} times in the string (recursive).")
                break
            else:
                print("Invalid input. Please enter 'iterative' or 'recursive'.")
        print()  # Add a blank line after each run
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
    print()  # Add a blank line after each run