"""
Assignment 4 Problem 2: Count occurrences of a substring in a string
July 30, 2025
Germán Montoya
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

# create a function to count occurrences, and return a tuple of the 
# start indices of all matches.
def allMatchesIndices(srch_str, sub_str):
    """
    allMatchesIndices will find all start indices of occurrences of sub_str 
    in srch_str. It takes two arguments: srch_str (the string to search within),
    and sub_str (the substring to search for). It returns a tuple of the start
    indices of all matches.
    """
    
