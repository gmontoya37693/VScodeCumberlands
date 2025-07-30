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
