"""
Assignment 4 Problem 3: Search for fuzzy matches
July 30, 2025
Germán Montoya
"""

def allMatchesIndices(srch_str, sub_str):
    """
    Find all starting indices of occurrences of a substring within a string.

    Arguments:
        srch_str: the string to search within
        sub_str: the substring to search for

    Returns:
        tuple of starting indices where the substring occurs within the string
    """

    indices = []
    start = 0
    while True:
        pos = srch_str.find(sub_str, start)
        if pos == -1:
            break
        indices.append(pos)
        start = pos + 1
    return tuple(indices)

def fuzzyMatching(subOne, subTwo, len_subOne):
    """
    Finds start positions of potential occurrences of 
    the initial substring that fit a fuzzy match of the 
    second substring.
    
    Arguments:
        subOne: tuple of starting positions of the initial substring
        subTwo: tuple of starting positions of the second substring
        len_subOne: length of the initial substring (int)
    
    Returns:
        tuple of starting positions where fuzzy matches occur
    """

    # Initialize an empty list to store the starting positions of fuzzy matches
    fuzzy_matches = []

    # Iterate through the starting positions of the initial substring
    for start in subOne:
        fuzzy = start + len_subOne + 1  # Add length of first substring and one wildcard
        if fuzzy in subTwo:              # Check if the fuzzy match exists
            fuzzy_matches.append(start)   # If it exists, add the start position to the list
    return tuple(fuzzy_matches)  # Return the tuple of fuzzy match starting positions

# Quick test for fuzzyMatching
# User inputs the string and both substrings
main_str = input("Enter the string to search in: ")
sub1 = input("Enter the first substring: ")
sub2 = input("Enter the second substring: ")

subOne = allMatchesIndices(main_str, sub1)
subTwo = allMatchesIndices(main_str, sub2)

result = fuzzyMatching(subOne, subTwo, len(sub1))
print(f"fuzzyMatching({subOne}, {subTwo}, {len(sub1)}) = {result}")