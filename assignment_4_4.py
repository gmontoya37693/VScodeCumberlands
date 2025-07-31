"""
Assignment 4 Problem 4: find and count matching strings
July 30, 2025
Germán Montoya      
"""

def allMatchesIndices(srch_str, sub_str):
    """
    Finds all start indices of occurrences of sub_str in srch_str.
    Returns a tuple of indices.
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

def fuzzyMatchesOnly(srch_str, sub_str):
    """
    Returns a tuple of start positions of fuzzy matches (one wildcard)
    Does not include exact matches.
    Arguments:
        srch_str: string to search
        sub_str: substring to find fuzzy matches for
    Returns:
        tuple of fuzzy match start positions (excluding exact matches)
    """
    fuzzy_indices = set()           # Define a set to avoid duplicates
    chars = set(srch_str)           # Secure unique characters in the search
    for i in range(len(sub_str)):   # Loop through each character in sub_str
        for c in chars:             # Loop through each character in srch_str
            if c != sub_str[i]:
                # Create a candidate substring with one character replaced
                candidate = sub_str[:i] + c + sub_str[i+1:]
                # Find all indices of the candidate substring in the search string     
                indices = allMatchesIndices(srch_str, candidate)
                # Add the found indices to the fuzzy_indices set
                fuzzy_indices.update(indices)
    exact_indices = set(allMatchesIndices(srch_str, sub_str))   # Find exact matches
    result = tuple(sorted(fuzzy_indices - exact_indices))       # Exclude exact matches from the result
    return result

# --- Testing code  ---
main_str = input("Enter the string to search in: ")
sub_str = input("Enter the substring to find fuzzy matches for: ")
result = fuzzyMatchesOnly(main_str, sub_str)
print(f"fuzzyMatchesOnly({main_str!r}, {sub_str!r}) = {result}")
