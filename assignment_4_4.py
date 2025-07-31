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
    Returns a tuple of start positions of fuzzy matches (one character incorrect)
    Does not include exact matches.
    Arguments:
        srch_str: string to search
        sub_str: substring to find fuzzy matches for
    Returns:
        tuple of fuzzy match start positions (excluding exact matches)
    """
    fuzzy_indices = set()
    chars = set(srch_str)
    for i in range(len(sub_str)):
        for c in chars:
            if c != sub_str[i]:
                candidate = sub_str[:i] + c + sub_str[i+1:]
                indices = allMatchesIndices(srch_str, candidate)
                fuzzy_indices.update(indices)
    exact_indices = set(allMatchesIndices(srch_str, sub_str))
    result = tuple(sorted(fuzzy_indices - exact_indices))
    return result

# --- Testing code (comment out before submitting) ---
# main_str = input("Enter the string to search in: ")
# sub_str = input("Enter the substring to find fuzzy matches for: ")
# result = fuzzyMatchesOnly(main_str, sub_str)
# print(f"fuzzyMatchesOnly({main_str!r}, {sub_str!r}) = {result}")

