"""
Assignment 6 problem 2: Create a Ghost-like game
where players take turns adding letters to a growing 
word fragment, with the goal of avoiding completion 
of a valid word.
August 14, 2025
German Montoya
"""

import random

# -----------------------------------
# Helper functions
# (you don't need to understand this code)

wordlist_file = "words.txt"

def import_wordlist():
    """
    Imports a list of words from external file
    Returns a list of valid words for the game
    Words are all in lowercase letters
    """
    print("Loading word list from file...")
    with open(wordlist_file) as f:                        # call file, read file to list
        wordlist = [word.lower() for word in f.read().splitlines()]
    print("  ", len(wordlist), "words loaded.") 
    return wordlist


def into_dictionary(sequence):
    """
    Returns a dictionary where the keys are elements of the sequence
    and the values are integer counts, for the number of times that
    an element is repeated in the sequence.
    sequence: string or list
    return: dictionary
    """
    # freqs: dictionary (element_type -> int)
    freq = {}
    for x in sequence:
        freq[x] = freq.get(x, 0) + 1
    return freq


# end of helper functions
# -----------------------------------

# Load the word dictionary by assignment the file name to 
# the wordlist variable 
wordlist = import_wordlist()

# Your programming begins here

# -----------------------------------
# Testing:
# - Changed the filename to match assignment standard (problem_6_2.py).
# - Verified word list loads successfully ("83667 words loaded.").
# - Confirmed helper functions (import_wordlist, into_dictionary) work as expected.