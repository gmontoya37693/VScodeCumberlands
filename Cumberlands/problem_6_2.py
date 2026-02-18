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

# Check if a fragment is a valid prefix
def is_valid_prefix(fragment, wordlist):
    """
    Returns True if fragment is a prefix of any word in wordlist,
    to check if still possible to extend the fragment into a valid word.
    Otherwise, player loses.
    fragment: string
    wordlist: list of strings
    """
    for word in wordlist:
        if word.startswith(fragment):
            return True
    return False

# Get player names
def get_player_names():
    player1 = input("Enter name for Player 1: ")
    player2 = input("Enter name for Player 2: ")
    return player1, player2

# Check current player to switch turns
def next_player(current, player1, player2):
    return player2 if current == player1 else player1

# Check if fragment is a complete word, and make 
# sure its lower case
def get_valid_letter(player_name):
    """
    Prompts the player for a single alphabetic character and returns it in lowercase.
    Keeps asking until a valid input is received.
    """
    while True:
        letter = input(f"{player_name}, enter your letter: ").lower()
        if len(letter) == 1 and letter.isalpha():
            return letter
        print("Invalid input. Please enter a single alphabetic character.")

# end of helper functions
# -----------------------------------

# Load the word dictionary by assignment the file name to 
# the wordlist variable 
wordlist = import_wordlist()
while True:
    # Start the game asking for player names
    # and initializing fragment empty
    player1, player2 = get_player_names()
    current_player = player1
    fragment = ""
    while True:
        print("Current fragment:", fragment)
        if not is_valid_prefix(fragment, wordlist):
            print("No valid words found. Player", current_player, "loses!")
            break
        letter = get_valid_letter(current_player)
        fragment += letter
        # Check if fragment is a complete word (longer than 3 letters)
        if len(fragment) > 3 and fragment in wordlist:
            print(f"'{fragment}' is a complete word. Player {current_player} loses!")
            break
        current_player = next_player(current_player, player1, player2)
    if input("Play another round? (y/n): ").lower() != 'y':
        break

# Your programming begins here

# -----------------------------------
# Testing:
# - Changed the filename to match assignment standard (problem_6_2.py).
# - Verified word list loads successfully ("83667 words loaded.").
# - Confirmed helper functions (import_wordlist, is_valid_prefix, get_player_names, next_player, get_valid_letter) work as expected.
# - Cleaned unused code in helper functions.
# - Secured loss condition for not possible prefix (player loses if fragment cannot be extended).
# - Secured loss condition for completing an existing word (player loses if fragment is a valid word longer than 3 letters).
# - Included a loop for multi-game play (ask to play another round).
# - Made sure all player inputs and fragments are strings.
# - Made sure letter input is a single, valid alphabetic character.
# - Tested entering two letters ("ab") as input; program rejected and asked again.
# - Tested entering uppercase letter ("A"); program accepted and converted to lowercase.
# - Tested entering a space (" "); program rejected