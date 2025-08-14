"""
Fill in these comments as required for this assignment

Add the programming purpose here
Add the data this code was started or revised
Add your name
"""
from problem_6_1 import *  # calls in Scrabble programming
from problem_6_1 import calc_word_score

# message colors
oText = '\033[0m' # reset style to original
goRed = '\033[91m'
goGreen = '\033[92m'
goYellow = '\033[93m'
#
# Test code
#
def test_dealing_hands():
    """
    test for dealing_hands
    """
    
    # see if the right kind of dictionary is ret

    hand = dealing_hands(letters_per_hand)
    if not type(hand) is dict:
        print(goRed + "UNSUCCESSFUL: test_dealing_hands()" + oText)
        print(goYellow + "\tUnexpected return type:", type(hand), oText)
        return # exit

    num = 0
    for k in hand.keys():
        if (not type(k) is str) or (not type(hand[k]) is int):
            print(goRed + "UNSUCCESSFUL: test_dealing_hands()" + oText)
            print(goYellow + "\tUnexpected type of dictionary: string -> int expected, but was", type(k), "->", type(hand[k]), oText)
            return # exit
        elif not k in "abcdefghijklmnopqrstuvwxyz":
            print(goRed + "UNSUCCESSFUL: test_dealing_hands()" + oText)
            print(goYellow + "\tDictionary keys are not lowercase letters." + oText)
            return # exit
        else:
            num += hand[k]
            
    if num != letters_per_hand:
            print(goRed + "UNSUCCESSFUL: test_dealing_hands()" + oText)
            print(goYellow + "\tdealing_hands() returned more letters than it was asked to." + oText)
            print(goYellow + "\tAsked for a hand of size", letters_per_hand, "but it returned a hand of size", num, oText)
            return # exit
        
    # tests the randomness
    repeats=0
    hand1 = dealing_hands(letters_per_hand)
    for i in range(20):                
        hand2 = dealing_hands(letters_per_hand)
        if hand1 == hand2:
            repeats += 1
        hand1 = hand2
        
    if repeats > 10:
        print(goRed + "UNSUCCESSFUL: test_dealing_hands()" + oText)
        print(goYellow + "\tSame hand returned", repeats, "times by dealing_hands(). This is HIGHLY unlikely." + oText)
        print(goYellow + "\tIs the dealing_hands implementation really using random numbers?" + oText)
        return # exit
    
    print(goGreen + "SUCCESS: test_dealing_hands()")

def test_scoring_words():
    """
    test for calc_word_score
    """
    UNSUCCESSFUL=False
    # dictionary of words and scores
    words = {("", 7):0, ("it", 7):2, ("was", 7):6, ("scored", 7):9, ("waybill", 7):65, ("outgnaw", 7):61, ("outgnawn", 8):62}
    for (word, n) in words.keys():
        score = calc_word_score(word, n)
        if score != words[(word, n)]:
            print(goRed + "UNSUCCESSFUL: test_calc_word_score()")
            print(goYellow + "\tExpected", words[(word, n)], "points but got '" + str(score) + "' for word '" + word + "', n=" + str(n))
            UNSUCCESSFUL=True
    if not UNSUCCESSFUL:
        print(goGreen + "SUCCESS: test_calc_word_score()" + oText)

def test_hand_updates():
    """
    test for hand_update
    """
    # test 1
    hand = {'a':1, 'q':1, 'l':2, 'm':1, 'u':1, 'i':1}
    word = "quail"

    hand2 = hand_update(hand.copy(), word)
    expected_hand1 = {'l':1, 'm':1}
    expected_hand2 = {'a':0, 'q':0, 'l':1, 'm':1, 'u':0, 'i':0}
    if hand2 != expected_hand1 and hand2 != expected_hand2:
        print(goRed + "UNSUCCESSFUL: test_hand_update('"+ word +"', " + str(hand) + ")" + oText)
        print(goYellow + "\tReturned: ", hand2, "-- but expected:", expected_hand1, "or", expected_hand2, oText)
        return # exit
        
    # test 2
    hand = {'e':1, 'v':2, 'n':1, 'i':1, 'l':2}
    word = "evil"

    hand2 = hand_update(hand.copy(), word)
    expected_hand1 = {'v':1, 'n':1, 'l':1}
    expected_hand2 = {'e':0, 'v':1, 'n':1, 'i':0, 'l':1}
    if hand2 != expected_hand1 and hand2 != expected_hand2:
        print(goRed + "UNSUCCESSFUL: test_hand_update('"+ word +"', " + str(hand) + ")" + oText)
        print(goYellow + "\tReturned: ", hand2, "-- but expected:", expected_hand1, "or", expected_hand2, oText)
        return # exit

    # test 3
    hand = {'h': 1, 'e': 1, 'l': 2, 'o': 1}
    word = "hello"

    hand2 = hand_update(hand.copy(), word)
    expected_hand1 = {}
    expected_hand2 = {'h': 0, 'e': 0, 'l': 0, 'o': 0}
    if hand2 != expected_hand1 and hand2 != expected_hand2:
        print(goRed + "UNSUCCESSFUL: test_hand_update('"+ word +"', " + str(hand) + ")" + oText)
        print(goYellow + "\tReturned: ", hand2, "-- but expected:", expected_hand1, "or", expected_hand2, oText)
        return # exit

    print(goGreen + "SUCCESS: test_hand_update()" + oText)

def test_word_validity(word_list):
    """
    test for word_is_valid
    """
    UNSUCCESSFUL=False
    # test 1
    word = "hello"
    hand = into_dictionary(word)

    if not word_is_valid(word, hand, word_list):
        print(goRed + "UNSUCCESSFUL: test_word_is_valid()" + oText)
        print(goYellow + "\tExpected True, but got False for word: '" + word + "' and hand:", hand, oText)
        UNSUCCESSFUL = True

    # test 2
    hand = {'r': 1, 'a': 3, 'p': 2, 'e': 1, 't': 1, 'u':1}
    word = "rapture"

    if  word_is_valid(word, hand, word_list):
        print(goRed + "UNSUCCESSFUL: test_word_is_valid()" + oText)
        print(goYellow + "\tExpected False, but got True for word: '" + word + "' and hand:", hand, oText)
        UNSUCCESSFUL = True        

    # test 3
    hand = {'n': 1, 'h': 1, 'o': 1, 'y': 1, 'd':1, 'w':1, 'e': 2}
    word = "honey"

    if  not word_is_valid(word, hand, word_list):
        print(goRed + "UNSUCCESSFUL: test_word_is_valid()" + oText)
        print(goYellow + "\tExpected True, but got False for word: '"+ word +"' and hand:", hand, oText)
        UNSUCCESSFUL = True                        

    # test 4
    hand = {'r': 1, 'a': 3, 'p': 2, 't': 1, 'u':2}
    word = "honey"

    if  word_is_valid(word, hand, word_list):
        print(goRed + "UNSUCCESSFUL: test_word_is_valid()" + oText)
        print(goYellow + "\tExpected False, but got True for word: '" + word + "' and hand:", hand, oText)
        UNSUCCESSFUL = True

    # test 5
    hand = {'e':1, 'v':2, 'n':1, 'i':1, 'l':2}
    word = "evil"
    
    if  not word_is_valid(word, hand, word_list):
        print(goRed + "UNSUCCESSFUL: test_word_is_valid()" + oText)
        print(goYellow + "\tExpected True, but got False for word: '" + word + "' and hand:", hand, oText)
        UNSUCCESSFUL = True
        
    # test 6
    word = "even"

    if  word_is_valid(word, hand, word_list):
        print(goRed + "UNSUCCESSFUL: test_word_is_valid()")
        print(goYellow + "\tExpected False, but got True for word: '" + word + "' and hand:", hand, oText)
        print(goYellow + "\t(If this is the only UNSUCCESSFUL, make sure word_is_valid() isn't mutating its inputs)" + oText)
        UNSUCCESSFUL = True        

    if not UNSUCCESSFUL:
        print(goGreen + "SUCCESS: test_word_is_valid()" + oText)


word_list = import_wordlist()
print("----------------------------------------------------------------------")
print("Testing calc_word_score...")
test_scoring_words()
print("----------------------------------------------------------------------")
print("Testing hand_updates...")
test_hand_updates()
print("----------------------------------------------------------------------")
print("Testing word_is_valid...")
test_word_validity(word_list)
print("----------------------------------------------------------------------")
print("Testing calc_word_score...")
print(calc_word_score("reading", 7))  # Should include bonus if word uses all hand
print(calc_word_score("read", 7))     # Should not include bonus
print(calc_word_score("zzz", 7))      # Should score 30 + bonus if qty == 3
print(calc_word_score("", 7))         # Should return 0
print("----------------------------------------------------------------------")
print("All done!")