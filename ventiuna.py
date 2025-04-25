deck = input('Enter cards from deck separated by commas: ').split(',')
deck = list(map(int, deck))
# Convert the deck elements to integers
print('Deck:', deck)
print('Deck length:', len(deck))
print('Deck sum:', sum(deck))