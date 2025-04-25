deck = input('Enter cards from the deck separated by commas').split(',')

print('The cards in the deck are: {}'.format(deck))
print('Deck summ = ', sum(deck))
print('Card missing to blackjack = ', 21 - sum(deck))