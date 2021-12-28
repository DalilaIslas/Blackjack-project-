import numpy as np

#create 52-card deck 
cards = {
  "ace": 11,
  "two": 2,
  "three": 3,
  "four": 4,
  "five": 5,
  "six": 6,
  "seven": 7,
  "eight": 8,
  "nine": 9,
  "ten": 10,
  "jack": 10,
  "queen": 10,
  "king": 10
}

#empty deck 
deck = []

for suit in range(4):
    for card in cards: 
        deck.append(card)

deck = np.array(deck)



#shuffle deck
np.random.shuffle(deck)
print(deck)

def addCard(hand, card):
    cardValue = cards[card]
    hand.append(cardValue)

def sumPoints(hand):
    return np.array(hand).sum()

def computeScore(points):
    if points <= 21:
        return points ** 2

    return 0

# 
nextCardIndex = 0
scores = []

while nextCardIndex !=  deck.size:
    # Initialize Hand State
    hand = []
    points = 0
    n = 2
    
    print('-------------')
    print('New hand')

    remainingCards = deck[nextCardIndex:deck.size].size

    print(f'Remaining cards: {remainingCards}')

    # Deal cards
    for i in range(n):
        addCard(hand, deck[nextCardIndex])
        nextCardIndex += 1

    points = sumPoints(hand)

 
    while points < 21:
        print(f'Points: {points}')

        decision = input("Stick (0) or Hit (1):")

        if decision == '0':
            print(f'Points: {points}')
            break
        
        addCard(hand, deck[nextCardIndex])
        nextCardIndex += 1

        points = sumPoints(hand)

    print(hand)

    scores.append(computeScore(points))

    print(f'Scores:{scores}')
