import random

blank =    [[0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]]
tiger =    [[0,0,1,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0]]
dragon =   [[0,0,0,0,0],
            [1,0,0,0,1],
            [0,0,0,0,0],
            [0,1,0,1,0],
            [0,0,0,0,0]]
frog =     [[0,0,0,0,0],
            [0,1,0,0,0],
            [1,0,0,0,0],
            [0,0,0,1,0],
            [0,0,0,0,0]]
rabbit =   [[0,0,0,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1],
            [0,1,0,0,0],
            [0,0,0,0,0]]
crab =     [[0,0,0,0,0],
            [0,0,1,0,0],
            [1,0,0,0,1],
            [0,0,0,0,0],
            [0,0,0,0,0]]
elephant = [[0,0,0,0,0],
            [0,1,0,1,0],
            [0,1,0,1,0],
            [0,0,0,0,0],
            [0,0,0,0,0]]
goose =    [[0,0,0,0,0],
            [0,1,0,0,0],
            [0,1,0,1,0],
            [0,0,0,1,0],
            [0,0,0,0,0]]
rooster =  [[0,0,0,0,0],
            [0,0,0,1,0],
            [0,1,0,1,0],
            [0,1,0,0,0],
            [0,0,0,0,0]]
monkey =   [[0,0,0,0,0],
            [0,1,0,1,0],
            [0,0,0,0,0],
            [0,1,0,1,0],
            [0,0,0,0,0]]
mantis =   [[0,0,0,0,0],
            [0,1,0,1,0],
            [0,0,0,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0]]
horse =    [[0,0,0,0,0],
            [0,0,1,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0]]
ox =       [[0,0,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,1,0,0],
            [0,0,0,0,0]]
crane =    [[0,0,0,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0],
            [0,1,0,1,0],
            [0,0,0,0,0]]
boar =     [[0,0,0,0,0],
            [0,0,1,0,0],
            [0,1,0,1,0],
            [0,0,0,0,0],
            [0,0,0,0,0]]
eel =      [[0,0,0,0,0],
            [0,1,0,0,0],
            [0,0,0,1,0],
            [0,1,0,0,0],
            [0,0,0,0,0]]
cobra =    [[0,0,0,0,0],
            [0,0,0,1,0],
            [0,1,0,0,0],
            [0,0,0,1,0],
            [0,0,0,0,0]]

# For test case
only_sideways = [[0,0,0,0,0],
                 [0,0,0,0,0],
                 [1,1,0,1,1],
                 [0,0,0,0,0],
                 [0,0,0,0,0]]

simple_card = [[0,0,0,0,0],
               [0,1,1,1,0],
               [0,1,0,1,0],
               [0,1,1,1,0],
               [0,0,0,0,0]]

all_cards = [tiger, dragon, frog, rabbit,
            crab, elephant, goose, rooster,
            monkey, mantis, horse, ox,
            crane, boar, eel, cobra]

card_stamps={1:[tiger,rabbit,crab,goose,monkey,ox,crane,eel],
             2:[dragon,frog,elephant,rooster,mantis,horse,boar,cobra]}


def seed_cards(seed):
    random.seed(12312)


def get_init_cards(do_shuffle=True, simple_cards=False, custom_cards=None):
    return ([crab,mantis],
            [monkey,boar],
            [frog])
    if simple_cards:
        cards = [simple_card, simple_card,
                 simple_card, simple_card,
                 simple_card]
    elif custom_cards is not None:
        cards = custom_cards
    else:
        cards = all_cards
    if do_shuffle:
        random.shuffle(cards)
    return ([cards[0], cards[1]],
            [cards[2], cards[3]],
            [cards[4]])

