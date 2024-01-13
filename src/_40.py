from itertools import product

card_values = ("A", "2", "3", "4", "5", "6", "7", "J", "Q", "K")
card_types = ("clubs", "diamonds", "hearts", "spades")


class Game:
    def __init__(self):
        self.new_game()

    def new_round(self):
        self.deck = set(product(card_values, card_types))
        self.ai_hand = set()
        self.player_front_predicted_hand = {}
        self.player_right_predicted_hand = {}
        self.player_left_predicted_hand = {}

    def new_game(self):
        self.points = (0, 0)  # AI is index 0
        self.carton = (0, 0)  # AI is index 0
        self.deck = set(product(card_values, card_types))
        self.table = set()
        self.ai_hand = set()
        self.player_front_predicted_hand = {}
        self.player_right_predicted_hand = {}
        self.player_left_predicted_hand = {}

    def play_ai_turn(self):
        pass

    def put_card_on_table(self, card):
        self.table.add(card)

    def remove_card_from_table(self, card):
        self.table.discard(card)

    def put_card_in_ai_hand(self, card):
        self.ai_hand.add(card)

    def remove_card_from_ai_hand(self, card):
        self.ai_hand.discard(card)
