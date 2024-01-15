from itertools import product

card_values = ("A", "2", "3", "4", "5", "6", "7", "J", "Q", "K")
card_types = ("clubs", "diamonds", "hearts", "spades")


class Game:
    def __init__(self):
        self.new_game()

    def new_round(self):
        self.deck = set(product(card_values, card_types))
        self.ai_hand = set()
        self.table = set()
        self.thrown_cards = []
        self.player_front_predicted_hand = {}
        self.player_right_predicted_hand = {}
        self.player_left_predicted_hand = {}

    def new_game(self):
        self.points = (0, 0)  # AI is index 0
        self.carton = (0, 0)  # AI is index 0
        self.deck = set(product(card_values, card_types))
        self.table = set()
        self.thrown_cards = []
        self.ai_hand = set()
        self.player_front_predicted_hand = {}
        self.player_right_predicted_hand = {}
        self.player_left_predicted_hand = {}

    # calculate probable hands of other players
    def exec_turn(self, just_thrown_card):
        last_thrown_card = None
        cards_to_remove_from_table = set()

        if len(self.thrown_cards) > 0:
            last_thrown_card = self.thrown_cards[-1]

        # Game or round just started
        if last_thrown_card is None:
            # Cannot tell anything about
            pass
        # Caida
        elif last_thrown_card[0] == just_thrown_card[0]:
            print("caida")
            cards_to_remove_from_table.add(last_thrown_card)
            cards_to_remove_from_table.add(just_thrown_card)

        self.thrown_cards.append(just_thrown_card)

        for card in cards_to_remove_from_table:
            self.remove_card_from_table(card)

    def play_ai_turn(self):
        # Throw next card based on:
        # - Probable hand of left player (to increase possibility of gain)
        # - Probable hand of right player (to decrease possibility of loss)
        pass

    def put_card_on_table(self, card):
        self.table.add(card)

    def remove_card_from_table(self, card):
        self.table.discard(card)

    def put_card_in_ai_hand(self, card):
        self.ai_hand.add(card)

    def remove_card_from_ai_hand(self, card):
        self.ai_hand.discard(card)
