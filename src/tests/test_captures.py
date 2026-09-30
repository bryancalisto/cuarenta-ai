import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from card import Card, Rank, Suit
from captures import (
    get_same_rank_captures,
    get_sum_captures,
    get_stairs_captures,
    valid_captures,
)


class TestGetSameRankCaptures:
    """Test get_same_rank_captures function."""

    def test_empty_table(self):
        """Test with empty table returns empty list."""
        played = Card(Rank.TRES, Suit.BRILLOS)
        table = []
        result = get_same_rank_captures(played, table)
        assert result == []

    def test_no_matching_ranks(self):
        """Test when no cards on table match the played card's rank."""
        played = Card(Rank.TRES, Suit.BRILLOS)
        table = [
            Card(Rank.CUATRO, Suit.CORAZONES_ROJOS),
            Card(Rank.CINCO, Suit.CORAZONES_NEGROS),
        ]
        result = get_same_rank_captures(played, table)
        assert result == []

    def test_single_matching_rank(self):
        """Test with a single matching rank on table."""
        played = Card(Rank.TRES, Suit.BRILLOS)
        table_card = Card(Rank.TRES, Suit.CORAZONES_ROJOS)
        table = [table_card]
        result = get_same_rank_captures(played, table)
        assert len(result) == 1
        assert result[0] == frozenset([table_card])

    def test_multiple_matching_ranks(self):
        """Test with multiple cards of the same rank on table."""
        played = Card(Rank.TRES, Suit.BRILLOS)
        card1 = Card(Rank.TRES, Suit.CORAZONES_ROJOS)
        card2 = Card(Rank.TRES, Suit.CORAZONES_NEGROS)
        card3 = Card(Rank.TRES, Suit.TREBOLES)
        table = [card1, card2, card3]
        result = get_same_rank_captures(played, table)
        assert len(result) == 3
        assert frozenset([card1]) in result
        assert frozenset([card2]) in result
        assert frozenset([card3]) in result

    def test_mixed_cards_on_table(self):
        """Test with mixed ranks on table, only matching ones are captured."""
        played = Card(Rank.SIETE, Suit.BRILLOS)
        matching_card = Card(Rank.SIETE, Suit.CORAZONES_ROJOS)
        non_matching = [
            Card(Rank.TRES, Suit.CORAZONES_NEGROS),
            Card(Rank.CUATRO, Suit.TREBOLES),
        ]
        table = non_matching + [matching_card]
        result = get_same_rank_captures(played, table)
        assert len(result) == 1
        assert result[0] == frozenset([matching_card])


class TestGetSumCaptures:
    """Test get_sum_captures function."""

    def test_empty_table(self):
        """Test with empty table returns empty list."""
        played = Card(Rank.CINCO, Suit.BRILLOS)
        table = []
        result = get_sum_captures(played, table)
        assert result == []

    def test_single_card_on_table(self):
        """Test with single card on table that cannot sum."""
        played = Card(Rank.CINCO, Suit.BRILLOS)
        table = [Card(Rank.TRES, Suit.CORAZONES_ROJOS)]
        result = get_sum_captures(played, table)
        assert result == []

    def test_two_cards_sum_to_played(self):
        """Test basic sum: 3 + 2 = 5."""
        played = Card(Rank.CINCO, Suit.BRILLOS)
        card1 = Card(Rank.TRES, Suit.CORAZONES_ROJOS)
        card2 = Card(Rank.DOS, Suit.CORAZONES_NEGROS)
        table = [card1, card2]
        result = get_sum_captures(played, table)
        assert len(result) == 1
        assert frozenset([card1, card2]) in result

    def test_multiple_sum_combinations(self):
        """Test with multiple valid sum combinations.
        Example: Play a 7 against [2, 3, 4, 5]
        - 2 + 5 = 7
        - 3 + 4 = 7
        """
        played = Card(Rank.SIETE, Suit.BRILLOS)
        card1 = Card(Rank.DOS, Suit.CORAZONES_ROJOS)
        card2 = Card(Rank.CINCO, Suit.CORAZONES_NEGROS)
        card3 = Card(Rank.TRES, Suit.TREBOLES)
        card4 = Card(Rank.CUATRO, Suit.BRILLOS)
        table = [card1, card2, card3, card4]
        result = get_sum_captures(played, table)
        assert len(result) == 2
        assert frozenset([card1, card2]) in result
        assert frozenset([card3, card4]) in result

    def test_no_valid_sums(self):
        """Test when table cards cannot sum to played card."""
        played = Card(Rank.SIETE, Suit.BRILLOS)
        table = [
            Card(Rank.AS, Suit.CORAZONES_ROJOS),
            Card(Rank.AS, Suit.CORAZONES_NEGROS),
        ]
        result = get_sum_captures(played, table)
        assert result == []

    def test_sum_with_same_rank_duplicates(self):
        """Test sums with duplicate ranks on table.
        Example: Play a 6 against [2, 2, 4]
        - 2 + 4 = 6 (first 2)
        - 2 + 4 = 6 (second 2)
        """
        played = Card(Rank.SEIS, Suit.BRILLOS)
        card1 = Card(Rank.DOS, Suit.CORAZONES_ROJOS)
        card2 = Card(Rank.DOS, Suit.CORAZONES_NEGROS)
        card3 = Card(Rank.CUATRO, Suit.TREBOLES)
        table = [card1, card2, card3]
        result = get_sum_captures(played, table)
        assert len(result) == 2
        assert frozenset([card1, card3]) in result
        assert frozenset([card2, card3]) in result

    def test_sum_with_ace(self):
        """Test sum with AS (value 1).
        Example: Play a 6 against [5, AS]
        - 5 + 1 = 6
        """
        played = Card(Rank.SEIS, Suit.BRILLOS)
        card1 = Card(Rank.CINCO, Suit.CORAZONES_ROJOS)
        card2 = Card(Rank.AS, Suit.CORAZONES_NEGROS)
        table = [card1, card2]
        result = get_sum_captures(played, table)
        assert len(result) == 1
        assert frozenset([card1, card2]) in result


class TestGetStairsCaptures:
    """Test get_stairs_captures function."""

    def test_empty_table(self):
        """Test with empty table returns empty list."""
        played = Card(Rank.TRES, Suit.BRILLOS)
        table = []
        result = get_stairs_captures(played, table)
        assert result == []

    def test_no_matching_rank(self):
        """Test when table has no card matching played card rank."""
        played = Card(Rank.TRES, Suit.BRILLOS)
        table = [
            Card(Rank.CUATRO, Suit.CORAZONES_ROJOS),
            Card(Rank.CINCO, Suit.CORAZONES_NEGROS),
        ]
        result = get_stairs_captures(played, table)
        assert result == []

    def test_matching_rank_no_sequence(self):
        """Test with matching rank but no consecutive cards."""
        played = Card(Rank.TRES, Suit.BRILLOS)
        table = [
            Card(Rank.TRES, Suit.CORAZONES_ROJOS),
            Card(Rank.CINCO, Suit.CORAZONES_NEGROS),
        ]
        result = get_stairs_captures(played, table)
        assert result == []

    def test_simple_sequence_3_4_5(self):
        """Test simple sequence: 3, 4, 5."""
        played = Card(Rank.TRES, Suit.BRILLOS)
        card_3 = Card(Rank.TRES, Suit.CORAZONES_ROJOS)
        card_4 = Card(Rank.CUATRO, Suit.CORAZONES_NEGROS)
        card_5 = Card(Rank.CINCO, Suit.TREBOLES)
        table = [card_3, card_4, card_5]
        result = get_stairs_captures(played, table)
        assert len(result) == 1
        assert frozenset([card_3, card_4, card_5]) in result

    def test_sequence_with_gaps(self):
        """Test that sequence breaks at gaps.
        Playing 3 against [3, 4, 5, 7] should capture 3, 4, 5 (not 7).
        """
        played = Card(Rank.TRES, Suit.BRILLOS)
        card_3 = Card(Rank.TRES, Suit.CORAZONES_ROJOS)
        card_4 = Card(Rank.CUATRO, Suit.CORAZONES_NEGROS)
        card_5 = Card(Rank.CINCO, Suit.TREBOLES)
        card_7 = Card(Rank.SIETE, Suit.BRILLOS)
        table = [card_3, card_4, card_5, card_7]
        result = get_stairs_captures(played, table)
        assert len(result) == 1
        assert frozenset([card_3, card_4, card_5]) in result

    def test_long_sequence_3_to_7(self):
        """Test longer sequence: 3, 4, 5, 6, 7."""
        played = Card(Rank.TRES, Suit.BRILLOS)
        card_3 = Card(Rank.TRES, Suit.CORAZONES_ROJOS)
        card_4 = Card(Rank.CUATRO, Suit.CORAZONES_NEGROS)
        card_5 = Card(Rank.CINCO, Suit.TREBOLES)
        card_6 = Card(Rank.SEIS, Suit.BRILLOS)
        card_7 = Card(Rank.SIETE, Suit.CORAZONES_ROJOS)
        table = [card_3, card_4, card_5, card_6, card_7]
        result = get_stairs_captures(played, table)
        assert len(result) == 1
        assert frozenset([card_3, card_4, card_5, card_6, card_7]) in result

    def test_multiple_sequences_different_starting_matches(self):
        """Test when multiple cards match rank, each producing a sequence.
        Playing 3 against [3, 4, 5, 3, 4, 5] should capture two sequences:
        - [3, 4, 5] from first 3
        - [3, 4, 5] from second 3
        """
        played = Card(Rank.TRES, Suit.BRILLOS)
        card_3a = Card(Rank.TRES, Suit.CORAZONES_ROJOS)
        card_4a = Card(Rank.CUATRO, Suit.CORAZONES_NEGROS)
        card_5a = Card(Rank.CINCO, Suit.TREBOLES)
        card_3b = Card(Rank.TRES, Suit.BRILLOS)
        card_4b = Card(Rank.CUATRO, Suit.CORAZONES_ROJOS)
        card_5b = Card(Rank.CINCO, Suit.CORAZONES_NEGROS)
        table = [card_3a, card_4a, card_5a, card_3b, card_4b, card_5b]
        result = get_stairs_captures(played, table)
        assert len(result) == 2

    def test_sequence_ending_with_jack(self):
        """Test sequence that ends at Jack.
        Playing 6 against [6, 7, J] should capture [6, 7, J].
        """
        played = Card(Rank.SEIS, Suit.BRILLOS)
        card_6 = Card(Rank.SEIS, Suit.CORAZONES_ROJOS)
        card_7 = Card(Rank.SIETE, Suit.CORAZONES_NEGROS)
        card_j = Card(Rank.JACK, Suit.TREBOLES)
        table = [card_6, card_7, card_j]
        result = get_stairs_captures(played, table)
        assert len(result) == 1
        assert frozenset([card_6, card_7, card_j]) in result

    def test_sequence_jack_queen_king(self):
        """Test sequence of face cards: J, Q, K."""
        played = Card(Rank.JACK, Suit.BRILLOS)
        card_j = Card(Rank.JACK, Suit.CORAZONES_ROJOS)
        card_q = Card(Rank.QUEEN, Suit.CORAZONES_NEGROS)
        card_k = Card(Rank.KING, Suit.TREBOLES)
        table = [card_j, card_q, card_k]
        result = get_stairs_captures(played, table)
        assert len(result) == 1
        assert frozenset([card_j, card_q, card_k]) in result

    def test_sequence_cannot_extend_past_king(self):
        """Test that sequence stops at King (no card after King).
        Playing J against [J, Q, K] should capture [J, Q, K].
        """
        played = Card(Rank.JACK, Suit.BRILLOS)
        card_j = Card(Rank.JACK, Suit.CORAZONES_ROJOS)
        card_q = Card(Rank.QUEEN, Suit.CORAZONES_NEGROS)
        card_k = Card(Rank.KING, Suit.TREBOLES)
        table = [card_j, card_q, card_k]
        result = get_stairs_captures(played, table)
        assert len(result) == 1
        assert frozenset([card_j, card_q, card_k]) in result

    def test_unordered_table_still_works(self):
        """Test that unordered table cards are handled correctly.
        The function sorts the table internally.
        """
        played = Card(Rank.TRES, Suit.BRILLOS)
        card_3 = Card(Rank.TRES, Suit.CORAZONES_ROJOS)
        card_4 = Card(Rank.CUATRO, Suit.CORAZONES_NEGROS)
        card_5 = Card(Rank.CINCO, Suit.TREBOLES)
        table = [card_5, card_3, card_4]  # Out of order
        result = get_stairs_captures(played, table)
        assert len(result) == 1
        assert frozenset([card_3, card_4, card_5]) in result


class TestValidCaptures:
    def test_sum_and_stairs_can_chain(self):
        """Test that sum captures can chain into a stair capture."""
        played = Card(Rank.CINCO, Suit.BRILLOS)
        card_2 = Card(Rank.DOS, Suit.CORAZONES_ROJOS)
        card_3 = Card(Rank.TRES, Suit.CORAZONES_NEGROS)
        card_5 = Card(Rank.CINCO, Suit.TREBOLES)
        card_6 = Card(Rank.SEIS, Suit.BRILLOS)
        card_7 = Card(Rank.SIETE, Suit.CORAZONES_ROJOS)
        table = [card_2, card_3, card_5, card_6, card_7]

        result = valid_captures(played, table)

        assert frozenset([card_2, card_3, card_5, card_6, card_7]) in result


    def test_miscellaneous_1(self):
        """Table contains 7,J,Q,3,5,6,4. The player throws a 7. The player has no option to capture all cards"""
        played = Card(Rank.SIETE, Suit.BRILLOS)
        card_7 = Card(Rank.SIETE, Suit.CORAZONES_ROJOS)
        card_j = Card(Rank.JACK, Suit.CORAZONES_ROJOS)
        card_q = Card(Rank.QUEEN, Suit.CORAZONES_NEGROS)
        card_3 = Card(Rank.TRES, Suit.TREBOLES)
        card_5 = Card(Rank.CINCO, Suit.BRILLOS)
        card_6 = Card(Rank.SEIS, Suit.CORAZONES_NEGROS)
        card_4 = Card(Rank.CUATRO, Suit.TREBOLES)
        table = [card_7, card_j, card_q, card_3, card_5, card_6, card_4]

        result = valid_captures(played, table)
        print(f"Valid captures for 7 on table {table}: {result}")

        # valid
        assert len(result) == 4
        assert frozenset([card_7]) in result
        assert frozenset([card_3, card_4]) in result
        assert frozenset([card_7, card_j, card_q]) in result
        assert frozenset([card_3, card_4, card_7, card_j, card_q]) in result

        # invalid
        assert frozenset([card_7, card_j, card_q, card_3, card_5, card_6, card_4]) not in result
