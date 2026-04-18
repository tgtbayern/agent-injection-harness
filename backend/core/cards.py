"""
Core poker card and deck representation.
Provides Card, Deck, and utility functions for poker game mechanics.
"""

from enum import IntEnum
from typing import List, Tuple, Optional
import random
import itertools


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


RANK_CHARS = '23456789TJQKA'
SUIT_CHARS = 'cdhs'
SUIT_SYMBOLS = {'c': '♣', 'd': '♦', 'h': '♥', 's': '♠'}


class Card:
    """Immutable card representation using a single integer [0..51]."""

    __slots__ = ('_id',)

    def __init__(self, rank: int, suit: int):
        self._id = (rank - 2) * 4 + suit

    @classmethod
    def from_id(cls, card_id: int) -> 'Card':
        c = cls.__new__(cls)
        c._id = card_id
        return c

    @classmethod
    def from_str(cls, s: str) -> 'Card':
        """Parse card from string like 'As', 'Td', '2c'."""
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s}")
        rank_idx = RANK_CHARS.index(s[0].upper())
        suit_idx = SUIT_CHARS.index(s[1].lower())
        return cls(rank_idx + 2, suit_idx)

    @property
    def id(self) -> int:
        return self._id

    @property
    def rank(self) -> int:
        return (self._id // 4) + 2

    @property
    def suit(self) -> int:
        return self._id % 4

    @property
    def rank_char(self) -> str:
        return RANK_CHARS[self.rank - 2]

    @property
    def suit_char(self) -> str:
        return SUIT_CHARS[self.suit]

    def __repr__(self):
        return f"{self.rank_char}{self.suit_char}"

    def __eq__(self, other):
        return isinstance(other, Card) and self._id == other._id

    def __hash__(self):
        return self._id

    def __lt__(self, other):
        return self.rank < other.rank or (self.rank == other.rank and self.suit < other.suit)


class Deck:
    """Standard 52-card deck with deal and remove operations."""

    def __init__(self):
        self.cards: List[Card] = [Card.from_id(i) for i in range(52)]
        self.dealt: set = set()

    def shuffle(self):
        random.shuffle(self.cards)
        self.dealt.clear()

    def remove(self, cards: List[Card]):
        """Remove specific cards from deck (for known board/hole cards)."""
        for c in cards:
            self.dealt.add(c.id)

    def remaining(self) -> List[Card]:
        return [c for c in self.cards if c.id not in self.dealt]

    def deal(self, n: int = 1) -> List[Card]:
        result = []
        for c in self.cards:
            if c.id not in self.dealt:
                result.append(c)
                self.dealt.add(c.id)
                if len(result) == n:
                    break
        return result


def parse_cards(s: str) -> List[Card]:
    """Parse space or comma separated card string: 'As Kd Qh'."""
    s = s.strip().replace(',', ' ')
    tokens = s.split()
    return [Card.from_str(t) for t in tokens]


def card_ids_to_mask(cards: List[Card]) -> int:
    """Convert card list to 64-bit bitmask."""
    mask = 0
    for c in cards:
        mask |= (1 << c.id)
    return mask


def all_hole_card_combos(dead_cards: List[Card] = None) -> List[Tuple[Card, Card]]:
    """Generate all 1326 possible hole card combinations, excluding dead cards."""
    dead = set(c.id for c in dead_cards) if dead_cards else set()
    available = [Card.from_id(i) for i in range(52) if i not in dead]
    return list(itertools.combinations(available, 2))
