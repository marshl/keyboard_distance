"""
Command line tool for finding statistics about the distance on a keyboard that various words take up
"""
import argparse
import dataclasses
import enum
import math
import os
from typing import Tuple, List

import requests

WORD_FILE_PATH = "words.txt"
SIMPLE_WORD_FILE_PATH = "words.simple.txt"

DEFAULT_KEY_SIZE_MM = 22  # millimetres


class KeyboardType(enum.Enum):
    QWERTY = "QWERTY"
    DVORAK = "DVORAK"


KEYBOARD_LAYOUT_MAP = {
    KeyboardType.QWERTY: ["qwertyuiop", "asdfghjkl", "zxcvbnm"],
    KeyboardType.DVORAK: ["   pyfgcrl", "aoeuidhtns", " qjkxbmwvz"],
}

KEYBOARD_ROW_X_OFFSETS = [0, 0.25, 0.75]


@dataclasses.dataclass
class Position2D:
    """
    A position in 2D space
    """

    x: float
    y: float

    def __sub__(self, other: "Position2D") -> "Position2D":
        """
        Subtracts the other vector from this one and returns the result
        :param other: The other position
        :return: A new vector of z where z = this-other
        >>> Position2D(x=10, y=5) - Position2D(x=3, y=2)
        Position2D(x=7, y=3)
        """
        return Position2D(x=self.x - other.x, y=self.y - other.y)

    def __add__(self, other: "Position2D") -> "Position2D":
        """
        Adds this vector to the other vector
        :param other: The other vector to add to
        :return: A new vector of the two combined
        """
        return Position2D(x=self.x + other.x, y=self.y + other.y)

    def __truediv__(self, other: float) -> "Position2D":
        """
        Gets this vector divided by a scalar value
        :param other: The scalar value to divide by
        :return: A new vector.
        """
        return Position2D(x=self.x / other, y=self.y / other)

    def dot_product(self, other: "Position2D") -> float:
        """
        Gets the dot product of this vector to the other vector
        :param other: The other vector
        :return: The dot product of the two vectors
        """
        return self.x * other.x + self.y * other.y

    def cross_product_scalar(self, other: "Position2D") -> float:
        """

        :param other:
        :return:
        >>> Position2D(x=5, y=0).cross_product_scalar(Position2D(x=0, y=4))
        1.0
        >>> Position2D(x=0, y=3).cross_product_scalar(Position2D(x=4, y=0))
        -1.0
        """
        return (self.x * other.y - self.y * other.x) / (
            self.magnitude() * other.magnitude()
        )

    def magnitude(self) -> float:
        """
        Gets the magnitude of this vector
        :return: The magnitude of this vector
        """
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def angle_between(self, other: "Position2D") -> float:
        """
        Gets the angle between this vector and the other vector
        :param other: The other vector to get the angle between
        :return:
        >>> Position2D(x=0.5, y=0.5).angle_between(Position2D(x=0, y=1))
        0.7853981633974484
        """
        return math.acos(
            self.dot_product(other) / (self.magnitude() * other.magnitude())
        )


def get_letter_position(
    key: str, keyboard_type: KeyboardType = KeyboardType.QWERTY
) -> Position2D:
    """
    Gets the Position2D of a key on a keyboard
    :param key: The key to get the position of
    :param keyboard_type: The kind of keyboard (qwerty/dvorak etc.)
    :return: The position of that key on the keyboard
    """
    key = key.lower()
    position = get_letter_position.position_cache.get(key)
    if position is not None:
        return position
    if not "a" <= key <= "z":
        raise ValueError("Letter must be between a and z")

    keyboard = KEYBOARD_LAYOUT_MAP[keyboard_type]
    y_index = next(y for y in range(len(keyboard)) if key in keyboard[y])
    x_index = keyboard[y_index].index(key)
    position = Position2D(x=KEYBOARD_ROW_X_OFFSETS[y_index] + x_index, y=y_index)
    get_letter_position.position_cache[key] = position
    return position


get_letter_position.position_cache = {}


def get_letter_distance(
    key_1: str, key_2: str, keyboard_type=KeyboardType.QWERTY
) -> float:
    """
    Gets the distance between two different keys
    :param key_1: The first key
    :param key_2: The second key
    :param keyboard_type: The type of keyboard (qwerty, dvorak etc.)
    :return: The distance between the two keys
    """

    distance = get_letter_distance.distance_cache.get((key_1, key_2))
    if distance is not None:
        return distance
    distance = (
        get_letter_position(key_1, keyboard_type)
        - get_letter_position(key_2, keyboard_type)
    ).magnitude()
    get_letter_distance.distance_cache[(key_1, key_2)] = distance
    return distance


get_letter_distance.distance_cache = {}


def is_simple_word(word: str) -> bool:
    """
    Gets whether the word is
    :param word:
    :return:
    """
    return len(word) > 1 and all("a" <= c <= "z" for c in word)


def keyboard_distance_to_cm(
    distance: float, key_size: float = DEFAULT_KEY_SIZE_MM
) -> float:
    """

    :param distance:
    :param key_size:
    :return:
    >>> keyboard_distance_to_cm(3)
    6.6
    """
    return distance * key_size / 10


def get_word_traversal_length(
    word: str,
    include_same_letter_distance: bool = True,
    keyboard_type=KeyboardType.QWERTY,
) -> Tuple[float, float]:
    """
    Gets the
    :param word:
    :param include_same_letter_distance:
    :param keyboard_type:
    :return:
    >>> get_word_traversal_length("qw")
    (1.0, 1.0)
    >>> get_word_traversal_length("qwerty")
    (5.0, 1.0)
    """
    if len(word) <= 1:
        return 0, 0

    total_length = 0
    gaps = 0
    current_letter = word[0]
    for next_letter in word[1:]:
        total_length += get_letter_distance(
            current_letter, next_letter, keyboard_type=keyboard_type
        )

        if next_letter != current_letter or include_same_letter_distance:
            gaps += 1
        current_letter = next_letter

    return total_length, total_length / gaps


def create_word_lists() -> None:
    """
    Downloads the word list and saves it to disk. Then creates a simplified version of that list.
    """
    if not os.path.exists(WORD_FILE_PATH):
        url = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
        response = requests.get(url, stream=True)
        with open(WORD_FILE_PATH, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    output.write(chunk)

    if not os.path.exists(SIMPLE_WORD_FILE_PATH):
        with open(WORD_FILE_PATH, encoding="utf-8") as file:
            with open(SIMPLE_WORD_FILE_PATH, "w", encoding="utf-8") as output_file:
                for word in file.readlines():
                    if is_simple_word(word[:-1]):
                        output_file.write(word)


def main():
    """
    Entry point
    """
    parser = argparse.ArgumentParser(
        description="Finds the distance of a word on the keyboard"
    )
    parser.add_argument("words", type=str, nargs="+", help="words to get distances for")
    parser.add_argument(
        "--largest",
        dest="largest",
        action="store_const",
        const=sum,
        default=max,
        help="sum the integers (default: find the max)",
    )

    args = parser.parse_args()

    if args.words:
        for word in args.words:
            total_length, relative_length = get_word_traversal_length(word)
            print(
                f"{word}: total length: {keyboard_distance_to_cm(total_length)}cm "
                f"relative length: {keyboard_distance_to_cm(relative_length)}cm"
            )

        return

    create_word_lists()
    best_words: List[Tuple[float, str]] = []
    best_limit = 10

    with open(SIMPLE_WORD_FILE_PATH, encoding="utf-8") as file:
        for word in file.readlines():
            word = word.strip()
            # if not is_simple_word(word) or len(word) < 6:
            #     continue

            _, relative_length = get_word_traversal_length(word)
            if not best_words or relative_length > best_words[-1][0]:
                best_words.append((relative_length, word))
                best_words.sort(key=lambda x: x[0], reverse=True)
                if len(best_words) > best_limit:
                    best_words.pop()

    print(best_words)


if __name__ == "__main__":
    main()
