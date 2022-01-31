"""
Command line tool for finding statistics about the distance on a keyboard that various words take up
"""
import argparse
import dataclasses
import enum
import math
import operator
import os
import shutil
import urllib.request
from itertools import combinations, groupby
from typing import Tuple, List

WORD_FILE_PATH = "words.txt"
SIMPLE_WORD_FILE_PATH = "words.simple.txt"

DEFAULT_KEY_SIZE_MM = 22  # millimetres


class KeyboardLayout(enum.Enum):
    """
    Enum for the different keyboard layouts
    """

    QWERTY = "QWERTY"
    DVORAK = "DVORAK"


KEYBOARD_LAYOUT_MAP = {
    KeyboardLayout.QWERTY: ["qwertyuiop", "asdfghjkl", "zxcvbnm"],
    KeyboardLayout.DVORAK: ["   pyfgcrl", "aoeuidhtns", " qjkxbmwvz"],
}

KEYBOARD_ROW_X_OFFSETS = [0, 0.25, 0.75]


@dataclasses.dataclass
class Position2D:
    """
    A position in 2D space
    """

    x: float  # pylint: disable=invalid-name
    y: float  # pylint: disable=invalid-name

    def __hash__(self):
        return hash((self.x, self.y))

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
        total_magnitude = self.magnitude() * other.magnitude()
        if total_magnitude == 0:
            return 0
        try:
            return math.acos(self.dot_product(other) / total_magnitude)
        except ValueError:
            return 0


def get_letter_position(
    key: str, keyboard_type: KeyboardLayout = KeyboardLayout.QWERTY
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


def get_key_position_distance(
    key_1: str, key_2: str, keyboard_type=KeyboardLayout.QWERTY
) -> float:
    """
    Gets the distance between two different keys
    :param key_1: The first key
    :param key_2: The second key
    :param keyboard_type: The type of keyboard (qwerty, dvorak etc.)
    :return: The distance between the two keys
    """
    distance = get_key_position_distance.distance_cache.get((key_1, key_2))
    if distance is not None:
        return distance
    distance = (
        get_letter_position(key_1, keyboard_type)
        - get_letter_position(key_2, keyboard_type)
    ).magnitude()
    get_key_position_distance.distance_cache[(key_1, key_2)] = distance
    return distance


get_key_position_distance.distance_cache = {}


def get_three_key_angle(
    key_1: str,
    key_2: str,
    key_3: str,
    keyboard_type: KeyboardLayout = KeyboardLayout.QWERTY,
) -> float:
    """
    Gets the angle formed by the triangle between three keys
    :param key_1: The first key
    :param key_2: The second key
    :param key_3: The third key
    :param keyboard_type: The type of keyboard the keys are on
    :return: The angle in radians (pi if the keys all line up, 0 if the 1st and 3rd key are same)
    >>> get_three_key_angle('q', 'w', 'e')
    3.141592653589793
    >>> get_three_key_angle('q', 'e', 'q')
    0.0
    """
    angle = get_three_key_angle.angle_cache.get((key_1, key_2, key_3))
    if angle is not None:
        return angle
    p1 = get_letter_position(key_1, keyboard_type)
    p2 = get_letter_position(key_2, keyboard_type)
    p3 = get_letter_position(key_3, keyboard_type)

    v1 = p2 - p1
    v2 = p2 - p3
    angle = v1.angle_between(v2)
    get_three_key_angle.angle_cache[(key_1, key_2, key_3)] = angle
    return angle


get_three_key_angle.angle_cache = {}


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
    keyboard_type=KeyboardLayout.QWERTY,
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
    for current_letter, next_letter in zip(word, word[1:]):
        total_length += get_key_position_distance(
            current_letter, next_letter, keyboard_type=keyboard_type
        )

        if next_letter != current_letter or include_same_letter_distance:
            gaps += 1

    return total_length, total_length / gaps


def get_word_traversal_angle(word: str) -> Tuple[float, float]:
    if len(word) <= 1:
        return 0, 0
    grouped_word = [x[0] for x in groupby(word)]
    total_angle = 0
    gaps = 0
    for previous_letter, current_letter, next_letter in zip(grouped_word, grouped_word[1:], grouped_word[2:]):
        total_angle += get_three_key_angle(previous_letter, current_letter, next_letter)

        if next_letter != current_letter:
            gaps += 1

    return total_angle, total_angle / gaps if gaps else 0


def create_word_lists() -> None:
    """
    Downloads the word list and saves it to disk. Then creates a simplified version of that list.
    """
    if not os.path.exists(WORD_FILE_PATH):
        url = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
        with urllib.request.urlopen(url) as response, open(
            WORD_FILE_PATH, "wb"
        ) as out_file:
            shutil.copyfileobj(response, out_file)

    if not os.path.exists(SIMPLE_WORD_FILE_PATH):
        with open(WORD_FILE_PATH, encoding="utf-8") as file:
            with open(SIMPLE_WORD_FILE_PATH, "w", encoding="utf-8") as output_file:
                for word in file.readlines():
                    if is_simple_word(word[:-1]):
                        output_file.write(word)


def on_segment(p: Position2D, q: Position2D, r: Position2D) -> bool:
    return max(p.x, r.x) >= q.x >= min(p.x, r.x) and max(p.y, r.y) >= q.y >= min(
        p.y, r.y
    )


def orientation(p: Position2D, q: Position2D, r: Position2D) -> int:
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:
        # Clockwise orientation
        return 1
    if val < 0:
        # Counterclockwise orientation
        return 2
    # Collinear orientation
    return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def do_intersect(p1: Position2D, q1: Position2D, p2: Position2D, q2: Position2D):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and on_segment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and on_segment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and on_segment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and on_segment(p2, q1, q2):
        return True

    # If none of the cases
    return False


def does_word_intersect_itself(
    word: str, keyboard_type: KeyboardLayout = KeyboardLayout.QWERTY
):
    positions = [get_letter_position(key, keyboard_type) for key in word]
    position_pairs = list(zip(positions, positions[1:]))
    position_combinations = list(combinations(position_pairs, 2))
    for pair_1, pair_2 in position_combinations:
        if len({*pair_1, *pair_2}) != 4:
            continue

        if do_intersect(pair_1[0], pair_1[1], pair_2[0], pair_2[1]):
            # if intersect(pair_1[0], pair_1[1], pair_2[0], pair_2[1]):
            #     print(pair_1, pair_2)
            return True
    return False


"""
// Given three collinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
bool onSegment(Point p, Point q, Point r)
{
    if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
        q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
       return true;
 
    return false;
}
 
// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(Point p, Point q, Point r)
{
    // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    // for details of below formula.
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);
 
    if (val == 0) return 0;  // collinear
 
    return (val > 0)? 1: 2; // clock or counterclock wise
}
 
// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
bool do_intersect(Point p1, Point q1, Point p2, Point q2)
{
    // Find the four orientations needed for general and
    // special cases
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);
 
    // General case
    if (o1 != o2 && o3 != o4)
        return true;
 
    // Special Cases
    // p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;
 
    // p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;
 
    // p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;
 
     // p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;
 
    return false; // Doesn't fall in any of the above cases
}"""


def print_word_distance(word: str, total_length: float, relative_length: float) -> None:
    """
    Prints the distance for a single word
    :param word:
    :param total_length:
    :param relative_length:
    :return:
    """
    print(
        f"{word}: total length: {keyboard_distance_to_cm(total_length)}cm "
        f"({keyboard_distance_to_cm(relative_length)}cm per movement)"
    )


def print_word_angle(word: str, total_angle: float, relative_angle: float) -> None:
    """
    Prints the distance for a single word
    :param word:
    :param total_angle:
    :param relative_angle:
    :return:
    """
    print(
        f"{word}: total angle: {total_angle * 180 / math.pi:2f} degrees "
        f"({relative_angle * 180 / math.pi:2f} degrees per movement)"
    )


def main():
    """
    Entry point
    """
    # # Driver program to test above functions:
    # p1 = Position2D(1, 1)
    # q1 = Position2D(10, 1)
    # p2 = Position2D(1, 2)
    # q2 = Position2D(10, 2)
    #
    # if do_intersect(p1, q1, p2, q2):
    #     print("Yes")
    # else:
    #     print("No")
    #
    # p1 = Position2D(10, 0)
    # q1 = Position2D(0, 10)
    # p2 = Position2D(0, 0)
    # q2 = Position2D(10, 10)
    #
    # if do_intersect(p1, q1, p2, q2):
    #     print("Yes")
    # else:
    #     print("No")
    #
    # p1 = Position2D(-5, -5)
    # q1 = Position2D(0, 0)
    # p2 = Position2D(1, 1)
    # q2 = Position2D(10, 10)
    #
    # if do_intersect(p1, q1, p2, q2):
    #     print("Yes")
    # else:
    #     print("No")
    #
    # print(does_word_intersect_itself("prip"))
    # return

    parser = argparse.ArgumentParser(
        description="Finds the distance of a word on the keyboard"
    )
    parser.add_argument("words", type=str, nargs="*", help="words to get distances for")
    parser.add_argument(
        "--word-count",
        dest="word_count",
        action="store",
        default=10,
        help="The number of words to get",
        type=int,
    )

    parser.add_argument(
        "--larger-than",
        dest="larger_than",
        action="store",
        default=4,
        help="The minimum size of the word",
        type=int,
    )
    parser.add_argument(
        "--smaller-than",
        dest="smaller_than",
        action="store",
        default=math.inf,
        help="The maximum size of the word",
        type=int,
    )
    parser.add_argument(
        "--relative",
        dest="relative",
        action="store_true",
        help="If set, relative distance will be measured instead of total distance",
    )
    parser.add_argument(
        "--lowest-first",
        dest="lowest_first",
        action="store_true",
        help="If set, lowest scores will be weighted higher",
    )
    parser.add_argument(
        "--compare-angle",
        dest="compare_angle",
        action="store_true",
        help="If set, angle between keys will be measured instead of distance",
    )
    parser.add_argument(
        "--non-intersecting",
        dest="non_intersecting",
        action="store_true",
        help="If set, only words that don't intersect with themselves will be used",
    )

    args = parser.parse_args()

    if args.words:
        for word in args.words:
            total_length, relative_length = get_word_traversal_length(word)
            print_word_distance(word, total_length, relative_length)
        return

    create_word_lists()
    best_words: List[Tuple[str, float, float]] = []

    comparison_operator = operator.lt if args.lowest_first else operator.gt

    with open(SIMPLE_WORD_FILE_PATH, encoding="utf-8") as file:
        for word in file.readlines():
            word = word.strip()
            if len(word) < args.larger_than or len(word) > args.smaller_than:
                continue

            if args.non_intersecting and does_word_intersect_itself(word):
                continue

            total_distance, relative_distance = (
                get_word_traversal_angle(word)
                if args.compare_angle
                else get_word_traversal_length(word)
            )
            if (
                len(best_words) < args.word_count
                or (
                    args.relative
                    and comparison_operator(relative_distance, best_words[-1][2])
                )
                or (
                    not args.relative
                    and comparison_operator(total_distance, best_words[-1][1])
                )
            ):
                best_words.append((word, total_distance, relative_distance))
                best_words.sort(
                    key=lambda x: x[2] if args.relative else x[1],
                    reverse=not args.lowest_first,
                )
                if len(best_words) > args.word_count:
                    best_words.pop()

        for group in best_words:
            word, total_distance, relative_distance = group
            if args.compare_angle:
                print_word_angle(word, total_distance, relative_distance)
            else:
                print_word_distance(word, total_distance, relative_distance)


if __name__ == "__main__":
    main()
