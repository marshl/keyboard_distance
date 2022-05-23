# keyboard_distance

Python tool to determine the distance on a keyboard that word
Inspired by _Why the longest English word is PAPAL and SPA is the pointiest._ By Matt Parker on __Stand Up Maths__
(https://www.youtube.com/watch?v=Mf2H9WZSIyw&t=845s)

# Table of Contents

1. [About the Project](#about-the-project)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
4. [Contributing](#contributing)

## About The Project

This is a script that can be used to figure out the keyboard travel distance of a word, as well as finding the word with
the smallest or largest distance among a group of words. It can also find words that don't have lines between letters
that intersect, and also find the angle between the lines and the smallest or largest angles of all words.

This project was inspired by the video _Why the longest English word is PAPAL and SPA is the pointiest._ by Matt Parker
on his YouTube Channel **Stand-Up Maths** https://www.youtube.com/watch?v=Mf2H9WZSIyw&t=845s

## Getting Started

This script is written to not require any third-party dependencies, so you don't need to install any modules or create a
virtualenv, however it is still recommended that you do so.

### Prerequisites

Before you start you will need to make sure you have the following installed:

- Python 3.8

### Installation

1. Clone the repo and `cd` into it
   ```sh
   git clone https://github.com/marshl/keyboard_distance.git
   cd keyboard_distance
   ```
2. Create a virtual environment
   ```sh
   python -m virtualenv --python $(which python3.8) venv
   ```

## Usage

### First usage

The first time you run `keyboard_distance` it will download a list of words from `dwyl` on GitHub
here https://github.com/dwyl/english-words It will then also create a "simplified" list, that is a list with only words
that only contain letters (words with numbers, apostrophes, hyphens etc. are excluded) in order to speed up processing.

Run the program with:

```shell
python keyboard_distance
```

Your firewall may block the word file being downloaded, if it does, you can download the words manually from
here https://raw.githubusercontent.com/dwyl/english-words/master/words.txt and the put that file in the same directory
you execute the script from named `words.txt`. The simplified file will then be generated as normal.

### Simple usage

Running the program without any arguments will by default find words with the longest total travel distance

```shell
python keyboard_distance
```

It will give the answer in the form of a list of words with the total travel distance in centimeters, and the average
distance per movement (a movement is the distance from one key to another, so a word with 4 letters has 3 movements)

Instead of using all English words, you can provide your only list of words after the program name:
```shell
python keyboard_distance the quick onyx goblin jumps over the lazy dwarf
```
For a full list of optional arguments, like searching by angle use the help command
```shell
python keyboard_distance --help
```

### Advanced usage

Here are some more advanced usage examples:

Finding the largest non-intersecting word:
```shell
python keyboard_distance --non-intersecting
```

Finding 100 words with the greatest angle between all movements:
```shell
python keyboard_distance --word-count=100 --compare-angle --smaller-than=10
```

## Contributing

I'm not sure what to put here. Create a pull request if you want I suppose? Or raise issue or something.

### Linting

To lint the project, run (you will need to install pylint into your virtual environment first).

```shell
python -m pylint keyboard_distance test 
```

All committed code should have first been auto-formatted with black. This repo doesn't have a pre-commit hook, but
please try to remember to run black before committing:

```shell
python -m black keyboard_distance/__main__.py
```

### Tests

To run the tests, run

```shell
python -m pytest
```
