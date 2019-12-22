import csv
import json
import nltk

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import *
from nltk.stem.util import suffix_replace


def word_tokenize(text):
    return nltk.re.findall(r"""(
            \w+         # word 
            |\.\.\.     # elipsis
            |\n{2,}     # white lines
            |-+         # dashes
            |[(){}[\]]  # brackets
            |[!?.,:;]   # punctuation
            |['"`]      # quotes
            |\S+        # fall through pattern
            )""",
                           text, nltk.re.VERBOSE)


def lemmatize(word, pos):
    # Need to specify pos=NOUN|ADJ|VERB|
    if pos == None:
        lemmas = wordnet._morphy(word, NOUN)
    else:
        lemmas = wordnet._morphy(word, pos)
    return min(lemmas, key=len) if lemmas else word


def stem(word):
    return word.lower()


def word_stem(word):
    vowels = "aeiouy"  # The vowels here includes y, we will deal with the case later.
    double_consonants = ("bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt")
    li_ending = "cdeghkmnrt"
    step0_suffixes = ("'s'", "'s", "'")
    step1a_suffixes = ("sses", "ied", "ies", "us", "ss", "s")
    step1b_suffixes = ("eedly", "ingly", "edly", "eed", "ing", "ed")
    step2_suffixes = ('ization', 'ational', 'fulness', 'ousness', 'iveness', 'tional', 'biliti', 'lessli', 'entli',
                      'ation', 'alism', 'aliti', 'ousli', 'iviti', 'fulli', 'enci', 'anci',
                      'abli', 'izer', 'ator', 'alli', 'bli', 'ogi', 'li')
    step3_suffixes = ('ational', 'tional', 'alize', 'icate', 'iciti', 'ative', 'ical', 'ness', 'ful')
    step4_suffixes = ('ement', 'ance', 'ence', 'able', 'ible', 'ment', 'ant', 'ent',
                      'ism', 'ate', 'iti', 'ous', 'ive', 'ize', 'ion', 'al', 'er', 'ic')

    word = word.lower()

    if word in stopwords.words('english'):
        return word

    # remove starting '
    if word.startswith("\x27"):
        word = word[1:]

    """
    Special cases with Y's:
    3 cases Y considered as a vowel, according to Merriam-Webster
    If Y is at the end of a words, we consider this Y as a vowel. e.g.: candy, deny
    If the word has no other vowels than Y, Y is considered as a vowel. e.g. gym
    If Y is in the middle of a syllable. e.g. system, borborygmus
    Thus, we will find the non-vowel y's, and replace them with Y as distinguish.
    """

    # We need to find the Y's, since Y is a special.
    # If a word starts with a y, it is not considered as a vowel.
    # Find starting Y
    if word.startswith("y"):
        word = "".join(("Y", word[1:]))

    # Find vowel + y
    # If any y follows a vowel, that Y is not considered as a vowel.
    for i in range(1, len(word)):
        if word[i - 1] in vowels and word[i] == "y":
            word = "".join((word[:i], "Y", word[i + 1:]))

    step1a_vowel_found = False
    step1b_vowel_found = False

    r1 = ""
    r2 = ""

    # R1 is the region after the first non-vowel following a vowel,
    # or is the null region at the end of the word if there is no
    # such non-vowel.
    #
    # R2 is the region after the first non-vowel following a vowel
    # in R1, or is the null region at the end of the word if there
    # is no such non-vowel.
    if word.startswith(("gener", "commun", "arsen")):
        if word.startswith(("gener", "arsen")):
            r1 = word[5:]
        else:
            r1 = word[6:]

        for i in range(1, len(r1)):
            if r1[i] not in vowels and r1[i - 1] in vowels:
                r2 = r1[i + 1:]
                break
    else:
        for i in range(1, len(word)):
            if word[i] not in vowels and word[i - 1] in vowels:
                r1 = word[i + 1:]
                break

        for i in range(1, len(r1)):
            if r1[i] not in vowels and r1[i - 1] in vowels:
                r2 = r1[i + 1:]
                break

    # Step 0
    # Remove the suffixes 's, s', '
    # The single -s suffix and possessives
    for suffix in step0_suffixes:
        if word.endswith(suffix):
            word = word[: -len(suffix)]
            r1 = r1[: -len(suffix)]
            r2 = r2[: -len(suffix)]
            break

    # Step 1a
    # Deal with "regular" suffix, such as ied, ies, sses
    for suffix in step1a_suffixes:
        if word.endswith(suffix):
            if suffix == 'sses':
                word = word[:-2]
                r1 = r1[:-2]
                w2 = r2[:-2]
            elif suffix in ("ied", "ies"):
                # For regular words, we remove the last 2 letter
                if len(word[: -len(suffix)]) > 1:
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]
                else:
                    # For short words, like pies, we only remove s.
                    word = word[:-1]
                    r1 = r1[:-1]
                    r2 = r2[:-1]
            # When suffix of this word is just s, we remove the last letter
            elif suffix == "s":
                word = word[:-1]
                r1 = r1[:-1]
                r2 = r2[:-1]
            break

    # Step 1b
    for suffix in step1b_suffixes:
        if word.endswith(suffix):
            if suffix in ("eed", "eedly"):
                if r1.endswith(suffix):
                    word = word[:-len(suffix)] + "ee"
                    # word = suffix_replace(word, suffix, "ee")
                    if len(r1) >= len(suffix):
                        r1 = r1[:-len(suffix)] + 'ee'
                    else:
                        r1 = ""

                    if len(r2) >= len(suffix):
                        r1 = r2[:-len(suffix)] + 'ee'
                    else:
                        r2 = ""
                else:
                    # For ed, edly+, ing, ingly part.
                    for letter in word[: -len(suffix)]:
                        if letter in vowels:
                            step1b_vowel_found = True
                            break
                    # If such suffix are found, we delete the the suffix.
                    if step1b_vowel_found:
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]

                        # After deletion
                        # If end with at, bl, iz, we add a e, and make to ate, ble, ize.
                        if word.endswith(("at", "bl", "iz")):
                            word = "".join((word, "e"))
                            r1 = "".join((r1, "e"))

                            if len(word) > 5 or len(r1) >= 3:
                                r2 = "".join((r2, "e"))

                        # If end with double consonants, we delete one consonant
                        elif word.endswith(double_consonants):
                            word = word[:-1]
                            r1 = r1[:-1]
                            r2 = r2[:-1]

                        # If the word is short we add e
                        elif (
                                r1 == ""
                                and len(word) >= 3
                                and word[-1] not in vowels
                                and word[-1] not in "wxY"
                                and word[-2] in vowels
                                and word[-3] not in vowels
                        ) or (
                                r1 == ""
                                and len(word) == 2
                                and word[0] in vowels
                                and word[1] not in vowels
                        ):

                            word = "".join((word, "e"))

                            if len(r1) > 0:
                                r1 = "".join((r1, "e"))

                            if len(r2) > 0:
                                r2 = "".join((r2, "e"))
            break

    # STEP 1c
    # If word now ends with Y or y, we replace y with i.
    if len(word) > 2 and word[-1] in "yY" and word[-2] not in vowels:
        word = "".join((word[:-1], "i"))
        if len(r1) >= 1:
            r1 = "".join((r1[:-1], "i"))
        else:
            r1 = ""

        if len(r2) >= 1:
            r2 = "".join((r2[:-1], "i"))
        else:
            r2 = ""

    # Step 2
    # In step 2, we go through each of the suffix, and replace them with disired ending.
    # These suffix are
    for suffix in step2_suffixes:
        if word.endswith(suffix):
            if r1.endswith(suffix):
                if suffix == "tional":
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]

                elif suffix in ("enci", "anci", "abli"):
                    word = "".join((word[:-1], "e"))

                    if len(r1) >= 1:
                        r1 = "".join((r1[:-1], "e"))
                    else:
                        r1 = ""

                    if len(r2) >= 1:
                        r2 = "".join((r2[:-1], "e"))
                    else:
                        r2 = ""

                elif suffix == "entli":
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]

                elif suffix in ("izer", "ization"):
                    word = word[:-len(suffix)] + 'ize'

                    if len(r1) >= len(suffix):
                        r1 = r1[:-len(suffix)] + 'ize'
                    else:
                        r1 = ""

                    if len(r2) >= len(suffix):
                        r2 = r2[:-len(suffix)] + 'ize'
                    else:
                        r2 = ""

                elif suffix in ("ational", "ation", "ator"):
                    word = word[:-len(suffix)] + 'ate'
                    if len(r1) >= len(suffix):
                        r1 = r1[:-len(suffix)] + 'ate'
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = r2[:-len(suffix)] + 'ate'
                    else:
                        r2 = "e"

                elif suffix in ("alism", "aliti", "alli"):
                    word = suffix_replace(word, suffix, "al")

                    if len(r1) >= len(suffix):
                        r1 = suffix_replace(r1, suffix, "al")
                    else:
                        r1 = ""

                    if len(r2) >= len(suffix):
                        r2 = suffix_replace(r2, suffix, "al")
                    else:
                        r2 = ""

                elif suffix == "fulness":
                    word = word[:-4]
                    r1 = r1[:-4]
                    r2 = r2[:-4]

                elif suffix in ("ousli", "ousness"):
                    word = suffix_replace(word, suffix, "ous")

                    if len(r1) >= len(suffix):
                        r1 = suffix_replace(r1, suffix, "ous")
                    else:
                        r1 = ""

                    if len(r2) >= len(suffix):
                        r2 = suffix_replace(r2, suffix, "ous")
                    else:
                        r2 = ""

                elif suffix in ("iveness", "iviti"):
                    word = suffix_replace(word, suffix, "ive")

                    if len(r1) >= len(suffix):
                        r1 = suffix_replace(r1, suffix, "ive")
                    else:
                        r1 = ""

                    if len(r2) >= len(suffix):
                        r2 = suffix_replace(r2, suffix, "ive")
                    else:
                        r2 = "e"

                elif suffix in ("biliti", "bli"):
                    word = suffix_replace(word, suffix, "ble")

                    if len(r1) >= len(suffix):
                        r1 = suffix_replace(r1, suffix, "ble")
                    else:
                        r1 = ""

                    if len(r2) >= len(suffix):
                        r2 = suffix_replace(r2, suffix, "ble")
                    else:
                        r2 = ""

                elif suffix == "ogi" and word[-4] == "l":
                    word = word[:-1]
                    r1 = r1[:-1]
                    r2 = r2[:-1]

                elif suffix in ("fulli", "lessli"):
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]

                elif suffix == "li" and word[-3] in li_ending:
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]
            break

    # Step 3
    for suffix in step3_suffixes:
        if word.endswith(suffix):
            if r1.endswith(suffix):
                if suffix == "tional":
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]
                elif suffix == "ational":
                    word = suffix_replace(word, suffix, "ate")
                    if len(r1) >= len(suffix):
                        r1 = suffix_replace(r1, suffix, "ate")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = suffix_replace(r2, suffix, "ate")
                    else:
                        r2 = ""
                elif suffix == "alize":
                    word = word[:-3]
                    r1 = r1[:-3]
                    r2 = r2[:-3]
                elif suffix in ("icate", "iciti", "ical"):
                    word = suffix_replace(word, suffix, "ic")
                    if len(r1) >= len(suffix):
                        r1 = suffix_replace(r1, suffix, "ic")
                    else:
                        r1 = ""
                    if len(r2) >= len(suffix):
                        r2 = suffix_replace(r2, suffix, "ic")
                    else:
                        r2 = ""
                elif suffix in ("ful", "ness"):
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                elif suffix == "ative" and r2.endswith(suffix):
                    word = word[:-5]
                    r1 = r1[:-5]
                    r2 = r2[:-5]
            break

    # Step 4
    for suffix in step4_suffixes:
        if word.endswith(suffix):
            if r2.endswith(suffix):
                if suffix == "ion":
                    if word[-4] in "st":
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
            break

    # Step 5
    if r2.endswith("l") and word[-2] == "l":
        word = word[:-1]
    elif r2.endswith("e"):
        word = word[:-1]
    elif r1.endswith("e"):
        if len(word) >= 4 and (
                word[-2] in vowels
                or word[-2] in "wxY"
                or word[-3] not in vowels
                or word[-4] in vowels
        ):
            word = word[:-1]

    word = word.replace("Y", "y")

    return word


def read_json(file):
    with open(file) as f:
        return json.load(f)


def write_json(file, obj):
    with open(file, 'w') as f:
        json.dump(obj, f)


def csv2dict(file):
    dic = {}
    with open(file, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            dic[idx] = dict(row)
            if idx == 5000:
                break
    return dic


def read_cvs(path):
    song_lyrics = []
    with open(path) as file:
        songdata = csv.reader(file, delimiter=",")
        for row in songdata:
            song_lyrics.append(row[3])
    return song_lyrics[1:-1]


if __name__ == '__main__':
    pass