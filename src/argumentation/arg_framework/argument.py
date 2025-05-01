class Argument:
    """
    Class representing an argument in the argumentation framework.
    Attributes:
        text (str): The text of the strongest supporting phrase.
        polarity (str): The polarity of the argument ('POS' or 'NEG').
        supporters (list): List of supporting arguments.
        attackers (list): List of attacking arguments.
        phrase (str): The Phrase object corresponding to strongest supporting phrase.
        size (int): The size of the argument used for plotting purposes.
    """

    def __init__(self, text, polarity, supporters, attackers, phrase, size):
        self.text = text
        self.polarity = 'POS' if polarity else 'NEG'
        self.supporters = supporters
        self.attackers = attackers
        self.phrase = phrase.text if phrase else '-'
        self.size = size
