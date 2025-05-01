import re
import string
from nltk import pos_tag
from nltk.corpus import stopwords


stop_words = stopwords.words('english')


def entity_instances_for_text(tokenizer, entities, text):
    """
    Extract entity instances from a given text using a tokenizer.

    Args:
        tokenizer: The tokenizer to tokenize the text.
        entities (list): A list of entity strings to match in the text.
        text (str): The input text to process.

    Returns:
        tuple or None: A tuple containing:
            - tokens (list): The tokenized text with entity mentions masked.
            - e_range (int): The index of the masked entity.
            - entity (str): The matched entity string.
        Returns None if no valid entity instance is found.
    """
    def joined_tokens(tokens, entity_range):
        """
        Join tokens into words based on entity ranges.

        Args:
            tokens (list): The tokenized text.
            entity_range (range): The range of indices for the entity.

        Returns:
            tuple: A tuple containing:
                - joined (list): The joined tokens.
                - entity_idx (int): The index of the entity in the joined tokens.
        """
        joined = []
        j_token = tokens[0]
        entity_idx = None
        for idx, t in enumerate(tokens):
            if idx == 0:
                continue
            if t.startswith('##'):
                # continuing same word
                j_token += t[2:]
            elif idx in entity_range and idx - 1 in entity_range:
                # continuing same multi-word entity
                j_token = j_token + " " + t
            else:
                # new word
                if idx - 1 in entity_range:
                    assert entity_idx is None
                    entity_idx = len(joined)
                joined.append(j_token)
                j_token = t
        if j_token:
            if len(tokens) - 1 in entity_range:
                assert entity_idx is None
                entity_idx = len(joined)
            joined.append(j_token)
        assert entity_idx is not None
        return joined, entity_idx

    def is_noun(tokens, entity_mention):
        """
        Check if an entity mention is a noun.

        Args:
            tokens (list): The tokenized text.
            entity_mention (tuple): The entity mention range and string.

        Returns:
            bool: True if the entity is a noun, False otherwise.
        """
        entity_range = range(entity_mention[0][0] - 1, entity_mention[0][1])
        joined, entity_idx = joined_tokens(tokens, entity_range)
        tags = [tag for _, tag in pos_tag(joined)]
        return tags[entity_idx].startswith('NN')

    def token_entity_match(first_token_idx, entity, tokens):
        """
        Match an entity to tokens starting from a specific index.

        Args:
            first_token_idx (int): The starting index in the tokens.
            entity (str): The entity string to match.
            tokens (list): The tokenized text.

        Returns:
            int or None: The length of the match if successful, otherwise None.
        """
        token_idx = first_token_idx
        remaining_entity = entity
        while remaining_entity:
            if remaining_entity == entity or remaining_entity.lstrip() != remaining_entity:
                # start of new word
                remaining_entity = remaining_entity.lstrip()
                if token_idx < len(tokens) and tokens[token_idx] == remaining_entity[:len(tokens[token_idx])]:
                    remaining_entity = remaining_entity[len(tokens[token_idx]):]
                    token_idx += 1
                else:
                    break
            else:
                # continuing same word
                if (token_idx < len(tokens) and tokens[token_idx].startswith('##')
                        and tokens[token_idx][2:] == remaining_entity[:len(tokens[token_idx][2:])]):
                    remaining_entity = remaining_entity[len(tokens[token_idx][2:]):]
                    token_idx += 1
                else:
                    break
        if remaining_entity or (token_idx < len(tokens) and tokens[token_idx].startswith('##')):
            return None
        else:
            return token_idx - first_token_idx

    def mask_tokens(tokens, entity_range, mask):
        """
        Mask tokens in a specific range.

        Args:
            tokens (list): The tokenized text.
            entity_range (tuple): The start and end indices of the entity range.
            mask (str): The mask token to replace the entity.

        Returns:
            list: The tokenized text with the entity masked.
        """
        start, end = entity_range
        tokens[start - 1] = mask
        return [t for idx, t in enumerate(tokens)
                if not idx in range(start, end)]

    tokens = tokenizer.tokenize(text)

    entity_mention = None
    for i in range(len(tokens)):
        for entity in entities:
            match_length = token_entity_match(i, entity, tokens)
            if match_length is not None:
                e_range = (i + 1, i + match_length)
                if entity_mention is not None:
                    if e_range[0] >= entity_mention[0][0] and e_range[1] <= entity_mention[0][1]:
                        # sub-entity
                        continue
                    elif not e_range[0] <= entity_mention[0][0] and e_range[1] >= entity_mention[0][1]:
                        return None
                entity_mention = (e_range, entity)  # + 1 taking into account the [CLS] token

    if entity_mention is None or not is_noun(tokens, entity_mention):
        return None

    e_range, entity = entity_mention
    tokens = mask_tokens(tokens, e_range, '[MASK]')  # mask entity mentions

    return tokens, e_range[0], entity


def ngrams(text, phraser):
    """
    Generate n-grams from a given text using a phraser and filter them based on part-of-speech tags.

    Args:
        text (list): The input text as a list of tokens.
        phraser: The phraser object used to generate n-grams.

    Returns:
        list: A list of n-grams filtered by part-of-speech tags.
    """
    if any(isinstance(subtext, list) for subtext in text):
        return

    tags = [tag for _, tag in pos_tag(text)]

    # Generate n-grams using the phraser
    unfiltered = [term.split('_') for term in phraser[text]]

    # Tag each n-gram with its corresponding part-of-speech tags
    tagged_unfiltered = []
    n = 0
    for term in unfiltered:
        tagged_unfiltered.append([(subterm, tags[n + idx]) for idx, subterm in enumerate(term)])
        n += len(term)

    def filter_ngram(term):
        """
        Filter n-grams based on part-of-speech tags.

        Args:
            term (list): A list of tuples containing words and their tags.

        Returns:
            list: A filtered list of n-grams.
        """
        # If the n-gram contains more than one word and any word is not a noun or adjective, split it
        if len(term) > 1 and any(not re.compile('NN|JJ').match(tag) for _, tag in term):
            return [subterm for subterm, _ in term]
        # Otherwise, join the n-gram into a single string
        return [' '.join([subterm for subterm, _ in term])]

    return [subterm for term in tagged_unfiltered for subterm in filter_ngram(term)]


def get_nouns(phrase, ngrams):
    """
    Extract nouns from a phrase using n-grams and part-of-speech tagging.

    Args:
        phrase (list): The input phrase as a list of tokens.
        ngrams (list): A list of n-grams generated from the phrase.

    Returns:
        list: A list of extracted nouns that meet the criteria for validity.
    """
    pos_tags = pos_tag(phrase)

    def is_noun(pos_tagged):
        """
        Check if a token is a noun.

        Args:
            pos_tagged (tuple): A tuple containing a word and its part-of-speech tag.

        Returns:
            bool: True if the token is a noun and not a punctuation or stop word, False otherwise.
        """
        word, tag = pos_tagged
        return tag.startswith('NN') and word not in string.punctuation and word not in stop_words

    def is_valid_term(pos_tagged):
        """
        Check if a token is a valid term.

        Args:
            pos_tagged (tuple): A tuple containing a word and its part-of-speech tag.

        Returns:
            bool: True if the token is valid (not a preposition and contains only alphanumeric characters), False otherwise.
        """
        alpha_numeric_pat = '^\w+$'
        word, tag = pos_tagged
        return tag != 'IN' and re.match(alpha_numeric_pat, word)

    nouns = []
    word_idx = 0
    for token in ngrams:
        if ' ' in token:
            # Multi-word n-gram
            words = token.split(' ')
            word_range = range(word_idx, word_idx + len(words))
            has_noun = any(is_noun(pos_tags[i]) for i in word_range)
            all_terms_valid = all(is_valid_term(pos_tags[i]) for i in word_range)
            if has_noun and all_terms_valid:
                nouns.append(token)
            word_idx += len(words)
        else:
            # Single-word token
            token_is_noun = is_noun(pos_tags[word_idx])
            is_valid = is_valid_term(pos_tags[word_idx])
            if len(token) > 1 and token_is_noun and is_valid:
                nouns.append(token)
            word_idx += 1
    return nouns


def relation_instances_for_text(tokenizer, aspects, syn_dict, text):
    """
    Extract relation instances from a given text using a tokenizer, aspects, and their synonyms.

    Args:
        tokenizer: The tokenizer to tokenize the text.
        aspects (list): A list of aspects to match in the text.
        syn_dict (dict): A dictionary mapping aspects to their synonyms.
        text (str): The input text to process.

    Returns:
        tuple or None: A tuple containing:
            - tokens (list): The tokenized text with aspect mentions masked.
            - aspect_indices (list): The indices of the masked aspects.
            - aspect_ids (list): The IDs of the aspects corresponding to the masked mentions.
        Returns None if no valid relation instances are found.
    """
    def joined_tokens(tokens, entity_ranges):
        """
        Join tokens into words based on entity ranges.

        Args:
            tokens (list): The tokenized text.
            entity_ranges (list): A list of ranges representing entity spans.

        Returns:
            list: A list of tuples containing:
                - joined token (str)
                - start index (int)
                - end index (int)
        """
        joined = []
        j_token = tokens[0]
        start = 0
        for idx, t in enumerate(tokens):
            if idx == 0:
                continue
            if t.startswith('##'):
                # continuing same word
                j_token += t[2:]
            elif any(idx in r and idx-1 in r for r in entity_ranges):
                # continuing same multi-word entity
                j_token = j_token + " " + t
            else:
                # new word
                joined.append((j_token, start, idx))
                j_token = t
                start = idx
        if j_token:
            joined.append((j_token, start, len(tokens)))
        return joined

    def noun_entity_mentions(tokens, entity_mentions):
        """
        Filter entity mentions to include only those that are nouns.

        Args:
            tokens (list): The tokenized text.
            entity_mentions (list): A list of entity mentions.

        Returns:
            list: A list of entity mentions that are nouns.
        """
        entity_ranges = [range(em[0][0]-1, em[0][1]) for em in entity_mentions]
        joined = joined_tokens(tokens, entity_ranges)
        tags = [tag for _, tag in pos_tag([t for t, _, _ in joined])]
        noun_ranges = [range(start, end) for idx, (_, start, end) in enumerate(joined) if tags[idx].startswith('NN')]
        return [em for idx, em in enumerate(entity_mentions) if entity_ranges[idx] in noun_ranges]

    def token_entity_match(first_token_idx, entity, tokens):
        """
        Match an entity to tokens starting from a specific index.

        Args:
            first_token_idx (int): The starting index in the tokens.
            entity (str): The entity string to match.
            tokens (list): The tokenized text.

        Returns:
            int or None: The length of the match if successful, otherwise None.
        """
        token_idx = first_token_idx
        remaining_entity = entity
        while remaining_entity:
            if remaining_entity == entity or remaining_entity.lstrip() != remaining_entity:
                # start of new word
                remaining_entity = remaining_entity.lstrip()
                if token_idx < len(tokens) and tokens[token_idx] == remaining_entity[:len(tokens[token_idx])]:
                    remaining_entity = remaining_entity[len(tokens[token_idx]):]
                    token_idx += 1
                else:
                    break
            else:
                # continuing same word
                if (token_idx < len(tokens) and tokens[token_idx].startswith('##')
                        and tokens[token_idx][2:] == remaining_entity[:len(tokens[token_idx][2:])]):
                    remaining_entity = remaining_entity[len(tokens[token_idx][2:]):]
                    token_idx += 1
                else:
                    break
        if remaining_entity or (token_idx < len(tokens) and tokens[token_idx].startswith('##')):
            return None
        else:
            return token_idx - first_token_idx

    def mask_tokens(tokens, entity_mentions):
        """
        Mask tokens corresponding to entity mentions.

        Args:
            tokens (list): The tokenized text.
            entity_mentions (list): A list of entity mentions.

        Returns:
            tuple: A tuple containing:
                - masked tokens (list)
                - mask indices (list): The indices of the masked tokens.
        """
        mask_indices = []
        del_indices = []  # accumulates indices of deleted tokens in multi-token ranges
        for (start, end), _ in entity_mentions:
            tokens[start-1] = '[MASK]'
            mask_indices.append(start - len(del_indices))
            del_indices += list(range(start, end))
        return [t for idx, t in enumerate(tokens) if idx not in del_indices], mask_indices

    tokens = tokenizer.tokenize(text)

    aspect_mentions = set()
    for i in range(len(tokens)):
        for idx, a in enumerate(aspects):
            for syn in syn_dict[a]:
                match_length = token_entity_match(i, syn, tokens)
                if match_length is not None:
                    aspect_mentions.add(((i + 1, i + match_length), idx))  # + 1 taking into account the [CLS] token

    if len(aspect_mentions) < 2:
        return None

    # filter out overlapping aspects
    aspect_mentions = [((s1, e1), a1) for (s1, e1), a1 in aspect_mentions
                       if not any(a1 != a2 and s1 >= s2 and e1 <= e2 for (s2, e2), a2 in aspect_mentions)]

    # filter out non-nouns
    aspect_mentions = noun_entity_mentions(tokens, aspect_mentions)

    # terms cannot refer to the same aspect
    if len(aspect_mentions) != 2 or aspect_mentions[0][1] == aspect_mentions[1][1]:
        return None

    entity_mentions = sorted(aspect_mentions, key=lambda em: em[0])
    tokens, aspect_indices = mask_tokens(tokens, aspect_mentions)  # mask entity mentions

    return tokens, aspect_indices, [em[1] for em in entity_mentions]
