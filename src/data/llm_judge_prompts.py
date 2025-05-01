ASPECT_JUDGE_PROMPT = '''
        You are an AI judge tasked with evaluating the correctness of terms within an Amazon product ontology.
        Specifically, you will assess whether a given term is appropriate and correctly categorized as a part, component (meronym) or attribute of a specified product.
        The terms should be general enough to represent common parts, components or attributes across different listings of the product.
        The ontology follows a meronymic structure, where terms should represent parts, components or attributes that logically fit within their associated products.
        Strictly follow this format to output the scores, followed by the explanation: Score: [[1-5]], e.g., Score: [[1]].

        Evaluation Criteria:
        Relevance: Does the term accurately represent a part, component or attribute of the specified product?
        Specificity: Is the term general enough to be meaningful within the context of the product, avoiding overly specific terms?
        Clarity: Does the term clearly convey the intended part, component or attribute, avoiding ambiguity?
        Product Fit: Is the term logically and contextually appropriate for the given product?

        Score 1: Completely incorrect, irrelevant, or overly specific term for the product. (e.g., Product: Smartphones, Term: Laptop)
        Score 2: Poorly fitting term with minimal relevance or appropriateness, or overly specific for the product. (e.g., Product: Smartphones, Term: diamond bling phone cover)
        Score 3: Fairly appropriate term with some relevance, but it may lack specificity, clarity, or perfect fit within the product. (e.g., Product: Smartphones, Term: Box)
        Score 4: Good term with relevance and a logical fit, but it may have slight ambiguities. (e.g., Product: Smartphones, Term: Features)
        Score 5: Excellent term that is highly relevant, specific enough, clear, general, and a perfect fit for the product. (e.g., Product: Smartphones, Term: Screen size)

        Term to evaluate:
        Product: {product}
        Term: {term}
        '''

RELATION_JUDGE_PROMPT = '''
        You are an AI judge evaluating the correctness of meronym (part-whole) and attribute (property-characteristic) relations within an Amazon product ontology.
        Your task is to score the given child-parent node relations based on how well the child node represents a part, property, or characteristic of the specified parent node.
        For each relation, you will analyze whether the child node logically and hierarchically fits as a part or attribute of the parent node in the context of the product category {category}.
        Strictly follow this format to output the scores, followed by the explanation: Score: [[1-5]], e.g., Score: [[1]].

        Evaluation Criteria:
        Logical Hierarchy: Does the child node represent a logical part, property, or characteristic of the parent node?
        Contextual Fit: Is the relation reasonable within the context of product categories commonly found on Amazon? Consider attributes relevant to listings, but allow flexibility for less common, yet valid, relationships.
        Clarity and Specificity: Does the relation avoid ambiguity and clearly define the part-whole or attribute-characteristic relationship? Acknowledge general, but correct, relations even if they lack specific detail.

        Score 1: Completely incorrect relation with no logical or contextual fit. (e.g., Child Node: apple, Parent Node: car)
        Score 2: Poor relation with minimal logical or contextual fit. (e.g., Child Node: van, Parent Node: bike helmet)
        Score 3: Fair relation with some logical fit but lacks strong contextual relevance or clarity. (e.g., Child Node: book, Parent Node: school)
        Score 4: Good relation with a logical and contextual fit but may have slight ambiguities. (e.g., Child Node: features, Parent Node: vehicle)
        Score 5: Excellent relation with a clear, logical, and contextual fit, with no ambiguities. (e.g., Child Node: chapter, Parent Node: book)

        Relation to evaluate:
        Child Node: {child}
        Parent Node: {parent}
        '''