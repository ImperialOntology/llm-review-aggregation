sem_eval_nshot_examples_json = [
    f"Review: In the shop, these MacBooks are encased in a soft rubber enclosure - so you will never know about the razor edge until you buy it, get it home, break the seal and use it (very clever con).\n"
    f'Output: {{"aspects": [{{"aspect": "rubber enclosure", "polarity": "positive"}}, {{"aspect": "edge", "polarity": "negative"}}]}}',
    f"Review: I investigated netbooks and saw the Toshiba NB305-N410BL.\n"
    f'Output: {{"aspects": []}}',
    f"Review: Great laptop that offers many great features!\n"
    f'Output: {{"aspects": [{{"aspect": "features", "polarity": "positive"}}]}}',
]

concept_examples = [
    f"Product: Smartphone\n"
    f"Candidate Aspect: battery\n"
    f'Output: {{"answer": "yes"}} (Explanation: Battery is a core component of a smartphone, making it a relevant and specific aspect.)',
    f"Product: Smartphone\n"
    f"Candidate Aspect: fast\n"
    f'Output: {{"answer": "no"}} (Explanation: Fast describes the smartphone\'s performance, but it is not a component or feature of the smartphone.)',
    f"Product: Laptop\n"
    f"Candidate Aspect: laptop bag\n"
    f'Output: {{"answer": "no"}} (Explanation: Although related, a laptop bag is an accessory, not a component or feature of the laptop itself.)',
    f"Product: Laptop\n"
    f"Candidate Aspect: apple\n"
    f'Output: {{"answer": "no"}} (Explanation: Apple refers to a brand of a laptop, but is too specific and does not generalise well across laptop products.)',
    f"Product: Earrings\n"
    f"Candidate Aspect: gift\n"
    f'Output: {{"answer": "no"}} (Explanation: Although earrings can be a gift, it is not a component or feature of the earring itself.)',
]

# examples for the prompt for part-whole relations extraction
selected_context_examples = [
    f'Sentence: only a couple gripes cause im picky.. the sunburst color of the finish was a little too dark.\nAspect1: finish\nAspect2: color\n'
    f"Output: {{\"meronym\": [{{\"part\": \"color\", \"whole\": \"finish\"}}]}}",
    f'Sentence: nice to use on vacation when shopping but fought the straps had to put knots in them to stay on my back.not water proof\nAspect1: water proof\nAspect2: straps\n'
    f"Output: {{\"meronym\": []}}",
    f'Sentence: great laptop, except for the worst keyboard ever almost everything about this laptop is great.\nAspect1: keyboard\nAspect2: laptop\n'
    f"Output: {{\"meronym\": [{{\"part\": \"keyboard\", \"whole\": \"laptop\"}}]}}",
    f'Sentence: i am also attaching an image taken of a tree in the sunlight so you can see the dynamic range and how the camera handles sun flares.all images are using default camera\'s settings except i switched to \"fine\" compression, the default is \"normal\", and no images were post processed.\nAspect1: camera\nAspect2: settings\n'
    f"Output: {{\"meronym\": [{{\"part\": \"settings\", \"whole\": \"camera\"}}]}}",
    f'Sentence: good buy really happy with  the style and color.\nAspect1: color\nAspect2: style\n'
    f"Output: {{\"meronym\": []}}"
]
