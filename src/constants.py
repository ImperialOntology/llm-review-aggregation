PROMPT_BASED_LLM_MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'

CACHE_DIR = '/vol/project/2024/mai/g24mai05/cache'

DISNEYLAND_CATEGORY = 'Disneyland'
CATEGORY_MAPPING = {
    "Stand Mixers": {
        'category': 'Home_and_Kitchen',
        'sub_categories': ['Stand Mixers'],
        'ontology_root': 'mixer',
    },
    "Games": {
        'category': 'Video_Games',
        'sub_categories': ['Games'],
        'ontology_root': 'game',
    },
    "Televisions": {
        'category': 'Electronics',
        'sub_categories': ['Television & Video', 'Televisions'],
        'ontology_root': 'tv',
    },
    "Wrist Watches": {
        'category': 'Clothing_Shoes_and_Jewelry',
        'sub_categories': ['Watches', 'Wrist Watches'],
        'ontology_root': 'watch',
    },
    "Necklaces": {
        'category': 'Clothing_Shoes_and_Jewelry',
        'sub_categories': ['Necklaces'],
        'ontology_root': 'necklace',
    },
    "Disneyland": {
        'category': 'Disneyland',
        'sub_categories': ['Disneyland'],
        'ontology_root': 'place',
    },
}


# BERT ontology extraction configs START

BERT_ENTITY_DATASET_PATH = 'bert-train-data/term_extraction_datasets'
BERT_RELATION_DATASET_PATH = 'bert-train-data/relation_extraction_datasets'
BERT_TRAIN_CATEGORIES = ['camera', 'backpack', 'cardigan', 'guitar', 'laptop']
ENTITY_NUM_CLASSES = 2  # entity, non-entity
RELATION_NUM_CLASSES = 3  # no relation, fst hasFeature snd, snd hasFeature fst
HIDDEN_OUTPUT_FEATURES = 768
TRAINED_WEIGHTS = 'bert-base-uncased'
MASK_TOKEN = '[MASK]'
ENTITY_PROB_THRESHOLD = 0.65
N_ASPECTS = 100
MAX_SEQ_LEN = 128
# optimizer parameters
DECAY_RATE = 0.01
LEARNING_RATE = 0.00002
MAX_GRAD_NORM = 1.0
# training parameters
N_EPOCHS = 3
WARM_UP_FRAC = 0.05
PHRASE_THRESHOLD = 4

# BERT ontology extraction configs END
