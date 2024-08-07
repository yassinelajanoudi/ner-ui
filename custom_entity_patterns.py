import spacy
from spacy.pipeline import EntityRuler

def add_custom_entities(nlp, patterns):
    """
    Add custom entities to the spaCy pipeline.

    Parameters:
    nlp (spacy.Language): The spaCy pipeline.
    patterns (list of dict): List of patterns to add.
    
    Returns:
    spacy.Language: The updated spaCy pipeline with added entity patterns.
    """
    # Ensure the entity ruler is added only once
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")
    
    # Add patterns to the entity ruler
    ruler.add_patterns(patterns)
    
    return nlp

# Example usage
nlp = spacy.load("en_core_web_trf")
patterns = [
    {"label": "ORG", "pattern": "Chemonics"},
    {"label": "ORG", "pattern": [{"LOWER": "youth"}, {"LOWER": "livelihoods"}]}
]
nlp = add_custom_entities(nlp, patterns)
