from datetime import datetime
import re
import spacy
from word2number import w2n
from custom_entity_patterns import add_custom_entities

nlp = spacy.load("en_core_web_trf")
patterns = [
    {"label": "ORG", "pattern": "Chemonics"},
    {"label": "ORG", "pattern": [{"LOWER": "youth"}, {"LOWER": "livelihoods"}]}
]
nlp = add_custom_entities(nlp, patterns)

def word_to_number(word):
    try:
        return w2n.word_to_num(word)
    except ValueError:
        return word
    
def parse_relative_date(date_str, doc):
    current_year = datetime.now().year
    date_str = " ".join([str(word_to_number(word)) for word in date_str.split(" ")])
    # Patterns for past relative dates
    past_match = re.search(
        r"(\d+)\s*(year|years|decade|decades|century|centuries) ago|the (last|past|previous) (\d+)\s*(year|years|decade|decades|century|centuries)",
        date_str,
    )
    if past_match:
        num = int(past_match.group(1) or past_match.group(4))
        unit = past_match.group(2) or past_match.group(5)
        if unit in ["year", "years"]:
            start_year = current_year - num
            return [start_year,current_year]
        elif unit in ["decade", "decades"]:
            start_year = current_year - num * 10
            return [start_year,current_year]
        elif unit in ["century", "centuries"]:
            start_year = current_year - num * 100
            return [start_year,current_year]

    # Patterns for future relative dates with ranges
    future_match = re.search(
        r"the (next|coming) (\d+)\s*(year|years|decade|decades|century|centuries)",
        date_str,
    )
    if future_match:
        num = int(future_match.group(2))
        unit = future_match.group(3)
        if unit in ["year", "years"]:
            end_year = current_year + num
            return [current_year,end_year]
        elif unit in ["decade", "decades"]:
            end_year = current_year + num * 10
            return [current_year,end_year]
        elif unit in ["century", "centuries"]:
            end_year = current_year + num * 100
            return [current_year,end_year]

    # Patterns for fixed relative terms with ranges
    if (
        "last year" in date_str
        or "past year" in date_str
        or "previous year" in date_str
    ):
        start_year = current_year - 1
        return [start_year,current_year]
    elif (
        "last decade" in date_str
        or "past decade" in date_str
        or "previous decade" in date_str
    ):
        start_year = current_year - 10
        return [start_year,current_year]
    elif (
        "last century" in date_str
        or "past century" in date_str
        or "previous century" in date_str
    ):
        start_year = current_year - 100
        return [start_year,current_year]

    # Patterns for fixed future terms with ranges
    if "next year" in date_str or "coming year" in date_str:
        end_year = current_year + 1
        return [current_year,end_year]
    elif "next decade" in date_str or "coming decade" in date_str:
        end_year = current_year + 10
        return [current_year,end_year]
    elif "next century" in date_str or "coming century" in date_str:
        end_year = current_year + 100
        return [current_year,end_year]

    is_since_in_doc = any(token.text == "since" for token in doc)
    is_until_in_doc = any(token.text == "until" for token in doc)
    if is_until_in_doc:
        return [current_year,date_str]
    if is_since_in_doc:
        return [date_str,current_year]
        
    date_str = re.findall(r'\d+', date_str)

    return date_str


def extract_entities(prompt):
    country, implementer, time, project = [], [], [], []

    doc = nlp(prompt)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            country.append(ent.text)
        elif ent.label_ == "ORG":
            implementer.append(ent.text)
        elif ent.label_ == "DATE":
            parsed_date = parse_relative_date(ent.text, doc)
            time.append(parsed_date)
        elif ent.label_ == "CARDINAL":
            project.append(word_to_number(ent.text))

    d = {}
    if len(project) > 1:
        d["Projects"] =  tuple(project)
    elif len(project) == 1:
        d["Projects"] = project[0]
    else:
        d["Projects"] = None
    if len(country) > 1:
        d["Country"] =  tuple(country)
    elif len(country) == 1:
        d["Country"] = country[0]
    else:
        d["Country"] = None
    if len(time) > 1:
        d["Time"] =  tuple(time)
    elif len(time) == 1:
        d["Time"] = time[0]
    else:
        d["Time"] = None
    if len(implementer) > 1:
        d["Implementer"] =  tuple(implementer)
    elif len(implementer) == 1:
        d["Implementer"] = implementer[0]
    else:
        d["Implementer"] = None

    return d

