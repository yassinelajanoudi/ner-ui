from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer
from predict import predict_text
from ner import extract_entities
import warnings
warnings.filterwarnings("ignore", message="Some weights of RobertaForSequenceClassification were not initialized")

models_and_tokenizers = {
    "Projects": (
        AutoPeftModelForSequenceClassification.from_pretrained("./models/project/roberta-base-peft"),
        AutoTokenizer.from_pretrained("./models/project/tokenizer_save_path"),
    ),
    "Country": (
        AutoPeftModelForSequenceClassification.from_pretrained("./models/country/roberta-base-peft"),
        AutoTokenizer.from_pretrained("./models/country/tokenizer_save_path"),
    ),
    "Time": (
        AutoPeftModelForSequenceClassification.from_pretrained("./models/time/roberta-base-peft"),
        AutoTokenizer.from_pretrained("./models/time/tokenizer_save_path"),
    ),
    "Implementer": (
        AutoPeftModelForSequenceClassification.from_pretrained("./models/implementer/roberta-base-peft"),
        AutoTokenizer.from_pretrained("./models/implementer/tokenizer_save_path"),
    ),
}

def process_prompt(prompt):
    categories = {"Projects": None, "Country": None, "Time": None, "Implementer": None}
    last_item = {"Projects": 0, "Country": 0, "Time": 0, "Implementer": 0}

    project_model, project_tokenizer = models_and_tokenizers["Projects"]
    project_predicted_class = predict_text(prompt, project_model, project_tokenizer)
    if project_predicted_class == 0:
        return {'Projects': None, 'models': [0, 0, 0, 0]}
    res = extract_entities(prompt)
    categories["Projects"] = res["Projects"]
    last_item["Projects"] = 1
    for category in categories.keys():
        if category == "Projects":
            continue
        model, tokenizer = models_and_tokenizers[category]
        predicted_class = predict_text(prompt, model, tokenizer)
        # categories[category] = 1 if predicted_class == 1 else 0
        if predicted_class == 1:
            categories[category] = res[category]
            last_item[category] = 1
    
    models_values = list(last_item.values())
    res = {"Projects":categories["Projects"]}
    i = 1
    for category in categories.keys():
        if category == "Projects":
            continue
        if category and models_values[i]:
            res[category] = categories[category]
            i += 1
    res["models"] = models_values
    return res


text = "what projects have worked on youth livelihoods in west africa?"
res = process_prompt(text)
print(res)
