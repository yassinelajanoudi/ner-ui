
def predict_text(text, inference_model, tokenizer):
    """
    Classifies the input text using the loaded PEFT model.

    Parameters:
    - text (str): The text to be classified.

    Returns:
    - str: The predicted class label for the input text.
    """
    # Tokenize the input text and prepare it for the model
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

    # Perform inference using the model
    output = inference_model(**inputs)

    # Get the predicted class by finding the index with the highest logit value
    prediction = output.logits.argmax(dim=-1).item()

    # Return the class label based on the prediction (assuming `id2label` mapping exists)
    # return id2label[prediction]
    return prediction


