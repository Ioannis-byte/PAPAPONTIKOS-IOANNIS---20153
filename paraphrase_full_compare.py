from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
try:
    from parrot import Parrot
    import torch
    parrot_available = True
except ImportError:
    parrot_available = False
from textblob import TextBlob

# Hugging Face T5 paraphraser (Vamsi/T5_Paraphrase_Paws)
def paraphrase_hf(text):
    model_name = "Vamsi/T5_Paraphrase_Paws"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_text = "paraphrase: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=5, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Parrot Paraphraser - παίρνει την καλύτερη παραφράση
def paraphrase_parrot(text):
    if not parrot_available:
        return "[Parrot Not Installed]"
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())
    responses = parrot.augment(input_phrase=text, adequacy_threshold=0.90, fluency_threshold=0.90)
    if responses:
        return responses[0][0]  # Επιστρέφουμε την πρώτη παραφράση
    else:
        return text

# TextBlob συνώνυμα - βασική παραφράση με εναλλαγή συνωνύμων
def paraphrase_textblob(text):
    blob = TextBlob(text)
    synonyms = []
    for word in blob.words:
        synsets = word.synsets
        if synsets:
            synonyms.append(synsets[0].lemmas()[0].name())
        else:
            synonyms.append(word)
    return ' '.join(synonyms)

texts = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn."
]

print("=== ΑΝΑΚΑΤΑΣΚΕΥΗ ΜΕ 3 ΒΙΒΛΙΟΘΗΚΕΣ ===\n")

for idx, text in enumerate(texts, 1):
    print(f"ΑΡΧΙΚΟ ΚΕΙΜΕΝΟ {idx}:\n{text}\n")

    hf_result = paraphrase_hf(text)
    print(f"Hugging Face T5:\n{hf_result}\n")

    parrot_result = paraphrase_parrot(text)
    print(f"Parrot Paraphraser:\n{parrot_result}\n")

    textblob_result = paraphrase_textblob(text)
    print(f"TextBlob Συνώνυμα:\n{textblob_result}\n")

    print("-" * 80 + "\n")
