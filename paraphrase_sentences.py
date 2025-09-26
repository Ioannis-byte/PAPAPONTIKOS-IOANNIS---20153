from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def paraphrase(text):
    input_text = "paraphrase: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=5, num_return_sequences=1, early_stopping=True)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

sentence1 = "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."
sentence2 = "During our final discuss, I told him about the new submission — the one we were waiting since last autumn."

print("Original 1:", sentence1)
print("Paraphrased 1:", paraphrase(sentence1))
print("Original 2:", sentence2)
print("Paraphrased 2:", paraphrase(sentence2))
