from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load tokenizer and model
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set target language to Urdu
tokenizer.src_lang = "ur_Arab"

# Example input
urdu_text = "یہ ایک بہت لمبی تحریر ہے جو اردو زبان میں لکھی گئی ہے اور اس کا خلاصہ نکالنا ہے تاکہ صارف آسانی سے سمجھ سکے۔"

# Tokenize and summarize
inputs = tokenizer(urdu_text, return_tensors="pt", max_length=512, truncation=True)
generated_ids = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["ur_Arab"], max_length=128)

# Decode and print summary
summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("🔍 Summarized Text:\n", summary)
