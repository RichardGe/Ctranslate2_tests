
# ANSI color codes
original_color = "\033[94m"  # Blue
translated_color = "\033[92m"  # Green
reset_color = "\033[0m"  # Reset to default color


import ctranslate2
from transformers import AutoTokenizer

def main():
	# Paths to your models
	tokenizer_model_dir = "opus-mt-en-de"	   # Path to the original model (for the tokenizer)
	ctranslate2_model_dir = "opus-mt-en-de-ct2" # Path to the converted CTranslate2 model

	# Load the tokenizer
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_dir, local_files_only=True)

	# Create a translator using the CTranslate2 model on the CUDA device
	translator = ctranslate2.Translator(ctranslate2_model_dir, device='cuda')

	# Text to translate
	text = """Technology has advanced rapidly in the last few decades, transforming the way we live, 
	work, and communicate. The internet, mobile devices, and artificial intelligence have all 
	played significant roles in shaping our modern world. As we move forward, the ethical implications 
	of these advancements become increasingly important. How can we ensure that technology benefits everyone equally, 
	without causing harm or reinforcing existing inequalities?"""

	# Tokenize the input text (keep the unrecognized parameter)
	input_ids = tokenizer.encode(text, return_special_tokens=False)
	tokens = tokenizer.convert_ids_to_tokens(input_ids)

	# Translate the tokenized text
	results = translator.translate_batch([tokens])

	# Get the translated tokens
	translated_tokens = results[0].hypotheses[0]

	# Convert tokens to IDs
	translated_ids = tokenizer.convert_tokens_to_ids(translated_tokens)

	# Convert IDs to text
	translated_text = tokenizer.decode(translated_ids, skip_special_tokens=True)

	print("\n\n")
	print(f"{original_color}Original text: {text} {reset_color}")
	print("\n\n")
	print(f"{translated_color}Translated text: {translated_text} {reset_color}")
	print("\n\n")

if __name__ == "__main__":
	main()


