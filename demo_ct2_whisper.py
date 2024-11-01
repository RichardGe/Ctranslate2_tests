

from colorama import Fore, Style
import ctranslate2
import librosa
import transformers
import numpy as np
from transformers import WhisperProcessor
import torch
import time
from pathlib import Path

# SELECT A WHISPER VERSION:
#
version_whisper = "tiny"
# version_whisper = "large-v2"
# version_whisper = "large-v3"


try:


	print("START with " + version_whisper + ".")

	# Set beam_size and compute_type parameters
	beam_size = 5  # Adjust as needed
	compute_type = "float16"  # Options: "int8", "int8_float16", "float16", "float32"

	# Load and resample the audio file.
	print("Loading audio with librosa...")


	# change the audio file here. from my tests, it works at least with .flac, .mp3
	audio_path = "sample1.flac"

	audio, _ = librosa.load(audio_path, sr=16000, mono=True)

	# Initialize the processor
	print("Initializing WhisperProcessor...")
	processor = WhisperProcessor.from_pretrained("whisper-" + version_whisper)

	# Detect the language once using the first 30 seconds
	print("Detecting language...")
	lang_detect_duration = 30 * 16000  # 30 seconds in samples
	audio_lang_detect = audio[:lang_detect_duration]
	inputs_lang = processor(audio_lang_detect, return_tensors="np", sampling_rate=16000)
	features_lang = ctranslate2.StorageView.from_array(inputs_lang.input_features)

	# Load the model with compute_type
	#  device=  "cpu"  or  "cuda"(for NVIDIA,CUDA or AMD,HIP GPU)
	print("Loading Whisper model with compute_type:", compute_type)
	model = ctranslate2.models.Whisper( "whisper-" + version_whisper + "-ct2",    device="cuda",   compute_type=compute_type )

	# Detect the language
	results = model.detect_language(features_lang)
	language, probability = results[0][0]
	print("Detected language %s with probability %f" % (language, probability))

	# Prepare the prompt
	prompt = processor.tokenizer.convert_tokens_to_ids(
		[
			"<|startoftranscript|>",
			language,
			"<|transcribe|>",
			"<|notimestamps|>",  # Remove this token to generate timestamps.
		]
	)


	print(f"Prompt token IDs: {prompt}")


	# Process the audio in 30-second chunks
	chunk_size_sec = 30
	chunk_size = chunk_size_sec * 16000  # Samples per chunk
	num_chunks = int(np.ceil(len(audio) / chunk_size))

	transcriptions = []

	start_time = time.time()

	for i in range(num_chunks):
		start = i * chunk_size
		end = min((i + 1) * chunk_size, len(audio))
		audio_chunk = audio[start:end]
		print(f"Processing chunk {i+1}/{num_chunks}, samples {start}:{end}")

		inputs = processor(audio_chunk, return_tensors="np", sampling_rate=16000)
		features = ctranslate2.StorageView.from_array(inputs.input_features)

		# Generate transcription for the chunk with beam_size
		print(Fore.GREEN +  "Generating transcription..."                         + Style.RESET_ALL )
		print(Fore.GREEN +  "      features.shape() = "   + str(features.shape)   + Style.RESET_ALL )
		print(Fore.GREEN +  "      len(prompt)      = "   + str(len(prompt))       + Style.RESET_ALL )
		results = model.generate(
			features,
			[prompt],
			beam_size=beam_size,
			max_length=448,
			repetition_penalty=1.0,
			return_scores=False,
			return_no_speech_prob=False,
		)

		transcription = processor.decode(results[0].sequences_ids[0])
		print( "Transcription:"  , Fore.BLUE + transcription.strip() + Style.RESET_ALL )
		transcriptions.append(transcription.strip())



	total_time = time.time() - start_time
	print("Total time taken for model to transcribe: %.2fs" % total_time)


	# Combine the transcriptions
	full_transcription = " ".join(transcriptions)


	print("END")


except Exception as e:
	print(f"-----> An error occurred: {e}")
print("ctranslate2.models.Whisper  OK")


