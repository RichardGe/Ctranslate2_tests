

print("START ...")


from colorama import Fore, Style
import time
from faster_whisper import WhisperModel


# SELECT A WHISPER VERSION:
#
model_size = "whisper-tiny-ct2"


try:

	print("WhisperModel init...")
	model = WhisperModel(model_size, device="cuda", compute_type="float16"  ,  local_files_only=True )

	print("model.transcribe ...")
	segments, info = model.transcribe("sample1.flac", beam_size=5   ,  without_timestamps=True   )

	print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

	start_time = time.time()

	segments = list(segments)

	# Concatenate all segments into a single string
	final_transcription = "".join([segment.text for segment in segments])
	print(Fore.GREEN + f"Final transcription:  {final_transcription}" + Style.RESET_ALL )

	total_time = time.time() - start_time
	print("Total time taken for model to transcribe: %.2fs" % total_time)

except Exception as e:
	print(f"-----> An error occurred: {e}")
print("ctranslate2.models.Whisper  OK")



