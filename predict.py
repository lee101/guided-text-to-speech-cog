import tempfile

from cog import BasePredictor, Input, Path

from parlerlib import gen_tts, write_mp3


class Predictor(BasePredictor):
    def predict(
            self,
            prompt: str = Input(description="Voice"),
            voice: str = Input(description="voice description"),
    ) -> Path:
        sample_rate, audio_arr = gen_tts(prompt, voice)

        wav = write_mp3(audio_arr, sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir="/dev/shm") as tmp_file:
            tmp_file.write(wav)
            tmp_file_path = tmp_file.name

        return Path(tmp_file_path)
