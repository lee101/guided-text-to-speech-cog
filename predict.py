import tempfile

from cog import BasePredictor, Input, Path
from PIL import Image, ImageFilter

from parlerlib import gen_tts, write_wav


class Predictor(BasePredictor):
    def predict(
        self,
        prompt: str = Input(description="Voice"),
        voice: str = Input(description="voice description"),
    ) -> Path:
        sample_rate, audio_arr = gen_tts(prompt, voice)

        wav = write_wav(audio_arr, sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/dev/shm") as tmp_file:
            tmp_file.write(wav)
            tmp_file_path = tmp_file.name

        return Path(tmp_file_path)
        # return Path("audio.wav")
    

        # out_path = Path(tempfile.mkdtemp()) / "out.png"
        # im.save(str(out_path))
        # return out_path