from predict import Predictor
def test_predict():

    predictor = Predictor()

    # Test the predictor
    out_path = predictor.predict(prompt="Hello World", voice="Male voice with a low pitch.")
    assert out_path.exists()
    assert out_path.is_file()
    assert out_path.suffix == ".wav"
    assert out_path.stat().st_size > 0