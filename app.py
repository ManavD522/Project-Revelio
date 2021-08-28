import os
import sys
import warnings
import librosa
from tensorflow.keras.models import load_model
from flask import Flask, url_for, request, render_template
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

app = Flask(__name__)

classes = [
    "Bronchiectasis",
    "Bronchiolitis",
    "COPD",
    "Healthy",
    "Pneumonia",
    "URTI",
]


def get_prediction_from_cnn(audio_path):
    if audio_path.split(".")[-1] not in ["wav", "mp3", "m4a"]:
        return "Wrong File Uploaded"

    clip, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=40)
    if mfccs.shape[1] == 862:
        model = load_model(
            "static/tfjs-models/Respiratory-disease-detection-model/respiratory-model.h5")
        mfccs = mfccs.reshape(1, 40, 862, 1)
        predictions = model.predict(mfccs)
        predictions = [("{:.2f}".format(pred * 100))
                       for pred in predictions[0]]
        return predictions
    else:
        return "Cannot process audio file"


@ app.route('/', methods=['GET', 'POST'])
def resp():
    if request.method == "GET":
        return render_template('Respiratory.html', got_prediction=False)
    if request.method == "POST":
        f = request.files.get('uploaded-audio', None)

        if f is None:
            return render_template('Respiratory.html', got_prediction=False)

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploaded-data', secure_filename(f.filename))
        f.save(file_path)
        prediction = get_prediction_from_cnn(file_path)

        if(type(prediction) == str):
            return render_template('Respiratory.html', got_prediction=False)
        print(prediction)
        return render_template('Respiratory.html', prediction=prediction, classes=classes, got_prediction=True)


if __name__ == "__main__":
    app.run(debug=True)
