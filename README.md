# Vision-Based ASL Assistive Communication Interface

A webcam tool that lets a patient call for help by **signing letters in American Sign Language**. It tracks the hand with MediaPipe, recognises the letters with a small neural network, and when letters are signed one after another it fills the screen with a large, easy-to-read alert.

| Sign these letters, one after another | The screen shows |
|---|---|
| **S → O → S** | Flashing red/green **EMERGENCY** (stays until a caregiver goes back) |
| **W → A** |  screen: **WATER PLEASE** |
| **C → O** |  screen: **TOO COLD** |
| **H → O** |  screen: **TOO HOT** |
| **N → U** |  screen: **NURSE NEEDED** |
| **H → U** |  screen: **HUNGRY** |


Download and install
```bash
git clone https://github.com/YOUR-USERNAME/asl-patient-communicator.git
cd asl-patient-communicator
```
# INSTRUCTION:
1. Create a virtual environment:

```bash
# Windows (Command Prompt)
py -3.11 -m venv .venv
.venv\Scripts\activate

# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install the packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Check that everything works
```bash
python check_setup.py
```
It tells you if something is missing (Python version, packages, model files, camera).

4. Run it
```bash
python predict_patient.py
```

### Controls
| Key | What it does |
|---|---|
| **B** (or Backspace / Esc / X) | **Back** - leave an alert and return to the camera |
| Mouse click on **< BACK** | Same as above |
| **F** | Toggle fullscreen |
| **Q** | Quit |


### How to sign
1. Show one hand to the camera. A green skeleton with red joints should appear.
2. Make a letter and **hold it still for about 0.7 seconds**. The bar under the letter fills up.
3. Change to the next letter. Letters must follow each other within **5 seconds**.
4. To repeat the same letter, lower your hand for a moment first.


## How it works

1. MediaPipe HandLandmarker finds 21 points on the hand in each camera frame.
2. The points are converted to 63 numbers that describe only the *shape* of the hand (position and size are removed). The same function (`landmarks_to_features` in `asl_common.py`) is used for training and live use.
3. A small neural network turns those numbers into one probability per letter. The last 7 frames are averaged so the answer doesn't flicker.
4. A letter only counts after it was confident (at least 70%) for 0.7 seconds.
5. A sequence matcher checks whether the most recent letters spell one of the signs, and if so shows the alert.

Because the network sees hand *shape* instead of pixels, it doesn't care about skin tone, lighting or background, and training mirrors the data so left and right hands both work.



## Credits and licenses

- Hand tracking: [MediaPipe](https://github.com/google-ai-edge/mediapipe) by Google (Apache 2.0). The hand model file is downloaded from Google on first run.
- Training images: the [ASL dataset by ayuraj](https://www.kaggle.com/datasets/ayuraj/asl-dataset) on Kaggle. It is **not** included in this repository and is downloaded on demand. Check its license on Kaggle before reusing it.
- Neural network: TensorFlow / Keras.
- This project's code is released under the [MIT License](LICENSE).
