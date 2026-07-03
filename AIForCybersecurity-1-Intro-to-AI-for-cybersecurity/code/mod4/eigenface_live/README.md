# Eigenface Live

macOS-only desktop app for **live face enrollment and verification** using the same PCA + MLP idea as [`FacialRecognitionEigenfaces.py`](../FacialRecognitionEigenfaces.py), but with your Mac camera instead of LFW.

## Features

- **Train tab:** capture ~40 face frames while you move your head in slow oval shapes
- **Verify tab:** check whether the live face matches a saved enrollment
- **Pipeline:** grayscale crop → PCA eigenfaces → MLP classifier + distance threshold
- **Security:** local-only storage, HMAC-signed enrollments, restrictive file permissions, sanitized identity names

## Requirements

- macOS with a built-in or external camera
- Python 3.11+ recommended
- Camera permission for your terminal or IDE when prompted

## Quick start

```bash
cd code/mod4/eigenface_live
make install
make run
```

Or without Make:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Usage

### Train

1. Open the **Train** tab.
2. Enter an identity label (letters, numbers, spaces, `-`, `_`).
3. Click **Start enrollment**.
4. Move your face slowly in oval shapes until the progress bar completes.
5. The model is saved under `data/enrollments/`.

### Verify

1. Open the **Verify** tab.
2. Choose an enrolled identity and click **Start verification**.
3. Look at the camera. The app reports `MATCH`, `NO_MATCH`, or `UNCERTAIN`.

## Architecture

```
eigenface_live/
├── services/     camera, face detection, oval capture, ML, persistence
├── ui/           Qt tabs, controller facade, camera widget
├── models/       domain types and enrollment artifact
└── utils/        image conversion and safe path helpers
```

Design choices:

- **Facade:** `AppController` coordinates services for the UI
- **Repository:** `ModelRepository` handles signed local persistence
- **Strategy:** oval-motion heuristic isolated in `OvalMotionCollector`
- **Dependency injection:** tabs receive the shared controller and camera widget

## Security notes

- No network calls; models never leave your machine
- Enrollment files are chmod `600`; directory chmod `700`
- Identity names are validated before use in filenames
- HMAC metadata signing detects tampered enrollment files
- This is a **course demo**, not production biometric auth: no liveness detection, anti-spoofing, or encryption at rest beyond OS file permissions

## Tests

```bash
make test
```

## Troubleshooting

| Issue | Fix |
| ----- | --- |
| Camera will not open | System Settings → Privacy & Security → Camera → allow Terminal or your IDE |
| No face detected | Improve lighting; center your face; move closer |
| `UNCERTAIN` during verify | Re-enroll with more oval motion; hold still briefly after moving |
| Import errors | Run `make install` inside `eigenface_live/` |

## Related course material

- [`docs/module-04-securing-user-authentication.md`](../../../docs/module-04-securing-user-authentication.md) — eigenfaces section
- [`FacialRecognitionEigenfaces.py`](../FacialRecognitionEigenfaces.py) — static LFW lab
