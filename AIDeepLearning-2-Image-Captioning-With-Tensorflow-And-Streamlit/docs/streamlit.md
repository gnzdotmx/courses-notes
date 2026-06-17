# Streamlit app

Interactive demo: upload an image, get a generated caption.

## Why Streamlit?
Training and BLEU evaluation run from the CLI and report **numbers on a fixed test split**. That proves the pipeline works, but it does not let you **try the model on a new photo** and judge the caption yourself. The Streamlit app closes that loop: it is the **human-facing deployment step** of the project.

| Without a UI | What you miss |
|--------------|---------------|
| CLI `train` / `evaluate` only | Scores on reference captions, not your own images |
| Saved `.keras` artifacts alone | No easy way to upload a JPG and read the output |
| Notebook-only inference | Harder to share with others; reload models every session |

**Why Streamlit specifically?**

1. **Course goal**: The Coursera project is *Image Captioning with TensorFlow and Streamlit*; deploying with Streamlit is part of the learning outcome, not an afterthought.
2. **Python-only UI**: No separate frontend stack. `app.py` reuses the same inference path as evaluation (`image_to_features` → `generate_caption` in `captioning/inference.py`).
3. **Fast feedback**: Upload an image, see the caption in seconds. That complements BLEU: you can spot fluent-but-wrong or paraphrased-but-good captions that n-gram scores miss (see [BLEU limitations](ai-concepts.md#limitation-lexical-overlap--meaning)).
4. **Compare trained runs**: The sidebar lists every `model_<prefix>.keras` bundle (e.g. `flickr8k` vs `flickr30k`) so you can switch models without retraining.
5. **Sensible defaults for a demo**: `@st.cache_resource` loads VGG16 and caption models once per session; file upload and image preview are built in.

**What it is not:** Streamlit here is a **single-user educational demo**, not a production API. It is not meant for high concurrency, authentication, or persistent storage of uploads. For those needs you would wrap the same inference code in a proper service (FastAPI, batch jobs, etc.).

## Validating the app
This repo has **no automated Streamlit test suite**. “Validating” the app means **manual smoke testing** after training: confirming the full path from saved artifacts to a caption on a **new** upload works. That complements CLI `evaluate` (BLEU on the fixed test split); it does not replace it.

| Check | What you are proving | Where it happens |
|-------|----------------------|------------------|
| **App starts** | Streamlit launches with `cwd=app/` so relative artifact paths resolve | `run_streamlit_app` in `captioning/cli.py` |
| **Trained models exist** | At least one `model_<prefix>.keras` bundle is discoverable | `list_model_prefixes()`; error in `app.py` if empty |
| **Artifact bundle loads** | `model_<prefix>.keras`, `tokenize_<prefix>.dump`, `maxlen_<prefix>.dump` are present and load together | `load_caption_bundle()` |
| **VGG16 extractor loads** | Shared `vgg16_model.h5` is available for live images | `load_vgg()` |
| **Upload → preprocess** | JPG/PNG opens as RGB, resizes to 224×224, `preprocess_input` | `load_image`, `image_to_features` |
| **Inference path** | Same greedy decode as evaluation: VGG16 features → `generate_caption` | `captioning/inference.py` via `app.py` wrapper |
| **Readable output** | `<start>` / `<end>` stripped before display | `generate_caption` in `app.py` |
| **Model switching** | Sidebar prefix change loads a different trained bundle | `st.sidebar.selectbox` + cached `load_caption_bundle` |
| **Human judgment** | Caption is fluent and roughly matches the image: paraphrases, nonsense, or missed objects that BLEU may not catch | You, on arbitrary photos |

**What validation does *not* test:** corpus BLEU scores, train/test split integrity, or training correctness. Those stay on the CLI (`train`, `evaluate`). Streamlit only proves **deployment**: artifacts + live preprocessing + the same inference code path you already used offline.

**Quick validation checklist:** train once → `make streamlit` → pick a model in the sidebar → upload a test image → confirm preview and a non-empty caption → switch prefix if you have multiple runs → try an image outside the training set.

## Why deploy to AWS EC2 instead of only running locally?
`make streamlit` on your laptop is the right way to **develop and debug** the app. The course still asks you to deploy on **AWS EC2** because local and cloud solve different problems.

| Local only (`localhost`) | EC2 deployment |
|--------------------------|----------------|
| Only you (same machine) can open the UI | A public URL anyone can visit |
| Stops when you close the laptop or terminal | Runs on a remote VM while the instance is up |
| Default bind address is `localhost` | Forces `--address 0.0.0.0`, `--headless`, security groups, SSH |
| Your OS, Python, and free RAM vary | Fixed Linux environment you choose (instance type, disk) |

**Why EC2 specifically?**

1. **Course goal**: *Image Captioning with TensorFlow and Streamlit* includes **cloud deployment** as a learning outcome: provision a VM, copy artifacts, install dependencies, and expose the Streamlit port safely.
2. **Share the demo**: Instructors, teammates, or portfolio visitors need a **link**, not SSH access to your Mac. EC2 gives the app a stable host on the internet.
3. **Separate training from serving**: You may train on a GPU machine or locally, then ship `model_*.keras`, tokenizer dumps, and `vgg16_model.h5` to a smaller CPU instance for inference-only demo traffic.
4. **Realistic ML workflow**: Notebooks and `streamlit run` on localhost are for iteration; putting a model behind a running service on a server is how many teams **close the loop** from experiment to something others can try.
5. **Ops skills that transfer**: Security groups (open port 8501), SSH, `systemd` or `nohup` to keep the process alive, and monitoring disk/RAM for TensorFlow + VGG16 mirror tasks you will see beyond this course.

**What local is still for:** fast iteration, breakpoint debugging, and verifying artifacts before you `scp` them to EC2. Deploy when the app works locally and you want it **reachable without your machine**.

## How to run

```bash
# Via CLI (recommended; sets cwd to app/)
python main.py streamlit
python main.py streamlit --port 8502 --headless

# Direct
streamlit run app.py

# Makefile (repo root)
make streamlit
make streamlit-app
```

## Prerequisites

Under `app/`:

- `vgg16_model.h5`: from feature extraction / training
- At least one trained bundle: `model_<prefix>.keras`, `tokenize_<prefix>.dump`, `maxlen_<prefix>.dump`

Train first, e.g. `make train EPOCHS=10`.

## Module: `app/app.py`

### Model discovery

`list_model_prefixes()` (`captioning/paths.py`) scans `model_*.keras` and the registry. Legacy single-file artifacts appear as **default (legacy artifacts)**.

Sidebar **Trained model** selectbox switches between prefixes (e.g. `flickr8k` vs `flickr30k`).

### Cached resources

`@st.cache_resource`:

- `load_vgg()`: VGG16 feature model
- `load_caption_bundle(prefix)`: caption model + tokenizer + maxlen

Avoids reloading large files on every interaction.

### Inference path

1. User uploads JPG/PNG.
2. `load_image` → RGB PIL image.
3. `image_to_features`: resize 224×224, `preprocess_input`, VGG16 predict (same as training pipeline).
4. `generate_caption`: calls `captioning.inference.generate_caption`, strips `<start>` / `<end>` for display.

### UI structure

- **Image**: uploader, preview, caption with spinner.
- **About**: short explanation of prefixes and feature caches.

## CLI launcher (`captioning/cli.py`)

`run_streamlit_app` runs `python -m streamlit run app.py` with `cwd=APP_DIR` so relative artifact paths resolve correctly.

Flags: `--port`, `--address`, `--headless`. Model selection is in the Streamlit sidebar.

## Deployment notes

See [Why deploy to AWS EC2?](#why-deploy-to-aws-ec2-instead-of-only-running-locally) for when cloud beats local-only.

On EC2 (or any remote Linux host):

- Bind `--address 0.0.0.0` and use `--headless` (e.g. `python main.py streamlit --address 0.0.0.0 --headless`).
- Open inbound **TCP 8501** (or your `--port`) in the instance security group.
- Copy `app/` artifacts (`vgg16_model.h5`, `model_*.keras`, tokenizer/maxlen dumps) to the server; run from `app/` so relative paths resolve.
- Streamlit is single-user oriented; not a production API replacement.
- Uploaded images stay in memory for the session; no server-side persistence by default.
