# Evaluation

BLEU-based evaluation and greedy caption generation on the test split.

## CLI

```bash
python main.py evaluate --datasets flickr8k -y
make evaluate DATASETS=flickr8k,flickr30k YES=1
```

Requires a trained bundle: `model_<prefix>.keras`, `tokenize_<prefix>.dump`, `maxlen_<prefix>.dump`, and matching feature caches.

`evaluate` sets `config.train=False`, `config.evaluate=True`, still loads datasets and features.

## Module: `captioning/inference.py`

### `generate_caption(model, tokenizer, photo, max_length)`

**Greedy decoding** (inference differs from teacher forcing). See also [Greedy caption generation at inference](ai-concepts.md#greedy-caption-generation-at-inference) for how this differs from a simplified “LSTM + previous word” mental model.

1. **Image features:** `photo` is the precomputed VGG16 vector (same vector at every step).
2. **Start prefix:** `"<start>"`.
3. **Loop** (at most `max_length` times):
   - Tokenize the **full prefix so far** (not only the last word).
   - Pad the token ids to `max_length`.
   - `model.predict([photo, sequence])` — image branch + text branch (`Embedding` → LSTM) are fused inside the model via `Add` + `Dense` + softmax.
   - Argmax → next token id → append word to prefix.
4. **Stop** when the predicted token is `<end>`, or when the loop reaches `max_length` (cap, not a target length).

Used in pipeline sample output and Streamlit (via wrapper in `app.py`).

### `evaluate_model(model, descriptions, images, tokenizer, max_length)`

For each test image with features:

1. Generate caption with `generate_caption`.
2. Collect references (tokenized ground-truth captions) and prediction.
3. Print **corpus BLEU** scores (NLTK `corpus_bleu`):

| Metric | Weights |
|--------|---------|
| BLEU-1 | (1, 0, 0, 0) |
| BLEU-2 | (0.5, 0.5, 0, 0) |
| BLEU-3 | (0.3, 0.3, 0.3, 0) |
| BLEU-4 | (0.25, 0.25, 0.25, 0.25) |

## BLEU: what it measures

BLEU compares n-gram overlap between generated and reference captions. Higher is better, but:

- **Pros**: Fast, standard in captioning papers, easy to compare runs.
- **Cons**: Penalizes valid paraphrases; multiple references per image help but do not fix semantic blindness.

Modern systems often also report **CIDEr**, **SPICE**, or human judgments; this course focuses on BLEU for simplicity.

## Pipeline hook

In `run_pipeline`, after training or loading the model:

```python
if config.evaluate:
    evaluate_model(model, test_description, test_features, tokenizer, maxlen)
```

A sample caption on one test image is always printed when the test split is non-empty.

## Timing

Evaluation time is recorded in `training_stats_<prefix>.json` under phase `"evaluation"`.
