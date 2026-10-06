# Module 6: Evaluating AI Models and Performance

## Challenges of AI on the alert path

### Objectives

- Three placements: no AI, AI as a back-end validator, AI inside the sensor
- FP and FN change the analyst’s job. They do not remove it
- The score can stay a black box. Artifacts are what you check
- Fit, scale, and CIA on the host. Data quality sets model quality

### Key concepts

| Term | Definition / notes |
| ---- | ------------------ |
| SIEM | Security information and event manager. Tier-1 queue where sensor alerts land |
| Signature sensor | Matches known threats. Misses anything with no signature |
| Statistical / anomaly sensor | Flags unusual behavior. Always carries false positives and false negatives |
| Triage | First pass that decides whether an alert is worth a full investigation |
| Investigation pathway | A suggested next step attached to the alert. Absent when the analyst has only raw metadata |
| Black box | The model’s own decision is opaque. Supporting artifacts can still be inspected |
| Artifact | Concrete check: traffic view, notification, threat-intel hit, policy violation, IOC |
| Explainable autonomic cybersecurity | Lower FP/FN until an automatic response is justified, plus clearer model decisions |
| Model engineering | Pick an algorithm for the classification problem, then tune it for this domain and this data |
| Data quality | Prerequisite of model quality. Good data beats a fancier model on bad data |

Study cue: **the analyst still validates the model’s call,** until error rates and explanations are good enough for an active response.

## The queue without AI

A tier-1 analyst receives alerts in the SIEM. Typical non-AI sources:

| Sensor | Layer | Style |
| ------ | ----- | ----- |
| Snort | Network IDS | Signature |
| Symantec Endpoint Protection | Host IDS | Signature / host policy |
| Bit9 (now Carbon Black) | Application allow / deny | Whitelist or blacklist |

Assume these sensors are statistical or signature-based.

| Sensor type | What it catches | What the analyst must assume |
| ----------- | --------------- | ---------------------------- |
| Signature | Known threats | Unknown threats produce **no alert** |
| Statistical | Departures from a baseline | Every alert can be a false positive or a false negative, so each one is vetted |

Study cue: **known = signature; weird = statistical; both still need a human.**

### Four jobs that stay on the analyst

1. **No suggested path.** Investigation starts from alert metadata alone.
2. **Personal regimen.** Nearly every alert is validated by a checklist the analyst owns.
3. **Unknowns are a blind spot.** With no signal, there is also no suggested way to go look for them.
4. **After the alert.** A long-term plan to keep the threat out, plus an internal and external communication plan.

## Where AI can sit

### Validator only

AI runs the first validation pass, on a regimen it chooses. The analyst still checks that pass. The hope is that the check is shorter than the original regimen.

Example: a Snort alert. The analyst drops the PCAP (or the flagged flow) into the tool. The tool inspects the traffic and any files inside it, then returns a decision. That pass can include anomaly detection on the flow and more than one kind of malware analysis on an attachment.

```text
Snort alert
  → analyst submits PCAP
    → AI: anomaly score on the flow + malware passes on embedded files
      → decision + supporting output
        → analyst validates that triage
```

### AI in the sensors

The same work moves to the front. When the alert reaches the SIEM, the metadata from those steps is already on it.

| Without AI | AI in the sensor |
| ---------- | ---------------- |
| Metadata only | A suggested investigation pathway |
| Analyst regimen on nearly every alert | Own regimen is optional |
| Unknown threats invisible | Some unknown-threat analysis already done, with a next step |
| Action plans and reports written by hand | AI feedback can draft action plans and internal or external reports |

Study cue: **front-loaded AI buys a path, a first look at unknowns, and a draft report.** It does not remove the need to check the result.

## Anomaly detectors and the new job

A basic AI network sensor is an anomaly detector. Same limitation as any statistical tool: false positives and false negatives. That limit does not make every decision unusable. It means the analyst is not replaced. The job changes: **validate what the model found**, and use the time saved to do more of that validation well.

A score $s(x)$ and a threshold $\tau$ are enough to see the tradeoff. $y = 1$ is malicious, $y = 0$ is benign.

$$
\mathrm{alert}(x) = \mathbf{1}\big[s(x) > \tau\big]
$$

$$
\mathrm{FPR}(\tau) = P\big(s(x) > \tau \mid y = 0\big),
\qquad
\mathrm{FNR}(\tau) = P\big(s(x) \le \tau \mid y = 1\big)
$$

| Move $\tau$ | Queue | Misses |
| ----------- | ----- | ------ |
| Lower | More false positives to vet | Fewer false negatives |
| Higher | Quieter queue | More malicious traffic scored as normal |

Precision is the share of alerts that were real. Low precision is a SIEM full of noise:

$$
\mathrm{Precision} = \frac{TP}{TP + FP},
\qquad
\mathrm{Recall} = \frac{TP}{TP + FN} = 1 - \mathrm{FNR}
$$

```python
# Illustration: one feature, z-score anomaly rule.
# tau is the only knob. It moves FPR and FNR in opposite directions.
def alerts(x, mu, sigma, tau):
    z = abs((x - mu) / sigma)
    return z > tau

# y = 1 malicious. Count mistakes at two operating points.
def confusion(y, flagged):
    tp = int(((y == 1) & flagged).sum())
    fp = int(((y == 0) & flagged).sum())
    fn = int(((y == 1) & ~flagged).sum())
    tn = int(((y == 0) & ~flagged).sum())
    return tp, fp, fn, tn
```

Study cue: **FP and FN are a threshold choice, not a reason to ignore the tool.**

## Explainability: score vs artifacts

Industry anomaly tools do two different things:

1. The model flags anomalous behavior. That decision can be a **black box**.
2. Supporting pieces then build **artifacts**: statistical context, indicators of compromise, and captured expert knowledge. Those artifacts are the check on the model’s call.

Concrete items the analyst can open:

- Network-traffic visualizations
- The alert notification itself
- Threat-intelligence matches
- Policy violations

Trust is earned. Early on, those artifacts get a full look. After the tool has been right often enough, less of that validation is required. Explainability, false positives, and false negatives remain the standing limits.

**Explainable autonomic cybersecurity** is the target on the other side of those limits: FP and FN low enough that an active response is reasonable, and model decisions transparent enough to justify that response.

AI also shows up off the network: system-level detectors, back-end alert validators, and other helper roles. The challenges below apply to the model in general, wherever it sits.

```python
# The score can stay opaque. The analyst signs off on artifacts.
def accept(score, artifacts, analyst_confident):
    if analyst_confident:
        return score > 0  # lighter check once the tool has a track record
    return score > 0 and artifacts.has_ioc and artifacts.policy_violation
```

## Model engineering

Recall the analytic-development step: choose an algorithm that fits the classification problem, then optimize it. This step is **more art than science**. There is no regimen that is perfect on every dataset.

What “expert” means here:

| Knowledge | Why it matters |
| --------- | -------------- |
| The algorithm | What it can represent, and how it fails |
| The domain | Which alerts, protocols, and host events actually matter |
| Data behavior | How that algorithm reads this kind of data |

Two deployment constraints sit next to accuracy:

- **Performance and scale** match the application (line-rate sensing vs. an offline PCAP review).
- The **host** of the model is defended on the CIA triad.

| Property | Attack on the model host |
| -------- | ------------------------ |
| Confidentiality | Training data, features, or weights are exposed |
| Integrity | Inputs, labels, or the model are altered so detections lie |
| Availability | The detector is overloaded or taken down so alerts stop |

Study cue: **fit the algorithm to the problem, tune with domain judgment, size it for the pipe, lock down the box it runs on.**

## Data quality

Model quality and data quality come as a pair. Rule of thumb: **good data beats a very good model on poor data.**

Failures to keep out of the training set:

| Failure | What it does to detection | Mitigation already in this course |
| ------- | ------------------------- | --------------------------------- |
| Bias | Systematic skew in who or what gets flagged | Prefer representative collection over a cleverer model |
| Mislabeled rows | The model learns the wrong class boundary | Fix labels before tuning |
| Unbalanced classes | Accuracy looks strong while attacks are missed | [Module 2](module-02-fraud-prevention-with-cloud-ai-solutions.md): class weights, random undersampling, synthetic oversampling |
| Missing values | Silent holes in the features the sensor relies on | Impute or drop with a rule you can defend; do not train through the holes |

Accuracy is the trap on a rare positive class. If almost every flow is benign, “always benign” scores high and catches nothing:

$$
\mathrm{Accuracy} = \frac{TP + TN}{N}
$$

A worked base rate: $1$ attack in $500$ flows, and a model that never alerts.

$$
\mathrm{Accuracy} = \frac{499}{500} = 0.998, \qquad \mathrm{Recall} = 0
$$

That is why module 2 scored fraud models with $F_1$ and ROC, not accuracy alone. $F_1$ punishes a model that buys precision by never firing, or recall by firing on everything:

$$
F_1 = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
$$

```python
def accuracy_trap(n=500, attacks=1):
    # Always predict benign. TN = n - attacks, TP = 0.
    accuracy = (n - attacks) / n   # 0.998
    recall = 0.0                    # every attack is a false negative
    return accuracy, recall
```

## Evaluating the analytic development process

### Objectives

- Overfitting: strong on the training rows, weak on new ones
- Large set: one $70$–$80\%$ split. Small set: $k$-fold, then average
- More reliable data usually beats a fancier algorithm
- Features, scale, and encodings before the model. Classical ML needs all three pieces
- Generalization error is bias, variance, and noise. Compare models with ROC-AUC

### Key concepts

| Term | Definition / notes |
| ---- | ------------------ |
| Overfitting | Train error looks good. Unseen data fails. Too little data, or too many useless features |
| Train / test split | One cut of a large set. Train on most of it. Score only the held-out chunk |
| $k$-fold CV | $k$ cuts. Each cut tests a different slice. Average the $k$ scores |
| Feature | A column as-is, or a transform that pulls a pattern out. The scientist picks which |
| Binarization | Raw value $\to$ $0$ or $1$. Natural fit for images |
| Binning | Map a number into a range. Shrinks the scale |
| Log transform | Shrinks the scale and keeps small values visible next to huge ones |
| Feature normalization | Rescale columns so a large-range feature cannot overshadow a small one |
| Generalization error | Error on unseen data. Bias $+$ variance $+$ noise |
| Bias | Error from wrong assumptions in the algorithm’s parameters |
| Variance | Error from reacting too hard to wiggles in the training sample |
| Noise | Irreducible mess in the labels or measurements. Some of it always stays |
| Optimized model | Right complexity, right assumptions, right sensitivity, right training-set size, solid test score |
| ROC | Curve of true positive rate vs false positive rate as the threshold moves |
| AUC | Area under that curve. One number for the whole threshold sweep |

No single piece (data, features, algorithm, score) is useful alone.

```text
enough reliable data
  → features (deep learning can learn these itself)
    → train / test, or k-fold
      → complexity in the middle (bias and variance)
        → score on unseen data (ROC-AUC)
```

## Overfitting

Training on the data is easy to overdo. The fit can look excellent and still collapse the first time the inputs are new. A model that only looks good on the rows it memorized has not been evaluated.

Two causes:

| Cause | What went wrong |
| ----- | --------------- |
| Too little data | The fit memorizes the few rows it saw |
| Too many unnecessary features | Extra columns give the model room to chase noise |

## A large set: one split

When the set is large enough, cut it once.

| Chunk | Share | Job |
| ----- | ----- | --- |
| Train | about $70\%$–$80\%$ | Fit the model |
| Test | about $20\%$–$30\%$ | Estimate error on rows the fit never saw |

$$
D = D_{\mathrm{train}} \cup D_{\mathrm{test}},
\qquad
\frac{|D_{\mathrm{train}}|}{|D|} \in [0.7,\, 0.8]
$$

The test chunk stays out of every fitting step. That held-out score is the stand-in for future data.

```python
def split_counts(n, train_frac=0.75):
    n_train = round(n * train_frac)   # band: 0.70 to 0.80
    n_test = n - n_train               # the rest: 0.20 to 0.30
    return n_train, n_test
```

## A small set: $k$-fold

A single $20\%$ test slice can be too small, or too lucky. $k$-fold keeps almost every row in training and still tests every row once.

For $k = 10$: train on $90\%$, test on $10\%$, repeat on a different $10\%$ each time, then average.

$$
\widehat{M} = \frac{1}{k} \sum_{i=1}^{k} M(D \setminus F_i,\; F_i)
$$

$F_i$ is fold $i$. $M$ is whatever score you care about (accuracy, $F_1$, AUC). $D \setminus F_i$ is the train set for that round.

```python
def kfold_mean(scores):
    # one score per fold; average the k passes
    return sum(scores) / len(scores)

# k = 10 → each test fold is n/10, train fold is the other 90%
def fold_sizes(n, k=10):
    test = n // k
    return n - test, test
```

Study cue: **large set → one split. Small set → $k$ splits, then the mean.**

## More data, or a better algorithm

When those two compete, **in most cases, get more reliable data.**

## Pieces of the process

The analytic development process is the map for the whole course. It also says how the parts depend on each other. Data with no algorithm and no features does nothing. An algorithm with no data does nothing.

Deep learning can learn from raw inputs, so hand-built features are optional. Classical machine learning **needs features** before the dataset is usable.

```text
classical ML:  data + features + algorithm
deep learning: data + algorithm   (features can be learned inside the net)
```

## What a feature is

Feature engineering is, in the usual case, a **mathematical transform that pulls a pattern out of the raw columns.** Sometimes the right feature is already a column heading and no transform is needed. That call belongs to the person building the model. The useful transforms depend on the dataset.

Three places to start:

### Binarization

Turn a raw value into a bit. Natural fit for images: the bits can be drawn back as a black-and-white picture.

$$
b(x) = \mathbf{1}[x > t]
$$

```python
def binarize(pixel, t=128):
    return int(pixel > t)  # 0 or 1; a grid of these is a binary image
```

### Binning

Group values into ranges so the scale collapses to a few bins.

$$
\mathrm{bin}(x) = j
\quad\text{when}\quad
e_{j-1} \le x < e_j
$$

```python
def bin_id(x, edges=(100, 1_000, 10_000)):
    # right-hand end of each bin. 50 → 0,  800 → 1,  50_000 → 3
    for j, edge in enumerate(edges):
        if x < edge:
            return j
    return len(edges)
```

Packet counts of $50$, $800$, and $50{,}000$ become bins $0$, $1$, and $3$. The model sees the range, not the raw magnitude.

### Log transform

Also shrinks scale. Unlike a flat bin, it **keeps small values visible** beside very large ones, because $\log$ grows slowly.

$$
x' = \log(1 + x)
$$

```python
import math

def log_feature(x):
    return math.log1p(x)  # log(1+x); 10 and 10_000 move much closer
```

| Raw $x$ | $\log(1+x)$ |
| ------- | ----------- |
| $10$ | $\approx 2.4$ |
| $100$ | $\approx 4.6$ |
| $10{,}000$ | $\approx 9.2$ |

Study cue: **start with binarize, bin, or log. Add a fancier transform only when the dataset asks for it.**

## Stop one feature from drowning the others

After the features exist, their **scales** may still disagree. A byte-count in the millions will dominate a flag in $\{0,1\}$ inside any distance or gradient, even when both columns matter.

Feature normalization is the fix. Two scalers to write:

**Min–max** maps a column onto $[0, 1]$.

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**Standardization** (z-score) centers a column and scales it by its spread.

$$
x' = \frac{x - \mu}{\sigma}
$$

```python
def minmax(x, lo, hi):
    return (x - lo) / (hi - lo)

def zscore(x, mu, sigma):
    return (x - mu) / sigma
```

| Method | Output | Use when |
| ------ | ------ | -------- |
| Min–max | $[0, 1]$ | You want a hard floor and ceiling |
| Z-score | mean $0$, spread $1$ | Outliers should not set the whole scale via $x_{\max}$ |

## Categories have to become numbers

Many algorithms only accept numeric columns. The right encoding depends on the dataset. Three usual options:

| Encoding | What you store | Watch-out |
| -------- | -------------- | --------- |
| Ordinal | One integer per category | The integers invent an order (`tcp=0 < udp=1`) |
| One-hot | One $0/1$ column per category | Wide, and no fake order |
| Dummy | One-hot with one column dropped | The dropped column is the baseline; avoids a redundant column |

Protocol $\{ \mathrm{tcp}, \mathrm{udp}, \mathrm{icmp} \}$:

| Value | Ordinal | One-hot $(t, u, i)$ | Dummy, icmp dropped $(t, u)$ |
| ----- | ------- | ------------------- | ---------------------------- |
| tcp | $0$ | $(1,0,0)$ | $(1,0)$ |
| udp | $1$ | $(0,1,0)$ | $(0,1)$ |
| icmp | $2$ | $(0,0,1)$ | $(0,0)$ |

```python
def one_hot(value, levels):
    return [int(value == level) for level in levels]

# one_hot("udp", ["tcp", "udp", "icmp"]) → [0, 1, 0]
```

Study cue: **ordinal when the categories are really ordered. One-hot or dummy when they are just names.**

## Generalization error

Model engineering starts from the error on **unseen** rows, the generalization error. Three parts:

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big]
=
\underbrace{\big(\mathbb{E}[\hat{f}(x)] - f(x)\big)^2}_{\mathrm{bias}^2}
+
\underbrace{\mathbb{E}\big[\big(\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\big)^2\big]}_{\mathrm{variance}}
+
\underbrace{\sigma^2}_{\mathrm{noise}}
$$

| Part | Grows when | What you do |
| ---- | ---------- | ----------- |
| Bias | The parameter assumptions are wrong | Relax a model that is too rigid |
| Variance | The fit flinches at every change in the sample | Simplify, or add data, so it stops memorizing |
| Noise | The labels or sensors are messy | Remove what you can. A remainder always stays |

Remove noise where you can. A remainder always stays. **Model complexity moves bias and variance.**

| Complexity | Bias | Variance | What you see |
| ---------- | ---- | -------- | ------------ |
| Too low | High | Low | Underfit. Train and test are both weak |
| Middle | Balanced | Balanced | The operating point |
| Too high | Low | High | Overfit. Train looks perfect, test does not |

```text
error
  | \  bias
  |  \
  |   \        total
  |    \      /
  |     \____/        ← middle: optimized complexity
  |     /    \
  |    /      \ variance
  +---------------- complexity
```

The point of this step is an **optimized model**: complexity, assumptions, and sensitivity to *this* data all sit in that middle.

```python
# Complexity knob: polynomial degree.
# Low degree → bias. High degree → variance. Pick the valley of test error.
def pick_degree(train_err, test_err):
    # parallel lists, index = degree
    return min(range(len(test_err)), key=lambda d: test_err[d])
```

Study cue: **bias = wrong assumptions. Variance = too sensitive. Aim between them.**

## Training-set size is part of the same picture

A second view of model engineering plots performance against **how many rows you trained on**, with bias and variance still on the chart.

| Training set | Typical pattern |
| ------------ | --------------- |
| Small | Train error stays low. Test error stays high (variance) |
| Larger | The two errors move toward each other |
| Large enough | Test performance levels off. More rows buy little |

An optimized model, from this view, has the **right training-set size** and a **test score** that matches that plateau. A high-bias model can be given more rows and still sit on a high error floor. A high-variance model usually improves as the set grows, which is why “more reliable data” wins so often.

```python
def learning_curve(rows, fit_and_score):
    # rows: growing prefixes of the training set
    # returns (n, train_score, test_score) once test_score flattens
    history = [fit_and_score(rows[:n]) for n in (50, 200, 1000, 5000)]
    return history
```

## ROC and AUC

The **ROC curve** plots true positive rate against false positive rate. Each point is one threshold. Precision, recall, and $F_1$ are in the alert-path notes above and in [module 2](module-02-fraud-prevention-with-cloud-ai-solutions.md).

$$
\mathrm{TPR} = \frac{TP}{TP + FN},
\qquad
\mathrm{FPR} = \frac{FP}{FP + TN}
$$

Sweep the threshold and you trace the curve. **AUC** collapses that curve to one number:

$$
\mathrm{AUC} = \int_{0}^{1} \mathrm{TPR}(\mathrm{FPR}) \, d(\mathrm{FPR})
$$

| AUC | Reading |
| --- | ------- |
| $0.5$ | No better than ranking at random |
| $\to 1$ | Malicious rows get higher scores than benign ones, across thresholds |

AUC is the tighter summary because a single TPR/FPR pair is only one operating point. A fraud detector can look strong at one cutoff and weak at the next; the area records the whole tradeoff.

```python
def roc_auc(fpr, tpr):
    # fpr, tpr: polyline from a threshold sweep, both starting at 0
    area = 0.0
    for i in range(1, len(fpr)):
        width = fpr[i] - fpr[i - 1]
        height = (tpr[i] + tpr[i - 1]) / 2  # trapezoid
        area += width * height
    return area
```

Study cue: **ROC is the curve. AUC is the number you compare models with.**
