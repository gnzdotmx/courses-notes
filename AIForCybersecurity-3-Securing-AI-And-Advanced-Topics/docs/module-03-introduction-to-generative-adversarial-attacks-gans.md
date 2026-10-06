# Module 3: Introduction to Generative Adversarial Attacks (GANs)

## Objectives

- Two nets compete until neither can improve alone. That is how AI can attack AI
- Deep nets learn their own features and are harder to explain
- Perceptron vs MLP; forward pass vs backprop; feed-forward, CNN, RNN
- Write $\min_G \max_D V$ and the alternating training loop
- Walk a GAN through data → model → train, using MNIST

## Key concepts

| Term | Definition / notes |
| ---- | ------------------ |
| GAN | Two NNs in competition until a **balance** (neither can improve by changing alone). Goodfellow, ~2014 |
| Generator $G$ | Maps noise (+ optional real glimpse) → fake samples meant to fool $D$ |
| Discriminator $D$ | Scores real vs fake; wants $D(x)\to 1$, $D(G(z))\to 0$ |
| Zero-sum game | One player’s gain = the other’s loss |
| Nash equilibrium | Both strategies optimized; neither improves by changing alone $\approx D\!\approx\!1/2$ |
| Minimax objective | $D$ maximizes, $G$ minimizes the same value function $V(D,G)$ |
| AI vs AI | GANs show that AI can be used to **attack** AI |
| Deep learning | Subset of ML; stacked artificial neurons that **learn their own features** |
| Explainability gap | DL often beats classical ML on accuracy but has **no clear reasoning** for the win |
| Perceptron | Single unit; only **linearly separable** problems |
| MLP | Layers of perceptrons; with a nonlinear activation, can approximate **any continuous** function (enough units) |
| Activation | Maps a linear score to a (usually nonlinear) response once a threshold is crossed |
| ReLU | $\mathrm{ReLU}(z)=\max(0,z)$ |
| Leaky ReLU | $\mathrm{LeakyReLU}(z)=\max(\alpha z, z)$, $\alpha \ll 1$; avoids dead zeros |
| Forward prop | Data / activations flow **input → output**; features are built along the way |
| Backprop | **Training** only: nudge weights from $\partial L / \partial w$ so output matches labels |
| Feed-forward NN | Data flow **forward only** (most common; CNN is this family too) |
| CNN | Feed-forward + **kernels / filters** that focus on local features |
| RNN | Extra connections so information can move **back and forth** (sequence memory) |
| MNIST | Handwritten digits; common GAN toy dataset |

## What a GAN is

Two neural nets are placed in **competition**. They train against each other until a **balanced** condition (a stalemate: generated samples look like real data; the detector cannot reliably tell them apart).

Study cue: **AI can attack AI.**

| Player | Goal | Not optimizing for |
| ------ | ---- | ------------------ |
| $G$ generator | Create fakes so close to real that $D$ cannot tell | Classical training-error alone |
| $D$ discriminator | Tell **real** from **fake** | Generating samples |

They **learn from each other** (coupled losses). Details below.

Standard two-player game (memorize the shape):

$$
\min_G \max_D \;
V(D,G) =
\mathbb{E}_{x \sim p_{\mathrm{data}}}[\log D(x)]
+ \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

| Player | Job | Wins when |
| ------ | --- | --------- |
| $D$ discriminator | Score real vs fake | $D(x) \to 1$, $D(G(z)) \to 0$ |
| $G$ generator | Turn noise $z$ into a fake $G(z)$ | $D(G(z)) \to 1$ (fakes fool $D$) |

Balance $\approx$ $D$ is guessing (~$1/2$) and $G$’s samples sit on $p_{\mathrm{data}}$.

```python
# Conceptual loop only — not a trained GAN
for _ in range(steps):
    x_real = sample_data()
    z = sample_noise()
    x_fake = G(z)
    # D: real → 1, fake → 0;  G: make D(x_fake) → 1
    d_loss = bce(D(x_real), 1) + bce(D(x_fake.detach()), 0)
    g_loss = bce(D(x_fake), 1)
```

---

## Deep learning (why GANs are “deep”)

Deep learning = ML that **mimics stacked brain-like units**. Layers of connected neurons → **deeply connected** nets.

| | Classical ML | Deep learning |
| - | ------------ | ------------- |
| Features | You (or a pipeline) design them | Net **invents and combines** features |
| Accuracy | Often lower on raw, complex data | Often **higher** |
| Explainability | Usually clearer | **Not inherent** — complexity hides the “why” |

Cyber hat: we only go deep enough to **use** these models for triage, not to become NN theorists.

---

## Perceptron → MLP

**Rosenblatt perceptron** — one unit, linear decision. Only works if classes are **linearly separable** (a hyperplane splits them).

$$
\hat{y} = f(w^\top x + b)
$$

Linearly separable $\iff$ $\exists\, w,b$ such that $y_i(w^\top x_i + b) > 0$ for all labeled pairs.

**MLP** = layers of perceptrons. Nonlinear $f$ + enough units $\Rightarrow$ can approximate **any continuous** function (universal approximation). That is the leap GANs (and most DL) sit on.

```text
x  →  [perceptron]  →  ŷ     only linear splits

x  →  [layer] → [layer] → … → ŷ     MLP: stacked, nonlinear
```

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def mlp_forward(x, weights, biases, act=relu):
    a = x
    for W, b in zip(weights, biases):
        a = act(W @ a + b)          # linear score → nonlinear response
    return a
```

### Activations (linear in, nonlinear out)

A unit fires when its score crosses a threshold; $f$ turns that into the next layer’s input. Nonlinearity is what lets the net **recreate mathematical functions**, not just hyperplanes.

$$
\mathrm{ReLU}(z) = \max(0, z)
\qquad
\mathrm{LeakyReLU}(z) =
\begin{cases}
z & z > 0 \\
\alpha z & z \le 0
\end{cases}
$$

| Activation | Shape | Study cue |
| ---------- | ----- | --------- |
| ReLU | $0$ then slope $1$ | Cheap; can “die” if $z<0$ always |
| Leaky ReLU | Small slope $\alpha$ for $z<0$ | Keeps a gradient on the negative side |

---

## Forward vs back: data flow ≠ training

Think of the net as **rebuilding a function** from data.

**Forward propagation** (inference / data flow): input hits layer 1 → activation → layer 2 → … Features (and new combinations) are formed **on the way forward**.

$$
a^{(0)} = x, \qquad
a^{(\ell)} = f\!\left(W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}\right), \qquad
\hat{y} = a^{(L)}
$$

**Backpropagation** (training, not a second data path in a feed-forward net): after $\hat{y}$ is compared to the labeled $y$, weights move so $\hat{y}$ gets as close as the architecture allows.

$$
L = L(\hat{y}, y)
\qquad
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

$\partial L / \partial w$ comes from the **error** (output vs training target), chained through layers. Goal: match the training set as closely as **neuron count + other knobs** allow.


```python
# Tiny supervised step: forward, then one weight update
y_hat = mlp_forward(x, weights, biases)
loss = ((y_hat - y) ** 2).mean()
# backprop ≈ autograd of loss w.r.t. each W, b; then:
# W -= lr * dL/dW
```

| Word | Means here |
| ---- | ---------- |
| Forward | How **data** moves (and how features form) |
| Backward / backprop | How **training** adjusts weights after a mismatch |

---

## Three architectures (by data path)

| Net | Data flow | Extra trick |
| --- | --------- | ----------- |
| Feed-forward | **Forward only** | Default MLP-style stack |
| CNN | Still **feed-forward** | **Kernels / filters** lock onto local features |
| RNN | **Back and forth** between units | Recurrence = memory of earlier steps |

CNN is *not* a different data-flow family: it is feed-forward with specialized filter layers.

RNN is the one that **allows information backward** in the architecture (loops), unlike backprop-as-training.

```text
FF / CNN:   x → h1 → h2 → ŷ          (CNN: hℓ uses kernels)

RNN:        x_t → h_t → ŷ_t
                 ↑  ↓
                h_{t-1}              (state travels along the sequence)
```

```python
# RNN step: hidden state carries the past forward (and, via unfolding, backward in time)
def rnn_step(x_t, h_prev, Wx, Wh, b, act=np.tanh):
    h_t = act(Wx @ x_t + Wh @ h_prev + b)
    return h_t
```

---

## Generator vs discriminator (deeper)

Both are NNs; roles differ.

| | Generator $G$ | Discriminator $D$ |
| - | ------------- | ----------------- |
| Input | Noise $z$ (+ glimpse of real) | Real $x$ **or** fake $G(z)$ |
| Output | Synthetic sample | $P(\text{real})$ / real-vs-fake score |
| Drive | **Fool** $D$ | **Catch** fakes |
| Obj. direction | Minimize $V$ (via faking well) | Maximize $V$ (via discerning well) |

$G$ is **not** a normal classifier minimizing training error — its loss comes from how well $D$ still sees through the fakes.

```text
z ~ noise ──► G ──► x_fake ──┐
                              ├──► D ──► real? / fake?
x_real ───────────────────────┘
         ▲                         │
         └──── loss signal to G ───┘
```

---

## Spam example: fake spam vs real spam

| Player | Goal |
| ------ | ------------------- |
| $D$ | Discern **real spam** from **fake spam** |
| $G$ | Create fake spam **indistinguishable** from real spam |

Not the classic “is this spam or ham?” classifier problem.

Intuition:

- $D$: $P(\text{spam} \mid \text{suspect words})$ — how likely is this the real thing?
- $G$: craft messages so those cues look authentic enough to pass $D$

Study cue: **generator invents plausible threats; discriminator is the detector under pressure.**

Fakes that sit on the real distribution can fill a rare class, or pressure-test the detector.

Toy of the roles only — $G$ walks a cue vector toward real spam; $D$ scores the match. No ham in the batch.

```python
import numpy as np

# Columns: free, click, verify, account
# Every row is spam. D never sees ham — only real campaign vs invented copy.
real_spam = np.array([
    [0.9, 0.8, 0.7, 0.3],
    [0.8, 0.9, 0.6, 0.4],
    [0.85, 0.75, 0.8, 0.2],
])
profile = real_spam.mean(axis=0)          # what real spam looks like

def D(x):
    """High = drawn from real spam. Low = invented."""
    return np.exp(-np.sum((x - profile) ** 2, axis=-1))

def G(z, step):
    """Walk a noisy cue vector toward the real-spam profile."""
    return z + step * (profile - z)

z = np.array([0.1, 0.2, 0.0, 0.9])        # "account" without the usual cues
print("D(real spam)", D(real_spam).round(2))
for step in (0.0, 0.5, 1.0):
    fake = G(z, step)
    print(f"step={step:.1f}  fake={fake.round(2)}  D(fake)={D(fake):.2f}")
# step 0: wrong cues, D(fake) ~ 0.17
# step 1: fake sits on the profile, D(fake) = 1  (same call D gives real spam)
```

---

## Zero-sum + Nash + Goodfellow’s equation

Three ideas, one equation. Study them in this order: the shared prize, who climbs or descends it, then the Nash numbers that say the game is over.

```text
1. Zero-sum     one score V
                D is paid +V, G is paid −V  →  payoffs sum to 0
        │
        ▼
2. Minimax      min_G max_D V(D,G)
                inner max: D climbs V;  outer min: G pulls that result down
        │
        ├── term 1   E[ log D(x) ]              x ~ p_data     only D
        │
        └── term 2   E[ log(1 − D(G(z))) ]      z ~ p_z → G → D
                     this is the contested term
        │
        ▼
3. Best D       freeze G, then
                D*(x) = p_data(x) / (p_data(x) + p_g(x))
        │
        ▼
4. Nash         plug D* in:
                V = −log 4 + 2·JSD(p_data ‖ p_g)
                JSD = 0  ⇒  p_g = p_data  ⇒  D* = 1/2  ⇒  V* = −log 4
```

**Zero-sum.** One number $V(D,G)$. Discriminator’s payoff is $+V$; generator’s payoff is $-V$. If $D$ gains $\Delta$, $G$ loses $\Delta$. The sum of payoffs is $0$.

**Nash equilibrium.** Both strategies are optimal, and neither player gains by changing **only their own** move while the other stays put. Here that balance is $p_g = p_{\mathrm{data}}$ and $D \approx 1/2$ (a coin-flip on real vs fake). $p_g$ is the distribution of fakes $G(z)$ when $z \sim p_z$.

Goodfellow’s value function is that shared $V$:

$$
\min_G \max_D \; V(D,G)
= \mathbb{E}_{x \sim p_{\mathrm{data}}}[\log D(x)]
+ \mathbb{E}_{z \sim p_z}[\log\!\bigl(1 - D(G(z))\bigr)]
$$

### Each piece

A probability in $(0,1)$ has $\log \le 0$. Closer to $0$ is a higher score. Driving a log toward $-\infty$ is how a player wrecks $V$.

| Piece | In words | Who moves it |
| ----- | -------- | ------------ |
| $\min_G$ | Choose $G$ to shrink $V$ | Generator, outer player |
| $\max_D$ | Choose $D$ to grow $V$ | Discriminator, inner player |
| $V(D,G)$ | The single shared score | $+V$ to $D$, $-V$ to $G$ |
| $\mathbb{E}_{x \sim p_{\mathrm{data}}}[\,\cdot\,]$ | Average over real samples $x$ | Data distribution, fixed |
| $\log D(x)$ | Log of “$x$ is real” | $D$ only. Rises toward $0$ as $D(x)\to 1$ |
| $\mathbb{E}_{z \sim p_z}[\,\cdot\,]$ | Average over noise draws $z$ | Prior $p_z$ (often Gaussian), fixed |
| $G(z)$ | Noise mapped to one fake | $G$’s only lever inside $V$ |
| $D(G(z))$ | “This fake looks real” | $D$ and $G$ pull it opposite ways |
| $1 - D(G(z))$ | “This fake is fake” | High when $D$ rejects the fake |
| $\log(1-D(G(z)))$ | Log of that rejection | Toward $0$ if $D$ wins the fake; toward $-\infty$ if $G$ fools $D$ |
| $+$ | Add the two averages into one $V$ | The zero-sum glue: both terms share one prize |

Term 1 has no $G$ in it, so the generator cannot change the real-data score. The fight sits in term 2.

| Term | $D$ grows $V$ ($G$ frozen) | $G$ shrinks $V$ ($D$ frozen) |
| ---- | -------------------------- | ---------------------------- |
| Real: $\log D(x)$ | Push $D(x)\to 1$ so the log $\to 0$ | No handle on this term |
| Fake: $\log(1-D(G(z)))$ | Push $D(G(z))\to 0$ so the log $\to 0$ | Push $D(G(z))\to 1$ so the log $\to -\infty$ |

### From best $D$ to Nash

Freeze $G$. The $D$ that maximizes $V$ is

$$
D^*(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_g(x)}.
$$

Substitute that $D^*$ back in. Goodfellow’s result is

$$
\max_D V(D,G)=-\log 4+2\,\mathrm{JSD}(p_{\mathrm{data}}\,\|\,p_g).
$$

Jensen–Shannon divergence is a distance between two distributions. It is $\ge 0$, and it is $0$ only when the distributions match. So the outer $\min_G$ is solved only at $p_g=p_{\mathrm{data}}$. Then $D^*=1/2$ on every $x$, each log equals $\log\tfrac{1}{2}=-\log 2$, and

$$
V^*=-\log 2-\log 2=-2\log 2=-\log 4.
$$

Nash check at that point:

- $G$ moves $p_g$ off the data. JSD becomes positive, so the best $D$ can raise $V$ above $-\log 4$. $G$ does not gain.
- $D$ leaves $1/2$ while $p_g$ still matches the data. $D^*$ was already the maximum, so $V$ falls. $D$ does not gain.

The equation is the **target game**. Training does not solve the inner $\max_D$ to completion each round. The loop in the next section takes one step of $D$, then one step of $G$.

Signs, so the code matches the formula:

```python
# V itself: D maximizes this, G minimizes it
V = log(D(x_real)).mean() + log(1 - D(G(z))).mean()

# Same game, written as losses to minimize:
d_loss = -log(D(x_real)).mean() - log(1 - D(G(z))).mean()   # = -V
# Practical G step: log(1 - D) is flat while D still rejects everything,
# so G is trained to maximize log D(G(z)) instead (same goal: fool D).
g_loss = -log(D(G(z))).mean()
```

---

## How GAN training differs from one classifier

Not: input → output → backprop once on a single loss.

Instead: **two** nets, **competing** objectives, losses intertwined. Compute cost is high; wall time often **hours → days**.

Training rules:

1. **Pretrain $D$** on real data (teach “what real looks like”).
2. **Match learning rates** so $G$ and $D$ stay equally strong (avoid one crushing the other).
3. Feed $G$: **noise + glimpse** of real data → try an indistinguishable copy.
4. $D$ keeps training on real **and** scores $G$’s copies; sends a loss signal back to $G$.
5. Loop until equilibrium.

```text
pretrain D on real
        │
        ▼
   ┌─ noise + glimpse → G → fake ─┐
   │                              ▼
   │                         D(real, fake)
   │                              │
   └──────── G ← loss ←───────────┘
        repeat until D ≈ 1/2
```

```python
# Sketch of alternating updates (PyTorch-ish pseudocode)
opt_D, opt_G = ..., ...

for epoch in range(E):
    # --- Discriminator step ---
    x = next_real_batch()
    z = torch.randn(batch, z_dim)
    fake = G(z).detach()                 # stop G grads here
    d_loss = bce(D(x), 1) + bce(D(fake), 0)
    opt_D.zero_grad(); d_loss.backward(); opt_D.step()

    # --- Generator step ---
    z = torch.randn(batch, z_dim)
    g_loss = bce(D(G(z)), 1)             # want D to say "real"
    opt_G.zero_grad(); g_loss.backward(); opt_G.step()
```

---

## MNIST demo (handwritten digits)

| Stage | What you see |
| ----- | ------------ |
| Data | MNIST digit images (excerpt) |
| $G$ input | Noise + some real-data character |
| Early output | Messy / not digits yet |
| After hours | Digits emerging; **not** at equilibrium yet, but clearly mimicking |

Study cue: visual progress of $G$ is the best “are we getting there?” check before losses look balanced.

---

## Analytic development process (GAN lens)

| Stage | In this demo |
| ----- | ------------ |
| Data engineering | MNIST already prepared for the GAN |
| Feature engineering | DL invents features — you **don’t** hand-pick them |
| Model engineering | Tune knobs carefully; keep $G$ and $D$ **equally equipped** (both use **leaky ReLU**) |
| Training | Alternating coupled updates (see above) |

Warning: over-tuning one net can put them **out of sync** — then equilibrium stalls.

---

## Training progression (early vs late epochs)

| Phase | $D$ | $G$ | Losses |
| ----- | --- | --- | ------------------------ |
| Early | Easy to spot fakes | Weak copies | $L_D$ low (good for $D$), $L_G$ high — **opposite** of the desired end state |
| Later | Harder to tell real from fake | Stronger copies | $L_G$ falls; $L_D$ rises toward chance |

$$
\text{early: } L_D \downarrow,\; L_G \uparrow
\qquad\longrightarrow\qquad
\text{late / eq.: } L_G \downarrow,\; L_D \uparrow \approx \text{coin-flip } D
$$
