# Module 1: Course Introduction

## Objectives

- Understand what AI is and how it relates to cybersecurity
- Understand course structure and prerequisites
- Set up any required tools or environments: [venv setup](#environment-setup)

## Key concepts


| Term      | Definition                                            |
| --------- | ----------------------------------------------------- |
| AI        | Algorithms that simulate human intelligence           |
| Weak AI   | Narrow, task-specific: all production AI today       |
| Strong AI | Human-level general intelligence: not achieved       |
| Super AI  | Exceeds human capability: theoretical                |
| ML        | AI subfield: learns patterns from data                |
| DL        | ML subfield: deep neural networks, many hidden layers |


# Notes

### Context

- Coined **1956** (Dartmouth); cycles of hype and stagnation tied to **funding**
- **AI winter** ~ reduced funding/progress (e.g. ~1974)
- Current wave since **~2012**: deep learning + enough **data & compute**

### AI subfields


| Category             | Examples                        |
| -------------------- | ------------------------------- |
| Behavior             | Robotics                        |
| Perception           | NLP, computer vision            |
| Cognition & learning | ML, fuzzy logic, expert systems |


ML dominates today, data-driven, and data/compute are widely available.

### Weak vs strong vs super


| Type       | Scope                | Status       | Example                                    |
| ---------- | -------------------- | ------------ | ------------------------------------------ |
| **Weak**   | One bounded task     | Today        | Driver-assist (not full autonomy)          |
| **Strong** | Human-level, general | Not achieved | Sci-fi fully autonomous cars               |
| **Super**  | Better than humans   | Theoretical  | Humans banned from driving: machines only |


**Takeaway:** Production = weak AI (mostly ML). Strong/super = research or fiction.

### AI ⊃ ML ⊃ DL


| Layer  | What it is                                                                                                   |
| ------ | ------------------------------------------------------------------------------------------------------------ |
| **AI** | Broad umbrella: rules, search, fuzzy logic, ML, …                                                            |
| **ML** | Learns from data (supervised, unsupervised, reinforcement)                                                   |
| **DL** | Neural nets with many hidden layers; auto-learns features; needs lots of data/compute; often a **black box** |

#### ML in one line

Supervised learning finds a function $f$ that maps inputs $x$ to outputs $y$ by minimizing error on labeled data:
$$

\hat{y} = f(x; \theta) \qquad \text{learn } \theta \text{ to minimize prediction error on training labels}

$$

Unsupervised learning finds structure without labels (clusters, reduced dimensions). Reinforcement learning maximizes cumulative **reward** from environment feedback.

### Why AI for cybersecurity

- Tech runs **16 critical infrastructure sectors**: under-defended, remotely exploitable
- Not enough analysts → AI as **force multiplier** for **Tier 1** triage & incident response

## Course structure

### Prerequisites

- **Python 3.10+** installed (`python3 --version`)

### Environment setup

Create an isolated **venv** for course scripts and notebooks, keeps dependencies separate from system Python.

```bash
# from the course repo root
cd /path/to/1-AI-for-Cybersecurity-Intro-to-AI-for-cybersecurity

python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn jupyter
```

**Verify:**

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

**Daily use:**

```bash
source .venv/bin/activate   # activate before running scripts or Jupyter
deactivate                  # when done
```


| Item     | Notes                                                               |
| -------- | ------------------------------------------------------------------- |
| `.venv/` | Local env: add to `.gitignore` if you version-control this repo    |
| Jupyter  | `jupyter notebook` or `jupyter lab`: match kernel to this venv     |


**Optional:** freeze deps after installs → `pip freeze > requirements.txt`

---

## Summary

- AI is broad and old
- Today's wave is ML/DL on big data. 
- All deployed systems are **weak AI**. Know the **AI → ML → DL** stack. 
- This course uses ML to augment Tier 1 security analysts.