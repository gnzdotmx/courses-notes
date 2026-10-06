# Module 5: Reinforcement Learning

## Agent, reward matrix, and Q-learning

### Objectives

- RL learns from reward and penalty, not from a labeled dataset
- Read $R$ on the maze, turn it into $Q$, take the row max to get the path
- $R$, $Q(s,a)$, and the policy do different jobs
- The agent is an orchestrator: paid only when smaller decisions land in order
- Where RL applies, and three ways to keep training data private

### Key concepts

| Term | Definition / notes |
| ---- | ------------------ |
| Agent | Learner that picks actions. Early on, trial and error (even random) is the point. **Fundamental** |
| Environment | The world the agent acts in, plus the feedback it returns. **Fundamental** |
| State $s$ | Where the agent is when it chooses. The $s$ in $Q(s,a)$ |
| Action $a$ | A move that traverses from one state to another. The $a$ in $Q(s,a)$ |
| Reward signal | Feedback on the action just taken. It guides learning. It does not label data, set weights, or set the network shape |
| Penalty | Negative feedback for moves that lead away from the goal |
| Reward matrix $R$ | Immediate feedback for each action: $r = R(s, a)$. It does **not** store $Q$, and it does **not** update the policy |
| $Q(s,a)$ | Expected **future** return for taking action $a$ in state $s$, then acting well afterward. **Fundamental** |
| $Q$ table | The matrix of those $Q(s,a)$ values after Q-learning has been applied to $R$ |
| Q-learning | Iterative update of $Q$. Convergence ⇒ a strategy that maximizes **cumulative** reward |
| Policy | The rule of action: $\pi(s) = \arg\max_a Q(s, a)$. A list of choices, separate from the $Q$ numbers |
| Orchestrator view | Agent as a task manager: reward only when smaller decisions land in the right order |

Study cue: **fundamentals are agent, environment, and $Q$.** Gradient descent and activation functions belong to neural-net training, not to this loop.

## Three kinds of machine learning

| Type | What it learns from |
| ---- | ------------------- |
| Supervised | Labeled training data |
| Unsupervised | Structure already in the data (e.g. distance between points) used to separate groups |
| Reinforcement | A different setup: an agent acts, the environment answers with reward or penalty |

Study cue: **RL’s theme is feedback from interaction.**

## The loop

```text
agent  -- action -->  environment (states)
agent  <-- reward or penalty --  environment
```

- Right actions move the agent toward the goal → **reward**.
- Wrong actions move it away → **penalty**.
- Early behavior is random trial and error. A careful action plan is unnecessary at the start.
- Winning the game means learning a strategy whose effect is to **maximize reward**.

Study cue: **goal = max reward, not a hand-written route.**

## States: the maze

Treat the environment as states. An action is a hop from one state to the next. Some states matter more because they lead to the **exit**.

Lecture maze (entrance **F**, exit **C**):

```text
F —— B —— D —— C  (C also links to itself, reward 100)
              |
              E
```

| State | Connects to | Role |
| ----- | ----------- | ---- |
| F | B only | Entrance |
| B | D, F | |
| D | B, C, E | Three-way choice |
| C | D, and C itself | Exit. Self-link pays **100** |
| E | (neighbor of D) | Side state; not the way out |

Every legal step is marked the same way in $R$. Only the exit self-loop is special (100), so the route itself has to be discovered by the algorithm.

## Reward matrix

$R$ is the feedback table. After the agent steps, the cell $R(s, a)$ says how good that step was. The learning rule then writes a different table, $Q$, which stores expected future reward. The policy is what reads $Q$ and picks the next action.

$$r = R(s, a), \qquad Q(s,a) \approx \mathbb{E}\big[G \mid s, a\big], \qquad G = \sum_{t} \gamma^t r_t$$

Lecture encoding:

| Cell | Meaning |
| ---- | ------- |
| $1$ | Valid connection between states |
| $0$ | Penalty (no useful transition) |
| $100$ | Reward (exit: C → C) |

Known rows (columns ordered F, B, D, E, C). E’s outgoing row is not given.

| From \ to | F | B | D | E | C |
| --------- | - | - | - | - | - |
| F | 0 | 1 | 0 | 0 | 0 |
| B | 1 | 0 | 1 | 0 | 0 |
| D | 0 | 1 | 0 | 1 | 1 |
| C | 0 | 0 | 1 | 0 | 100 |

A step is $s \xrightarrow{a} s'$ with payoff $r = R(s, a)$. In this maze the action *is* the choice of next state.

## From $R$ to a path: the $Q$ table

The **state-transition table** is $R$ after the **Q function** has been applied. One Q-learning step always takes the **largest value on the current row**:

| At | Row says | Go to |
| -- | -------- | ----- |
| F | Largest entry | B |
| B | Smaller entry returns to F; larger entry | D |
| D | B and E are **tied** and smaller; the greater entry | C |

Learned path:

$$F \rightarrow B \rightarrow D \rightarrow C$$

Two matrices on the slide (convergence of the same idea) both support that path if you repeat the row-max read. Once the updates settle, the agent has learned to maximize reward and can win the game it is playing.

Greedy policy after convergence:

$$\pi(s) = \arg\max_a Q(s, a)$$

## Q-learning, at the level this course uses

Study cue: **$Q$ is the missing piece between $R$ and the path.** Detailed algebra is explicitly out of scope; the course only needs the role of each term.

Usual tabular update:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \Big]$$

| Symbol | Role |
| ------ | ---- |
| $Q(s,a)$ | Value of taking $a$ in $s$, then acting greedily afterward |
| $r$ | Immediate reward from $R$ |
| $\alpha$ | Step size (how hard this sample overwrites the old value) |
| $\gamma$ | Discount on the best next value ($0 \le \gamma < 1$) |
| $\max_{a'} Q(s', a')$ | Best continuation from the next state |

The bracket is the surprise: immediate reward plus discounted future value, minus the value you already stored. Repeat over random episodes until $Q$ stops moving. That stable table is convergence.

## Timeline: the table before and after

- **$R$ (before).** The only filled table. Each cell is the immediate payoff of one step: $0$, $1$, or $100$. Fixed for the whole run.
- **$Q$ at the start.** Same shape as $R$, every cell $0$. No future value has been learned yet.
- **Q-learning (during).** A random legal step reads $r = R(s,a)$ and nudges that one cell toward $r + \gamma \max_{a'} Q(s', a')$.
- **Backup.** The exit self-loop is learned first, $Q(C, C) \to 1000$. That value then raises D → C, then B → D, then F → B.
- **$Q$ after convergence.** Cells stop moving. The row maxima are F → B, B → D, D → C.
- **Policy (readout).** $\pi(s) = \arg\max_a Q(s,a)$. The path is $F \rightarrow B \rightarrow D \rightarrow C$, kept separate from the numbers in $Q$.

## Worked example

Runnable maze. E’s outgoing links are not specified; this snippet sends E back to D, which is enough for episodes to continue and produces the tie (from D, B and E score the same, C scores higher).

```python
import numpy as np

states = ["F", "B", "D", "E", "C"]  # C = exit
# 1 = legal step, 0 = penalty, 100 = goal (C stays at C)
R = np.array([
    [  0,  1,  0,  0,   0],  # F → B
    [  1,  0,  1,  0,   0],  # B → F, D
    [  0,  1,  0,  1,   1],  # D → B, E, C
    [  0,  0,  1,  0,   0],  # E → D (outgoing link not specified)
    [  0,  0,  1,  0, 100],  # C → D, C → C
], dtype=float)

Q = np.zeros_like(R)
alpha, gamma = 0.8, 0.9
rng = np.random.default_rng(0)

def legal(s):
    return np.flatnonzero(R[s] > 0)

for _ in range(500):
    s = 0  # start at the entrance
    for _step in range(30):
        a = int(rng.choice(legal(s)))          # trial and error
        s2 = a                                  # action = next state
        Q[s, a] += alpha * (R[s, a] + gamma * Q[s2].max() - Q[s, a])
        s = s2

# greedy path: largest Q on each row → F → B → D → C
```

Example $Q$ after that run (rounded). Take the max on each row.

| From \ to | F | B | D | E | C |
| --------- | - | - | - | - | - |
| F | 0 | **732** | 0 | 0 | 0 |
| B | 660 | 0 | **812** | 0 | 0 |
| D | 0 | 732 | 0 | 732 | **901** |
| C | 0 | 0 | 812 | 0 | **1000** |

C → C sits at the fixed point of the self-loop, $100 + \gamma Q \Rightarrow Q = 100 / (1 - \gamma) = 1000$ when $\gamma = 0.9$. That 1000 backs up along D, then B, then F, so the greedy path is forced toward the exit.

## Using RL as an orchestrator

In security the agent is a **task manager**, not a worker:

- The agent moves through states of a larger job.
- The small tasks stay outside the agent. The agent only moves from state to state and checks whether they are done.
- A reward arrives when the checks pass, and only for the **best decisions in a specific order**.
- Stack that idea: an agent can learn a large decision by managing a series of smaller ones.

Study cue: **reward the sequence, and the policy becomes the playbook.** The same loop (states, $R$, $Q$) is what you would wire up for that orchestrator.

## RL attacker: evade a detector you have never seen

### Objectives

- A neural-net attacker still needs the target’s features. An RL attacker needs a score
- Anderson et al.: local detector as the environment, functionality-preserving PE edits as the moves
- Reward falls as the detection score rises. Policy is still $\arg\max_a Q(s,a)$

### Key concepts

| Term | Definition / notes |
| ---- | ------------------ |
| NN white-box | Full access to the target model |
| NN black-box | Less access, but the attacker still learns the target’s **feature space** |
| RL attacker | No structure and no features. Explore, take reward or penalty, maximize reward |
| Local detector | Surrogate malware model, trained on a large set. It *is* the environment |
| Preserving edit | PE layout changes; program behavior stays the same |
| Action set | Those edits: IAT imports, section renames, unused sections, section padding |
| State (study MDP) | The current variant. Lecture also calls the edit menu the allowable states |
| Reward | Higher when the local model’s detection level is lower |
| Transfer | A strong local model stands in for the real detector, so local evasion carries over |

### What the supervised attacker still needs

Earlier attacks in this course were neural nets, so supervised. Two access levels:

| Attack | Access | Still required |
| ------ | ------ | -------------- |
| White-box | Full model | Weights, structure, gradients |
| Black-box | Partial | Enough interaction to learn the target **feature space** |

An RL agent is set up differently. It samples actions in an environment, collects reward and penalty, and keeps only what increases return. That search can produce evasive samples for a detector whose structure and features were never given to the agent.

Study cue: **NN attackers need a feature space. The RL attacker needs a score.**

### Anderson et al. at Black Hat

H. S. Anderson and co-authors turned that idea into a malware-evasion setup, with an open environment others can experiment in. Reported evasion is in the same range as NN adversarial attacks, and higher on some detectors.

Setting: static Windows malware models. Useful features come from the **PE** file: headers and sections. The authors do not train against the remote target. They fit a **local** detector on a large dataset and let the agent live inside that model.

Before learning starts they fix a menu of **functionality-preserving** edits: the bytes that change are PE metadata, and the payload’s behavior stays the same. That menu is the space the agent is allowed to move through.

```text
current malware
    -- one preserving edit -->  new PE, same behavior
new PE
    -- detection level -->  local detector
agent
    <-- reward or penalty --
```

Each step builds another variant and submits it. Payoff tracks how detected that variant is. Maximizing reward means a file that still does the original job, with a PE different enough that the detector misses it.

Transfer check: if the local model is representative of the target, or stricter, beating the local model is treated as beating the target. The authors checked this by also evading publicly hosted detectors.

They released the same setup as a public **RL malware-evasion framework**, so the state and action lists are concrete.

### States and actions in the framework

Where an edit can land:

- Fields in the PE header
- PE sections
- Import table and export table
- ASCII strings

Moves the agent can take:

- Add functions to the import address table
- Rename an existing section
- Add a new unused section
- Append space at the end of a section

Every move returns a detection level. Higher reward goes to the edit that leaves that level lower. Same rule as the maze: the objective pays for the sequence that maximizes reward, here a sequence of PE edits.

MDP reading (the edit menu is sometimes called the states; below, the current binary is the state and each edit is an action):

| MDP piece | In this attack |
| --------- | -------------- |
| Agent | Policy that picks the next edit |
| Environment | Local detector plus the PE it scores |
| $s_t$ | Variant after the edits so far |
| $a_t$ | One preserving modification |
| $r_t$ | From how malicious the local model still thinks $s_{t+1}$ is |

### Return

Let $p(s) \in [0, 1]$ be the local detector’s malware score. Immediate reward falls as that score rises:

$$r_t = 1 - p(s_{t+1})$$

Return to maximize:

$$G = \sum_{t=0}^{T} \gamma^t r_t, \qquad \pi^* = \arg\max_\pi \mathbb{E}_\pi[G]$$

After a $Q$ table exists, the policy is the maze policy again:

$$\pi(s) = \arg\max_a Q(s, a)$$

The “exit” is a low detection score. $\gamma$ still discounts later edits, so an early edit that enables a later miss is worth more than a late one.

### Toy loop

One state, four labeled edits. Values are stub detection levels $p$ from a local model. The agent only sees $r = 1 - p$, then takes the max, as on the maze.

```python
# p = local detector score after that edit (higher = more malware-like)
actions = {
    "add_import": 0.80,
    "rename_section": 0.55,
    "add_section": 0.30,
    "pad_section": 0.70,
}

def reward(p: float) -> float:
    return 1.0 - p

best = max(actions, key=lambda a: reward(actions[a]))  # add_section, r = 0.70
```

- The agent never needs the target’s weights or feature list. It needs the local model’s detection level after each edit.
- Edits that change behavior are off the menu. The reward is for the same program in a new PE.
- Local-model quality is the transfer story. A weak surrogate can be evaded and still say nothing about the real detector.
- This is the orchestrator pointed at a detector: a sequence of small edits, paid only when the score drops.

## RL attacks vs neural-net attacks

| | RL attack | NN attack |
| - | --------- | --------- |
| How it learns | Trial and error, continuous feedback | A **static pre-trained** model |
| Data | Reward signal. No labeled set | Supervised training of that model |
| Over time | Adapts when the environment changes | Fixed until someone retrains it |
| Objective | Maximize **cumulative** reward $G = \sum_t \gamma^t r_t$ | Fool the current model (FGSM, PGD, GAN) |

The reward signal is required. Drop it and the agent has nothing to climb. These attacks are not a supervised-learning setup.

Best defense among the usual training tricks: **adversarial training** (train on the adversarial examples). Gradient clipping, data augmentation, and batch normalization are not that defense.

### Where RL shows up

| Application | Why it fits |
| ----------- | ----------- |
| Intrusion detection | Policy that updates as traffic and threats change |
| Adaptive traffic signals | Timing actions, reward from flow or delay |
| Automated trading | Sequential buy/sell actions, reward from return |

Speech-to-text and face recognition are supervised models, so they sit outside this loop.

## Privacy of training data

These three protect privacy:

| Technique | What it does |
| --------- | ------------ |
| Homomorphic encryption | Compute on ciphertext. The data stays encrypted during processing |
| Differential privacy | Add calibrated noise so one person’s record cannot be singled out of the result |
| Federated learning | Many devices train one model. **Raw data stays on the device**; only updates move |

Federated learning’s job is that privacy split, not faster training, not less compute on one machine, and not easier app integration.

These two are useful and **do not** add privacy:

| Technique | What it actually does |
| --------- | --------------------- |
| Data normalization | Rescales features |
| Early stopping | Halts training to limit overfitting |
