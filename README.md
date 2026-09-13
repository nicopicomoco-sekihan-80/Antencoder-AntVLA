Antencoder-AntVLA

Small, field-deployable VLA with language-conditioned latent distributions.

AntVLA is designed around a simple principle:

Do not force language to determine information that language does not specify.

Rather than building a large model that tries to understand arbitrarily complex instructions, we aim to build a small VLA that works reliably on simple real-world manipulation tasks.

Core Idea

AntVLA represents visual information and action information as continuous latent variables:

𝑧
𝑉
=
𝐸
𝑉
(
𝑉
)

𝑧
𝐴
=
𝐸
𝐴
(
𝐴
)

where:

𝐸
𝑉
: Vision Encoder
𝐸
𝐴
: Action Encoder
𝑧
𝑉
: Vision latent
𝑧
𝐴
: Action latent

For vision, MobileCLIP is used as the main visual encoder.

The key idea is that Language should not be forced to reconstruct a single point in either latent space.

Instead, Language defines a conditional distribution:

𝑝
(
𝑧
𝑉
∣
𝐿
)

𝑝
(
𝑧
𝐴
∣
𝐿
)

where 
𝐿
 is language.

Why Distribution Instead of MSE?

A language instruction is inherently ambiguous.

For example:

"pick the cube"


may correspond to:

red cube
blue cube
green cube


Similarly, the same instruction may be executed using different valid trajectories:

approach from the left
approach from the right
approach from above


A conventional MSE objective tries to predict a single latent:

𝐿
→
𝑧
^

This can produce an average latent that does not correspond to any valid visual state or action.

AntVLA instead models:

𝐿
→
𝑝
(
𝑧
∣
𝐿
)

so that multiple valid solutions can coexist.

Unified View

The same principle is applied to both Vision and Action:

                    Language
                        │
                 Language Encoder
                        │
                  Language latent
                        │
              ┌─────────┴─────────┐
              ↓                   ↓
        p(z_V | L)           p(z_A | L)
              ↓                   ↓
        Vision space          Action space
            What                 How


Thus:

Vision latent captures what / state.
Action latent captures how.
Language latent provides a compact semantic representation from which both distributions can be recovered.

Language is therefore not the final goal.

The ultimate goal is a compact representation that supports robust robot intelligence.

Proposed Architecture
                RGB / Observation
                       │
                  MobileCLIP
                       │
                  Vision latent
                       │
                       │
                       ├─────────────┐
                       │             │
                       │          Fusion
                       │             ↑
                       │             │
Robot Action ──→ Action Encoder ─→ Action latent
                                     │
                                     ↓
                                  VLA / Policy
                                     │
                                     ↓
                                Robot Action


Language
   │
   ↓
Language Encoder
   │
   ↓
Language latent
   ├────────→ p(z_V | L)
   │
   └────────→ p(z_A | L)


The initial implementation can use a Gaussian or Mixture Density Network (MDN):

𝑝
𝜃
(
𝑧
∣
𝐿
)
=
∑
𝑘
=
1
𝐾
𝜋
𝑘
(
𝐿
)
𝑁
(
𝑧
;
𝜇
𝑘
(
𝐿
)
,
Σ
𝑘
(
𝐿
)
)

with negative log-likelihood:

𝐿
N
L
L
=
−
log
⁡
𝑝
𝜃
(
𝑧
∣
𝐿
)

Research Hypothesis

We hypothesize that:

Modeling Vision and Action latents as language-conditioned distributions is more appropriate than point regression because language does not uniquely specify visual state or action realization.

Furthermore:

As language contains more information about the visual state or action, the corresponding conditional distribution should become more concentrated.

For example:

"pick cube"
        ↓
high uncertainty

"pick red cube"
        ↓
lower visual uncertainty

"pick red cube on the left"
        ↓
lower visual/state uncertainty


The same principle applies to action realization.

Data Philosophy

AntVLA does not aim to solve arbitrarily complicated instructions.

The target is:

simple tasks
×
diverse objects
×
diverse environments
×
diverse trajectories


Examples of primitive tasks:

Pick
Place
Push
Pull
Open
Close
Insert
Stack


The objective is to make a small model robust to real-world variation, rather than making a large model capable of interpreting increasingly complicated instructions.

Current Progress
AntVLA v6-alpha

Action → Language baseline:

PickCube
100 / 100 = 100.0%

StackCube
100 / 100 = 100.0%

Overall
200 / 200 = 100.0%


The model also achieves:

UNSEEN TRAJECTORY TEST

PickCube
100 / 100 = 100.0%

StackCube
100 / 100 = 100.0%

Overall
200 / 200 = 100.0%


These experiments establish the initial Action Encoder → Language baseline.

The next step is to introduce visual information and replace point-based latent regression with language-conditioned latent distributions.

Roadmap
 Action Encoder
 Action → Language baseline
 Unseen trajectory evaluation
 MobileCLIP Vision Encoder
 Vision latent + Action latent fusion
 Language Encoder
 
𝑝
(
𝑧
𝑉
∣
𝐿
)
 
𝑝
(
𝑧
𝐴
∣
𝐿
)
 Gaussian baseline
 MDN / multimodal latent distribution
 Simple-task real-world evaluation
 Lightweight deployment evaluation
Design Philosophy

AntVLA is not intended to be:

A large model that understands everything.

It is intended to be:

A small model that understands enough to work.

The central research question is therefore not:

"How much language can the model generate?"

but:

"How much of the visual state and action can language specify, and how should the remaining ambiguity be represented in latent space?"

This motivates language-conditioned distributions over both Vision and Action latents.
