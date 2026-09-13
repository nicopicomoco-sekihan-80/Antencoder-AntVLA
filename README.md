Antencoder-AntVLA
Small VLA for Real-World Manipulation

AntVLA is a research project toward a small, deployable Vision-Language-Action model for real-world manipulation.

The goal is not to build a large model that understands arbitrarily complicated instructions.

Instead:

Build a small model that can reliably understand and act on simple real-world tasks.

The central idea is to represent What and How as continuous latent variables and use Language as a compact semantic interface between them.

1. Motivation

A robot trajectory contains information about how an action was performed.

Visual observations contain information about what exists and the current state.

Language provides semantic information about the task, but it does not uniquely specify everything.

For example:

"pick the cube"


may correspond to:

red cube
blue cube
green cube


and the same instruction may be executed through many valid trajectories:

approach from left
approach from right
approach from above


Therefore, forcing Language to predict a single Vision or Action latent with MSE can create an artificial average representation.

2. Core Idea

AntVLA represents:

Vision
  ↓
Vision Encoder
  ↓
z_V
  ↓
What / State


and:

Action trajectory
  ↓
Action Encoder
  ↓
z_A
  ↓
How


The long-term goal is to learn a Language representation from which these latent variables can be recovered as conditional distributions.

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

where:

𝑧
𝑉
: Vision latent
𝑧
𝐴
: Action latent
𝐿
: Language
𝑝
(
𝑧
𝑉
∣
𝐿
)
: language-conditioned Vision latent distribution
𝑝
(
𝑧
𝐴
∣
𝐿
)
: language-conditioned Action latent distribution

Language is therefore not the final objective.

It is a compact semantic representation of the information that connects vision and action.

3. Why Distributions?

A point regression objective assumes:

𝐿
→
𝑧
^

For example:

"pick cube"
        ↓
      MSE
        ↓
one Vision latent


But multiple Vision states may be valid.

Likewise:

"pick cube"
        ↓
      MSE
        ↓
one Action latent


may average several valid approaches.

This can produce a latent that corresponds to no actual valid state or trajectory.

AntVLA instead models:

𝐿
→
𝑝
(
𝑧
∣
𝐿
)

so multiple valid solutions can coexist.

4. Vision and Action

The same principle is applied to both modalities.

                 Language
                     │
              Language Encoder
                     │
                Language latent
                     │
           ┌─────────┴─────────┐
           ↓                   ↓
      p(z_V | L)          p(z_A | L)
           ↓                   ↓
     Vision latent         Action latent
        What                  How


This gives a unified treatment of ambiguity.

Vision ambiguity
"pick cube"

→ red cube
→ blue cube
→ green cube

Action ambiguity
"pick the cube"

→ approach from left
→ approach from right
→ approach from above


The model should represent these alternatives rather than averaging them.

5. Vision Encoder

The current plan uses MobileCLIP as the Vision Encoder.

RGB / Observation
       ↓
   MobileCLIP
       ↓
   Vision latent
       ↓
      z_V


MobileCLIP is chosen because AntVLA prioritizes:

compact models
efficient inference
real-world deployment
useful semantic visual representations

The objective is not to maximize model scale.

6. Action Encoder / Action Teacher

The first development target is the Action Teacher.

Action trajectory
       ↓
 Action Encoder
       ↓
      z_A
       ↓
 Action Teacher
       ↓
 action semantics


The Action Teacher establishes a meaningful Action latent space before Language is asked to recover it.

The initial goal is for:

same action primitive
      ↓
similar latent

different action primitives
      ↓
separated latent


For example:

Pick
Stack
Push
Place


should form distinguishable semantic regions while trajectories within the same primitive remain relatively close.

7. Language Encoder

Once the Action latent is established, Language is introduced.

Language
   ↓
Language Encoder
   ↓
z_L
   ├────────→ p(z_A | L)
   │
   └────────→ p(z_V | L)


The Language Encoder should not simply predict a single latent.

Instead, it should represent the uncertainty that remains because Language does not fully specify the underlying visual state or action realization.

8. Mixture Density Model

A simple Gaussian can be used as a baseline:

𝑝
𝜃
(
𝑧
∣
𝐿
)
=
𝑁
(
𝑧
;
𝜇
𝐿
,
Σ
𝐿
)

However, a single Gaussian cannot naturally represent strongly multimodal solutions.

Therefore, the main candidate is a Mixture Density Network:

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

The training objective is negative log-likelihood:

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

This allows the model to represent multiple valid latent modes.

9. VLA Architecture

The long-term architecture is:

                         RGB
                          │
                     MobileCLIP
                          │
                       z_V
                    What / State
                          │
                          │
                          ├──────────────┐
                          │              │
                          │            Fusion
                          │              ↑
                          │              │
Action trajectory ──→ Action Encoder ─→ z_A
                                         How
                                          │
                                          ↓
                                       Policy
                                          │
                                          ↓
                                     Robot Action


Language
   │
   ↓
Language Encoder
   │
   ↓
  z_L
   ├────────→ p(z_V | L)
   │
   └────────→ p(z_A | L)


The Policy remains the ultimate downstream objective.

Language generation itself is not the primary goal.

10. Slow and Fast Intelligence

A longer-term goal is to use the latent space as an interface between slow reasoning and fast control.

                Slow Reasoning
                      │
             latent intervention
                      │
             ┌────────┴────────┐
             ↓                 ↓
       Vision latent       Action latent
          What                 How
             │                 │
             └────────┬────────┘
                      ↓
                  Fast Policy
                      ↓
                    Robot


For example, a slow reasoning module may determine:

"grasp this object"


without directly controlling the robot at every timestep.

Instead, it can intervene in the relevant latent representation:

Slow reasoning
      ↓
modify / select latent
      ↓
Fast policy
      ↓
robot action


This enables a possible separation between:

slow reasoning / planning
fast perception / control

while keeping the interface compact.

11. Data Philosophy

AntVLA does not aim to solve increasingly complicated instructions.

The target is:

simple tasks
×
diverse objects
×
diverse environments
×
diverse trajectories

Candidate primitive tasks include:

Pick
Place
Push
Pull
Open
Close
Insert
Stack


The goal is to make a small model robust to real-world variation, rather than making a large model capable of interpreting arbitrary instructions.

12. Simulation First, Real Data Later

ManiSkill is currently used as a controlled research environment.

Its role is:

controlled laboratory for learning and analyzing the latent structure.

It provides:

reproducible trajectories
clean action signals
explicit task labels
large numbers of trajectories
controlled variation

However, ManiSkill is not assumed to be the final target domain.

The eventual goal is validation on real-world manipulation data.

ManiSkill
   ↓
latent / teacher validation
   ↓
real-world data
   ↓
generalization
   ↓
deployment

13. Current Development
v6-alpha

Current Action → Language baseline:

PickCube
100 / 100 = 100.0%

StackCube
100 / 100 = 100.0%

Overall
200 / 200 = 100.0%


Unseen trajectory evaluation:

PickCube
100 / 100 = 100.0%

StackCube
100 / 100 = 100.0%

Overall
200 / 200 = 100.0%


These experiments establish the initial Action Encoder baseline.

14. Current Priority

The immediate goal is not to build the complete VLA.

The development order is:

1. Action Encoder
        ↓
2. Action Teacher
        ↓
3. Analyze Action latent
        ↓
4. Language Encoder
        ↓
5. p(z_A | L)
        ↓
6. MobileCLIP Vision Encoder
        ↓
7. p(z_V | L)
        ↓
8. Vision + Action fusion
        ↓
9. VLA Policy
        ↓
10. Real-world evaluation


The first milestone is therefore:

Build an Action latent space that represents manipulation semantics rather than merely memorizing task labels.

15. Research Hypothesis

The central hypothesis is:

Language does not uniquely determine visual state or action realization. Therefore, predicting a conditional distribution over Vision and Action latents is more appropriate than forcing a single point estimate.

A second hypothesis is:

As language specifies more information, the corresponding conditional latent distribution should become more concentrated.

For example:

"pick cube"
        ↓
high uncertainty

"pick red cube"
        ↓
lower visual uncertainty

"pick red cube on the left"
        ↓
lower uncertainty about visual state


The same principle applies to Action.

16. Design Philosophy

AntVLA is not:

A large model that understands everything.

AntVLA aims to be:

A small model that understands enough to work.

The research focus is therefore not language generation.

The core question is:

How can a compact latent space represent What and How, preserve the ambiguity that language leaves unspecified, and provide an interface between reasoning and fast robot control?

Roadmap
 Action Encoder
 Action → Language baseline
 Unseen trajectory evaluation
 Action Teacher
 Action latent analysis
 Language Encoder
 
𝑝
(
𝑧
𝐴
∣
𝐿
)
 MobileCLIP Vision Encoder
 
𝑝
(
𝑧
𝑉
∣
𝐿
)
 Gaussian baseline
 MDN / multimodal latent distribution
 Vision + Action fusion
 VLA Policy
 Real-world evaluation
 Slow reasoning → latent intervention
 Lightweight deployment
