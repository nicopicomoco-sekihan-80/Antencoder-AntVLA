Antencoder-AntVLA v8
Overview

This project investigates how vision-language representations should be formed for vision-language-action (VLA) models.

The research began with an explicit What / How factorization of language and has now led to a new hypothesis:

Perhaps the deeper problem is not only how much information language contains, but the assumption that language should first be converted into an independent semantic representation before being grounded in vision.

v8 therefore explores two related approaches:

Model 2: explicitly factorize language into What and How, then construct task-relevant representations.

Model 3-Lite: do not explicitly factorize language and do not first construct an independent language representation. Instead, language and vision interact directly through joint attention.

The longer-term goal is to extend this representation into a latent-space planner, where a world model can perform intermediate reasoning in latent space rather than requiring natural language to serve as the internal reasoning medium.

Model 2 — Explicit What / How Factorization

Model 2 starts from the observation that a raw language representation can contain information that is irrelevant to the current action.

Instead of passing the entire instruction directly to the action model, the language is explicitly decomposed into:

Language
   │
   ├── What → object / target
   │
   └── How  → action / operation


For example:

"pick the banana"

What → banana
How  → pick


The purpose is to provide the model with a representation focused on information relevant to vision and action.

The important point is that Model 2 does not use BERT.

The language-side representations are designed specifically around the What / How decomposition rather than relying on a general-purpose pretrained language encoder.

A Question Raised by Model 2

Model 2 addresses the problem of unnecessary language information by explicitly extracting the components considered relevant to action.

However, this leads to a deeper question:

Why should language be interpreted independently of the visual scene in the first place?

Consider:

"pick the banana"


Before seeing the image, the model can construct an abstract representation of:

banana + pick


But the actual action depends on the current visual context.

The relevant banana is not an abstract object. It is an object in a particular scene.

This motivates Model 3-Lite.

Model 3-Lite — Vision-Conditioned Language Interpretation

Model 3-Lite removes the explicit What / How decomposition.

More importantly, it avoids first converting the entire language instruction into a completed independent semantic representation.

Instead, raw language tokens and visual tokens interact directly.

Image
  ↓
Vision Encoder
  ↓
Vision Tokens
       ↘
         Joint Attention
       ↗
Language Tokens
  ↓
Token Embedding

       ↓
Multimodal Representation
       ↓
Action Prediction


The key idea is:

Language should be interpreted while attending to the visual scene.

The intended interaction is therefore:

Language ↔ Vision


rather than:

Language
   ↓
Independent semantic representation
   ↓
Vision fusion

Why Model 3-Lite?

The difference can be illustrated with:

"pick the banana"


Model 2 explicitly extracts:

What = banana
How  = pick


and constructs task-relevant representations from them.

Model 3-Lite instead asks whether the meaning required for action can emerge through interaction:

"pick the banana"
        ↕
   visual scene
        ↓
task-relevant interpretation
        ↓
      action


The hypothesis is that the visual context should participate during the formation of the language representation rather than only after it has already been constructed.

A Counterargument

A natural objection is:

What happens if the instructed object is not visible?

For example:

Instruction:
"pick the banana"

Image:
apple + cube + table


The language still provides information about what the system should search for.

Therefore, Model 3-Lite does not assume that vision completely determines language.

Instead:

Language
   ↓
what should be searched / acted upon
   ↕
Vision
   ↓
what is currently observable


The hypothesis is that these two sources of information should interact during interpretation.

The Core Difference

The conceptual difference between the two approaches is therefore:

Model 2

Explicit factorization

Language
   ↓
What / How
   ↓
task-relevant representation
   ↓
Vision / Action

Model 3-Lite

Joint interpretation

Language ↔ Vision
        ↓
task-relevant representation
        ↓
Action


Model 2 asks:

Can explicitly removing irrelevant language information produce a better action representation?

Model 3-Lite asks:

Is it necessary to construct an independent language representation before visual grounding at all?

Experimental Setup

The current v8 experiment uses Bridge data.

An initial dataset of approximately 560 samples has been prepared for the first Model 3-Lite experiment.

The initial goals are:

Verify that Model 3-Lite can fit the training data.

Verify that vision-language joint attention is functioning as intended.

Test generalization across combinations of actions and objects.

Compare the resulting behavior with the explicit What / How approach of Model 2.

Action Representation

The existing action autoencoder is reused.

Action trajectory
        ↓
Action Autoencoder
        ↓
   Action latent


Model 3-Lite predicts the corresponding action representation:

Image + Language
       ↓
 Model 3-Lite
       ↓
 Action latent
       ↓
Action Decoder
       ↓
Action trajectory


This allows the initial experiment to focus on the vision-language representation problem.

Future Direction — Latent-Space Planner

Model 3-Lite is not intended to be the final architecture.

A longer-term goal is to introduce a Planner that uses a world model and performs intermediate reasoning in latent space.

The motivation is that natural language does not necessarily need to be the medium of internal reasoning.

Instead of:

Language
   ↓
Language reasoning
   ↓
Action


the future architecture may use:

Language
   ↓
Grounding
   ↓
Latent state
   ↓
        ┌──────────────┐
        │    Planner   │
        │              │
        │ latent       │
        │ reasoning    │
        └──────┬───────┘
               ↓
         future latent
               ↓
          Action


The Planner would use a world model to predict and evaluate future latent states.

In this formulation, language serves primarily as a means of specifying the task or goal, while the internal reasoning process takes place in a compact latent representation.

This is an important motivation for Model 3-Lite.

If vision and language can first be grounded into a shared task-relevant latent space, that latent space can potentially become the medium for intermediate planning.

The long-term architecture is therefore envisioned as:

Language
    +
  Vision
    ↓
Grounded Representation
    ↓
Latent State
    ↓
World Model + Planner
    ↓
Predicted Future Latents
    ↓
Action Decoder
    ↓
Robot Action


The intended concept is latent-space intermediate reasoning rather than requiring the model to repeatedly convert its internal state into natural language during planning.

Research Roadmap

The current research can therefore be viewed as a progression:

Model 2
Explicit What / How
        ↓
Remove unnecessary language information
        ↓
Question the language-first assumption
        ↓
Model 3-Lite
Vision ↔ Language joint interpretation
        ↓
Grounded latent representation
        ↓
World Model
        ↓
Planner
        ↓
Latent-space intermediate reasoning
        ↓
Action


Each stage addresses a different question:

Model 2: Can explicit factorization make language representations more useful for action?

Model 3-Lite: Can language be interpreted more effectively through direct interaction with vision?

Future Planner: Can the resulting grounded latent representation serve as an internal medium for world-model-based planning?

Status
Model 2

What / How decomposition: research foundation

BERT: not used

Language representation: task-oriented What / How representation

Action autoencoder: available

Model 3-Lite

Research hypothesis: defined

Raw language + visual interaction: planned

Initial Bridge dataset: prepared

Model architecture: next implementation target

Initial experiment: pending

Future Planner

World-model-based planner: future direction

Latent-space intermediate reasoning: future direction

Natural-language-free internal reasoning: future direction

Current v8 Structure
v8/
│
├── action_autoencoder.pt
├── train.py
├── requirements.txt
│
├── bridge_data_v2/
│
├── data/
│   └── bridge_tfds/
│
├── checkpoints/
│
└── src/
    ├── dataset.py
    └── model.py


The current implementation is being extended from the Model 2 foundation toward Model 3-Lite.

Research Direction

The broader question is:

Where should the meaning of language be formed in a vision-language-action system, and what representation should the system use for internal reasoning?

Model 2 explores explicit language factorization.

Model 3-Lite explores vision-conditioned language interpretation.

The future Planner explores latent-space reasoning with a world model.

The overall direction is therefore a progression from language-centric representations toward grounded latent representations that can support perception, action, and eventually internal planning within a shared latent space.
