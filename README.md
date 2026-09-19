Antencoder-AntVLA v8
Overview

This project investigates how vision-language representations should be formed for vision-language-action (VLA) models.

The current research started from a simple question:

How can a VLA model extract only the information from language that is actually relevant to the observed scene and the desired action?

The research first explored an explicit What / How decomposition of language, followed by a new hypothesis:

Perhaps the problem is not only that language representations contain too much information.
Perhaps the deeper problem is trying to interpret language independently of vision in the first place.

Based on this observation, v8 explores two related approaches:

Model 2: explicitly factorize language into task-relevant components.

Model 3-Lite: interpret language and vision jointly, allowing visual information to participate in the formation of the language representation itself.

The goal is not simply to make the model larger, but to investigate where and when language should acquire its task-relevant meaning.

Research Direction
Model 2 — Explicit Language Factorization

The initial approach uses a pretrained language representation such as BERT.

A raw instruction is first encoded:

Language
   ↓
BERT
   ↓
Language latent
   ↓
Task-relevant information
   ↓
Vision / Action


The motivation is that a general language model contains considerably more information than is necessary for action generation.

For example:

"pick the banana"


contains linguistic information that may be irrelevant to the robot's immediate action.

Therefore, Model 2 introduces an explicit What / How decomposition:

             Language
                 ↓
               BERT
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
      What               How
    (object)           (action)
        ↓                 ↓
        └────────┬────────┘
                 ↓
              Action


The hypothesis is that explicitly extracting task-relevant information can reduce unnecessary information in the language representation.

A New Question

Model 2 raises a deeper question.

Even if unnecessary information can be removed from the language representation, should language be interpreted independently before vision is considered?

For example:

"pick the banana"


can be represented linguistically before looking at the image.

But the actual meaning required for action depends on the current visual context.

The banana referred to by the instruction is not an abstract banana. It is the banana that exists—or needs to be searched for—in the current environment.

This leads to the next hypothesis:

Instead of first constructing a complete language representation and then combining it with vision, language may need to be interpreted while attending to the visual scene.

Model 3-Lite — Vision-Conditioned Language Interpretation

Model 3-Lite removes the explicit What / How decomposition and does not use an independent language encoder whose representation is completed before visual interaction.

Instead, raw language tokens and visual tokens are introduced into a joint attention mechanism.

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

Language is not fully interpreted before seeing the visual scene. Its task-relevant representation is formed through interaction with vision.

This is different from simply concatenating two already-computed embeddings.

The intended interaction is:

Language ↔ Vision


rather than:

Language → Language Representation
                         ↓
                    Vision Fusion

Why Model 3-Lite?

Consider the instruction:

"pick the banana"


A conventional language-first approach may construct an abstract representation of:

banana + pick


before examining the image.

Model 3-Lite asks whether this ordering is actually appropriate.

Instead:

"pick the banana"
        ↕
   visual scene
        ↓
task-relevant interpretation
        ↓
      action


The model can use the visual scene to determine which aspects of the language are relevant to the current action.

This is particularly important for grounding.

A Counterargument: What If the Object Is Not Visible?

A natural objection to Model 3-Lite is:

What happens if the instructed object is not currently visible?

For example:

Instruction:
"pick the banana"

Image:
apple + cube + table


The language still provides information about what should be searched for, even though the object is not currently visible.

Therefore, Model 3-Lite does not assume that vision completely determines language.

Instead, the intended relationship is:

Language
   ↓
what should be searched / acted upon
   ↕
Vision
   ↓
what is currently observable


The hypothesis is that these two sources of information should interact during interpretation rather than forcing language to produce a complete task representation independently of vision.

The Central Research Question

The comparison between Model 2 and Model 3-Lite therefore becomes:

Model 2

First understand and compress language, then combine it with vision.

Model 3-Lite

Interpret language while looking at vision.

The research question is:

Can a VLA model learn more task-relevant and compositional representations when language is interpreted jointly with visual context, rather than being fully encoded independently beforehand?

Experimental Setup

The current v8 experiment uses Bridge data.

A small initial dataset of approximately 560 samples is prepared for the first Model 3-Lite experiments.

The purpose of this initial experiment is not large-scale performance optimization.

Instead, the first goals are:

Verify that Model 3-Lite can fit the training data.

Verify that vision-language interaction is functioning as intended.

Test generalization across combinations of actions and objects.

Compare the resulting behavior with the explicit factorization used in Model 2.

Action Representation

The existing action autoencoder is reused:

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
Action decoder
       ↓
Action trajectory


This allows the experiment to focus on the vision-language representation problem rather than simultaneously learning an entirely new action representation.

Experimental Hypothesis

The experiments are designed around two competing hypotheses.

Model 2 hypothesis

Explicitly removing irrelevant language information and separating:

What + How


will produce a more useful representation for action generation.

Model 3-Lite hypothesis

The deeper issue is not merely excessive information in language representations.

Instead, the problem may be the assumption that language should be semantically completed before visual grounding.

Model 3-Lite therefore tests whether:

Vision ↔ Language


joint interaction can produce a more action-relevant representation naturally.

Status
Model 2

What / How decomposition: implemented as the current research foundation

BERT-based language representation: part of the existing approach

Action representation: action autoencoder available

Model 3-Lite

Research hypothesis: defined

Initial Bridge dataset: prepared

Model architecture: next implementation target

Initial experiment: pending

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

Long-Term Direction

The objective is not simply to compare two network architectures.

The broader question is:

Where should the meaning of language be formed in a vision-language-action system?

Possible answers explored in this project are:

Model 2:
Language
   ↓
Semantic / task representation
   ↓
Vision
   ↓
Action


versus:

Model 3-Lite:
Language
    ↕
  Vision
    ↓
Task-relevant representation
    ↓
Action


Model 3-Lite therefore represents a shift from language representation compression toward vision-conditioned language interpretation.
