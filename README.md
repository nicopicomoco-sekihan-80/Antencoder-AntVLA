AntVLA v9 — Language Representation Experiments
Overview

AntVLA v9 compares different ways of representing language in a Vision-Language-Action (VLA) model.

The experiments are organized into three main stages:

V2
Language encoder
    ↓
Language is interpreted first
    ↓
What / How representation
    ↓
VLA

V2.5
Static / pretrained language representation
    +
Vision
    ↓
Language and objects are interpreted together
    ↓
Action prediction

V3
Learnable embedding
    +
Vision
    ↓
Language and objects are interpreted together
    ↓
Action prediction


The main research question is:

Should language be substantially interpreted before entering the VLA model, or should language and visual information be interpreted together?

V2 — What / How Language Encoder

V2 uses a dedicated language encoder.

The language is processed into a structured What / How representation before being used by the VLA system.

Language
   ↓
Language Encoder
   ↓
What / How
   ↓
VLA
   ↓
Action


The important characteristic of V2 is that the language representation is already substantially interpreted before the visual information is integrated.

Concept
Language
    ↓
"what object?"
"what action?"
    ↓
What / How representation
    ↓
Vision + representation
    ↓
Action


V2 therefore represents the language-first approach.

V2.5 — Static Vector Experiments

V2.5 investigates an intermediate approach between the language-first V2 architecture and the jointly learned V3 architecture.

Instead of using a dedicated What / How language encoder, V2.5 constructs language embeddings from word vectors.

The key idea is:

Language is represented as vectors, but the representation itself is not necessarily learned as part of the final VLA model.

V2.5 contains three experiments.

V2.5-A — Pretrained Word2Vec

V2.5-A uses a pretrained Word2Vec model.

Each token is converted into its pretrained word vector, and the resulting vector sequence is combined to construct the language embedding.

Language
   ↓
Tokenization
   ↓
Pretrained Word2Vec
   ↓
Vector sequence
   ↓
Embedding
   ↓
Vision + Language
   ↓
Action


The Word2Vec representation is learned independently of the VLA dataset.

This provides a baseline for testing whether general-purpose semantic word representations are useful for VLA learning.

V2.5-B — Word2Vec Trained on the VLA Dataset

V2.5-B trains Word2Vec using the language contained in the VLA dataset.

VLA Dataset Language
        ↓
     Word2Vec
        ↓
Dataset-specific word vectors
        ↓
     Embedding
        ↓
Vision + Language
        ↓
Action


Unlike V2.5-A, the word representation is adapted to the vocabulary and linguistic distribution of the VLA dataset.

However, the Word2Vec training itself does not use the action labels.

This experiment therefore tests whether dataset-specific language statistics alone improve the representation.

V2.5-C — Word2Vec Trained with Action and Language

V2.5-C extends the idea further by learning the representation using both language and action information.

Language ─────┐
              ↓
        Representation
        Learning
              ↑
              │
Action ───────┘
              ↓
       Static Vector
              ↓
       Vision + Language
              ↓
            Action


The purpose is to test whether incorporating task/action information into the language representation produces a more useful embedding for VLA learning.

This is the most task-oriented representation within the V2.5 group.

V3 — Learnable Embedding

V3 uses a learnable embedding layer such as PyTorch nn.Embedding.

Language
   ↓
Token IDs
   ↓
nn.Embedding
   ↓
Language Representation
        +
      Vision
        ↓
Joint Transformer
        ↓
Action


Unlike V2.5, the embedding is learned directly during VLA training.

The language representation can therefore change according to the visual and action prediction objectives.

Core Difference

The experiments can be summarized as follows.

Version	Language representation	Action used to create representation?	Joint interpretation with vision?
V2	What / How language encoder	Indirectly / task-designed	No, language is substantially interpreted first
V2.5-A	Pretrained Word2Vec	No	Yes
V2.5-B	Word2Vec trained on VLA language	No	Yes
V2.5-C	Word2Vec using language + action	Yes	Yes
V3	Learnable nn.Embedding	Through VLA objective	Yes
Interpretation of the Experimental Axis

The experiments are not simply different embedding implementations.

They investigate where semantic interpretation happens.

V2
Language
   ↓
Interpret
   ↓
What / How
   ↓
Vision
   ↓
Action


Language has a relatively complete representation before entering the VLA interaction.

V2.5
Language
   ↓
Vector representation
        +
Vision
   ↓
Joint interpretation
   ↓
Action


Language is represented by static vectors, while its final task meaning emerges through interaction with vision.

V3
Language
   ↓
Learnable embedding
        +
Vision
   ↓
Joint representation learning
   ↓
Action


The language representation itself is learned together with the VLA task.

Research Question

The central comparison is therefore:

Does a VLA system benefit from first constructing a relatively complete language representation, or from interpreting language and visual objects together?

V2 provides the language-first condition.

V2.5 provides several static-vector conditions between language-first encoding and fully learnable VLA embeddings.

V3 provides the jointly learned embedding condition.

This creates a controlled progression:

V2
Language-first interpretation
        ↓
V2.5-A
General pretrained semantic vectors
        ↓
V2.5-B
VLA-dataset-specific semantic vectors
        ↓
V2.5-C
Action-aware semantic vectors
        ↓
V3
Jointly learned language embedding


The goal is to determine how much task-specific information should be present in the language representation before language and vision are jointly interpreted.
