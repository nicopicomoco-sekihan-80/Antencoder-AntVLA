# Antencoder-AntVLA
## Small VLA for Real-World Manipulation

AntVLA is a research project toward a small, deployable Vision-Language-Action (VLA) model for real-world manipulation.

The goal is not to build a large model capable of understanding arbitrarily complicated instructions.

Instead, AntVLA aims to:

> Build a small model that can reliably understand and execute simple real-world tasks.

The central idea is to represent **What** and **How** as continuous latent variables and use Language as a compact semantic interface between them.

A longer-term extension is to introduce an intermediate **Action Planner** between slow language-level reasoning and fast low-level control. This planner would decompose complex tasks into reusable manipulation primitives rather than requiring the VLA policy or language model to learn every multi-step task directly.

---

## Motivation

A robot trajectory contains information about **how** an action was performed.

Visual observations contain information about **what exists** and the current physical state.

Language provides semantic information about the task, but it does not uniquely specify either the visual state or the exact trajectory.

For example:

> "pick the cube"

may correspond to:

- red cube
- blue cube
- green cube

Likewise, the same instruction may be executed through many valid trajectories:

- approach from the left
- approach from the right
- approach from above

Therefore, forcing Language to predict a single Vision or Action latent with a point-regression objective can create an artificial average representation.

AntVLA instead investigates conditional distributions over latent representations.

---

## Core Idea

AntVLA separates physical information into two complementary latent spaces.

### Vision

```text
RGB / Observation
       ↓
Vision Encoder
       ↓
      z_V
       ↓
   What / State
