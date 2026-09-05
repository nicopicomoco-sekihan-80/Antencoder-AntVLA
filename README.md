# AntVLA

## Overview

**AntVLA** is a vision-language-action representation learning framework
for robot manipulation.

The core idea is to decompose manipulation knowledge into three
complementary components:

- **What** — object and semantic representation
- **How** — action and manipulation representation
- **State** — physical and spatial state

AntVLA adopts a **Teacher-Student architecture** to bridge
world-side multimodal representations and language-side representations.

The Teacher learns structured representations from vision and robot
action data, while the Student learns to recover the corresponding
representations directly from language.

---

## Architecture

Here is the block diagram of the AntVLA framework architecture:

```mermaid
graph LR

    Vision[Vision]
    Action[Robot Action]
    Teacher[Teacher]

    State[State]
    WhatV["y_V: What"]
    HowT["z_T: How"]

    Language[Language]
    Student[Student]

    WhatL["y_L: What"]
    HowL["z_L: How"]

    Decoder[Multimodal Decoder]
    Output[Generated Language]

    Vision --> Teacher
    Action --> Teacher

    Teacher --> State
    Teacher --> WhatV
    Teacher --> HowT

    Language --> Student

    Student --> WhatL
    Student --> HowL

    WhatV -.->|Contrastive Loss| WhatL
    HowT -.->|Distillation Loss| HowL

    State --> Decoder
    WhatV --> Decoder
    HowT --> Decoder

    Decoder --> Output
    Language -.->|Reconstruction Loss| Output
```

