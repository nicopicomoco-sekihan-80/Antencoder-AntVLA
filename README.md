AntVLA
Overview
AntVLA is a vision-language-action representation learning framework
for robot manipulation.
The core idea is to decompose manipulation knowledge into three
complementary components:

What — object and semantic representation
How — action and manipulation representation
State — physical and spatial state
AntVLA adopts a Teacher-Student architecture to bridge
world-side multimodal representations and language-side representations.
The Teacher learns structured representations from vision and robot
action data, while the Student learns to recover the corresponding
representations directly from language.

Architecture
graph TD

    %% =========================================================
    %% Style
    %% =========================================================
    classDef input fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;
    classDef encoder fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C;
    classDef latent fill:#E8F5E9,stroke:#388E3C,stroke-width:2px,color:#1B5E20;
    classDef language fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#E65100;
    classDef loss fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#B71C1C;
    classDef decoder fill:#ECEFF1,stroke:#455A64,stroke-width:2px,color:#263238;
    classDef teacher fill:#E1F5FE,stroke:#0288D1,stroke-width:2px;
    classDef student fill:#FFF8E1,stroke:#F9A825,stroke-width:2px;

    %% =========================================================
    %% Teacher: World -> Latent
    %% =========================================================
    subgraph Teacher["Teacher: World-Side Representation Learning"]
        direction TB

        Vision["Vision<br>Image / Video"]:::input
        Action["Action<br>Robot Trajectory"]:::input

        Vision --> StateEnc["State Encoder"]:::encoder
        Vision --> ObjEnc["Object Encoder"]:::encoder
        Action --> ActEnc["Action Encoder"]:::encoder

        StateEnc --> State["State<br>Physical / Spatial State"]:::latent
        ObjEnc --> yV["y_V<br>Object / What"]:::latent
        ActEnc --> zT["z_T<br>Action / How"]:::latent

        State --> Decoder
        yV --> Decoder
        zT --> Decoder

        Decoder["Multimodal Decoder"]:::decoder
        Decoder --> LangRec["Generated Language"]:::language
    end

    %% =========================================================
    %% Student: Language -> Latent
    %% =========================================================
    subgraph Student["Student: Language-Side Distillation"]
        direction TB

        LangInput["Language Target<br>pick up the banana"]:::language

        LangInput --> LangEnc["Language Encoder"]:::encoder

        LangEnc --> yL["y_L<br>Object / What"]:::latent
        LangEnc --> zL["z_L<br>Action / How"]:::latent
    end

    %% =========================================================
    %% Alignment / Grounding
    %% =========================================================
    subgraph Grounding["Cross-Modal Grounding & Distillation"]
        direction LR

        ObjAlign["Object Alignment<br>InfoNCE / Contrastive Loss"]:::loss
        ActAlign["Action Distillation<br>MSE / Cosine Loss"]:::loss
        LangAlign["Language Reconstruction<br>Cross-Entropy Loss"]:::loss
    end

    %% What alignment
    yV -.->|"What"| ObjAlign
    yL -.->|"What"| ObjAlign

    %% How alignment
    zT -.->|"Teacher"| ActAlign
    zL -.->|"Student"| ActAlign

    %% Language reconstruction
    LangRec --> LangAlign
    LangInput --> LangAlign

    %% =========================================================
    %% Conceptual labels
    %% =========================================================
    What["WHAT<br>Object / Semantic Identity"]:::student
    How["HOW<br>Action / Manipulation"]:::student
    World["WORLD<br>Physical / Spatial State"]:::teacher

    yV -.-> What
    yL -.-> What

    zT -.-> How
    zL -.-> How

    State -.-> World
Representation
Teacher: World Side
The Teacher receives multimodal robot data:
Vision — image or video observations
Action — robot manipulation trajectories
These inputs are transformed into structured latent representations:
State — physical and spatial state of the environment
y_V — visual and semantic representation of the manipulated object
z_T — action and manipulation representation
The three representations are combined by a multimodal decoder
to reconstruct the corresponding language description.
Student: Language Side
The Student receives a language instruction such as:
"pick up the banana"
The Language Encoder decomposes the instruction into two
complementary representations:
y_L — What: object / semantic identity
z_L — How: action / manipulation
The Student therefore learns to infer structured manipulation
representations from language.
Cross-Modal Alignment
AntVLA uses three complementary learning objectives.
1. Object Alignment
The visual object representation y_V and language object
representation y_L are aligned using contrastive learning.
y_V  <---- Contrastive Learning ---->  y_L
              InfoNCE Loss
This encourages both modalities to share a common semantic space
for object identity and meaning.
2. Action Distillation
The language-side action representation z_L is distilled from
the Teacher representation z_T.
z_T  ----------------------------->  z_L
          Distillation Loss
           MSE / Cosine
The objective is to make the Student recover the manipulation
knowledge encoded by the Teacher.
3. Language Reconstruction
The Teacher's multimodal representation is used to reconstruct
the target language.
State + y_V + z_T
        │
        ▼
Multimodal Decoder
        │
        ▼
Generated Language
        │
        ▼
Language Reconstruction Loss
This provides a language-level learning signal that connects
the latent representations to their natural-language meaning.
Overall Objective
The training objective can be expressed as:
L
=
λ
o
b
j
L
o
b
j
+
λ
a
c
t
L
a
c
t
+
λ
l
a
n
g
L
l
a
n
g

where:

$\mathcal{L}_{\mathrm{obj}}$ is the object contrastive loss
$\mathcal{L}_{\mathrm{act}}$ is the action distillation loss
$\mathcal{L}_{\mathrm{lang}}$ is the language reconstruction loss
$\lambda_{\mathrm{obj}}, \lambda_{\mathrm{act}}, \lambda_{\mathrm{lang}}$
control the contribution of each objective
Training
During training, the Teacher learns structured representations
from the physical world, while the Student learns to recover
the corresponding representations from language.
             WORLD
               │
       ┌───────┴───────┐
       ▼               ▼
    Vision           Action
       │               │
       └───────┬───────┘
               ▼
           TEACHER
               │
        ┌──────┼──────┐
        ▼      ▼      ▼
      State    y_V    z_T
               │      │
               │      │
               │      └──────┐
               │             │
               ▼             ▼
           WHAT            HOW
               ▲             ▲
               │             │
               │             │
           y_L ◄──────────── z_L
               ▲             ▲
               └──────┬──────┘
                      │
                   STUDENT
                      ▲
                      │
                  Language
Inference
At inference time, the Student can encode a natural-language
instruction into structured manipulation representations:
Language Instruction
        │
        ▼
Language Encoder
        │
   ┌────┴────┐
   ▼         ▼
  y_L       z_L
 WHAT       HOW
   │         │
   └────┬────┘
        ▼
   VLA Controller
        │
        ▼
   Robot Action
This provides a pathway from language to structured representations
that can be used for vision-language-action control.
Repository Structure
Antencoder-AntVLA/
├── antencoder_v3/
├── README.md
└── ...
Project Status
This project is under active development.
Current development focuses on:

structured What / How / State representation learning
cross-modal alignment between vision and language
action representation distillation
integration with vision-language-action control

