# AntVLA

## Overview

AntVLA is a vision-language-action representation learning framework
that decomposes manipulation knowledge into:

- **What**: object / semantic representation
- **How**: action / manipulation representation
- **State**: physical and spatial state

The framework uses a Teacher-Student architecture to align
world-side representations with language-side representations.

## Architecture

```mermaid
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

        LangInput["Language Target<br>\"pick up the banana\""]:::language

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

‘’’
