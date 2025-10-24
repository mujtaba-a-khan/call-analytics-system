# Domain-Driven Design (DDD)

## Overview

In this project, DDD principles are applied to model the core domain of call processing, analysis, and insights generation. This ensures that the system is aligned with business needs, such as transcribing calls, labeling them intelligently, performing semantic searches, and enabling natural language queries.

## Bounded Contexts

### Ingestion & Normalisation
- The upload workflow instantiates processors directly inside the UI (`src/ui/pages/upload.py:84`).
- CSV parsing, validation, and mapping live in `CSVProcessor` (`src/core/csv_processor.py:32`).
- Audio pre-processing for speech-to-text is handled by `AudioProcessor` (`src/core/audio_processor.py:29`).

### Speech-to-Text & Labeling
- Whisper integration returns a `TranscriptionResult` value object (`src/ml/whisper_stt.py:26`).
- Rule-based enrichment happens inside `LabelingEngine` (`src/core/labeling_engine.py:24`).

### Persistence & Indexing
- `StorageManager` owns file-system persistence for tabular data (`src/core/storage_manager.py:19`).
- The vector store bridge runs through `DocumentIndexer` in the Chroma client layer (`src/vectordb/indexer.py:19`).

### Analytics & Experience
- The analysis page composes metrics, semantic search, and visualisations in one Streamlit view (`src/ui/pages/analysis.py:74`).
- Shared UI components are defined under `src/ui/components`, keeping presentation logic close to the pages that invoke them.

## Domain Concept

### Entities
- **CallRecord** is the single domain entity, modelled with Pydantic to validate structure and derived labels (`src/core/data_schema.py:45`). There is no separate aggregate wrapper in code; the `CallDataFrame` helper simply provides tabular utilities (`src/core/data_schema.py:91`).

### Value Objects
- **TranscriptionResult** carries Whisper output in an immutable dataclass (`src/ml/whisper_stt.py:26`).
- **LLMResponse** wraps answers from the local LLM gateway (`src/ml/llm_interface.py:31`).

### Services
- **LabelingEngine** performs rule evaluation and scoring for call categorisation (`src/core/labeling_engine.py:24`).
- **SemanticSearchEngine** orchestrates vector queries and metadata filtering (`src/analysis/semantic_search.py:16`).
- **QueryInterpreter** translates natural language into filter specs (`src/analysis/query_interpreter.py:27`).

### Persistence Abstractions
- **StorageManager** controls persistence, caching, and snapshotting of tabular call data (`src/core/storage_manager.py:19`).
- **DocumentIndexer** pushes enriched records into the vector database (`src/vectordb/indexer.py:19`).

### Workflow Coordination
- The Streamlit pages compose services imperatively, for example the upload flow calling CSV/audio processors before persisting and indexing (`src/ui/pages/upload.py:84`), and the analysis page chaining storage, metrics, and semantic search (`src/ui/pages/analysis.py:101`).

## *(A)* Event-Storming

This view captures how uploads, processing, storage, and querying are chained together when a user operates the app.

```{mermaid}
flowchart TB
    subgraph ACTORS["👥 ACTORS"]
        direction LR
        U["👤 User/Agent"]
        S["🤖 System/Scheduler"]
    end
    
    subgraph COMMANDS["⚡ COMMANDS"]
        direction TB
        C1["📤 Upload<br/>Audio"]
        C1B["📤 Upload<br/>CSV"]
        C2["🎙️ Transcribe<br/>Audio"]
        C3["🏷️ Apply<br/>Labels"]
        C4["💾 Store<br/>Records"]
        C5["📇 Index<br/>Embeddings"]
        C6["🔍 Semantic<br/>Query"]
    end
    
    subgraph EVENTS["🔔 DOMAIN EVENTS"]
        direction TB
        E1(("📁 File<br/>Uploaded"))
        E2(("✅ Transcription<br/>Done"))
        E3(("🏷️ Labels<br/>Applied"))
        E4(("💾 Records<br/>Stored"))
        E5(("📇 Indexed"))
        E6(("🔍 Query<br/>Done"))
    end
    
    subgraph VIEWS["📊 READ MODELS / VIEWS"]
        direction TB
        R1["📋 Call Records<br/>View"]
        R2["📈 Analytics<br/>Dashboard"]
        R3["🔎 Search<br/>Results"]
    end
    
    %% Audio workflow
    U ===>|trigger| C1
    C1 ==>|executes| E1
    E1 ==>|initiates| C2
    C2 ==>|completes| E2
    E2 ==>|triggers| C3
    
    %% CSV workflow
    U ===>|trigger| C1B
    C1B ==>|executes| E1
    E1 ==>|triggers| C3
    
    %% Common path
    C3 ==>|produces| E3
    E3 ==>|updates| C4
    C4 ==>|confirms| E4
    E4 ==>|displays| R1
    E4 ==>|starts| C5
    C5 ==>|generates| E5
    E5 ==>|updates| R3
    E3 ==>|refreshes| R2
    
    %% Query workflow
    U ===>|submits| C6
    C6 ==>|returns| E6
    E6 ==>|shows| R3
    
    %% System automation
    S ===>|schedules| C2
    S ===>|schedules| C5
    
    %% Professional Gray-Blue Theme - Actors
    style U fill:#E8EAF6,stroke:#3F51B5,stroke-width:3px,color:#1A237E,font-size:15px,font-weight:bold
    style S fill:#E8EAF6,stroke:#3F51B5,stroke-width:3px,color:#1A237E,font-size:15px,font-weight:bold
    
    %% Commands - Dark Blue
    style C1 fill:#BBDEFB,stroke:#1976D2,stroke-width:3px,color:#0D47A1,font-size:14px,font-weight:bold
    style C1B fill:#BBDEFB,stroke:#1976D2,stroke-width:3px,color:#0D47A1,font-size:14px,font-weight:bold
    style C2 fill:#BBDEFB,stroke:#1976D2,stroke-width:3px,color:#0D47A1,font-size:14px,font-weight:bold
    style C3 fill:#BBDEFB,stroke:#1976D2,stroke-width:3px,color:#0D47A1,font-size:14px,font-weight:bold
    style C4 fill:#BBDEFB,stroke:#1976D2,stroke-width:3px,color:#0D47A1,font-size:14px,font-weight:bold
    style C5 fill:#BBDEFB,stroke:#1976D2,stroke-width:3px,color:#0D47A1,font-size:14px,font-weight:bold
    style C6 fill:#BBDEFB,stroke:#1976D2,stroke-width:3px,color:#0D47A1,font-size:14px,font-weight:bold
    
    %% Events - Teal
    style E1 fill:#B2DFDB,stroke:#00897B,stroke-width:3px,color:#004D40,font-size:14px,font-weight:bold
    style E2 fill:#B2DFDB,stroke:#00897B,stroke-width:3px,color:#004D40,font-size:14px,font-weight:bold
    style E3 fill:#B2DFDB,stroke:#00897B,stroke-width:3px,color:#004D40,font-size:14px,font-weight:bold
    style E4 fill:#B2DFDB,stroke:#00897B,stroke-width:3px,color:#004D40,font-size:14px,font-weight:bold
    style E5 fill:#B2DFDB,stroke:#00897B,stroke-width:3px,color:#004D40,font-size:14px,font-weight:bold
    style E6 fill:#B2DFDB,stroke:#00897B,stroke-width:3px,color:#004D40,font-size:14px,font-weight:bold
    
    %% Read Models - Gray
    style R1 fill:#CFD8DC,stroke:#546E7A,stroke-width:3px,color:#263238,font-size:14px,font-weight:bold
    style R2 fill:#CFD8DC,stroke:#546E7A,stroke-width:3px,color:#263238,font-size:14px,font-weight:bold
    style R3 fill:#CFD8DC,stroke:#546E7A,stroke-width:3px,color:#263238,font-size:14px,font-weight:bold
    
    %% Subgraphs
    style ACTORS fill:#F5F5F5,stroke:#3F51B5,stroke-width:2px,color:#1A237E,font-size:16px,font-weight:bold
    style COMMANDS fill:#F5F5F5,stroke:#1976D2,stroke-width:2px,color:#0D47A1,font-size:16px,font-weight:bold
    style EVENTS fill:#F5F5F5,stroke:#00897B,stroke-width:2px,color:#004D40,font-size:16px,font-weight:bold
    style VIEWS fill:#F5F5F5,stroke:#546E7A,stroke-width:2px,color:#263238,font-size:16px,font-weight:bold
    
    linkStyle default stroke:#37474F,stroke-width:3px
```

I use this flow to mirror how the Streamlit upload logic in `src/ui/pages/upload.py:84` chains together CSV and audio processing, and how the subsequent indexing happens through `src/vectordb/indexer.py:58` before results are surfaced in `src/ui/pages/analysis.py:101`.

## *(B)* Core Domain Chart

When I did the event storming I spotted five clear domains that already cover the whole product.The core-domain chart below to show how they connect.

- **Call Analytics Core (Core Domain)** – the rules, scoring, and search logic that turns raw calls into insights (`src/core/labeling_engine.py`, `src/analysis/semantic_search.py`, `src/analysis/aggregations.py`).
- **Ingestion & Normalisation (Supporting Domain)** – the upload flow plus CSV and audio processing that prepare the call data (`src/ui/pages/upload.py`, `src/core/csv_processor.py`, `src/core/audio_processor.py`).
- **Knowledge Indexing (Supporting Domain)** – embedding generation and vector storage for fast semantic lookups (`src/vectordb/indexer.py`, `src/ml/embeddings.py`).
- **Analyst Experience (Supporting Domain)** – Streamlit pages and helpers that let an analyst explore and query the calls (`src/ui/pages/analysis.py`, `src/analysis/query_interpreter.py`, `src/ui/components`).
- **Platform Operations (Generic Domain)** – storage, logging, and config plumbing that keep the rest running (`src/core/storage_manager.py`, `src/core/logging.py`).

## *(C)* Core Domain and Relationships 

To make the core-domain chart explicit, the following view groups the discovered domains and highlights the mappings between them.

```{mermaid}
flowchart LR
    subgraph CORE["🎯 Core Domain"]
        direction TB
        CAC["Call Analytics Core"]
    end

    subgraph SUPPORTING["⚙️ Supporting Domains"]
        direction TB
        IN["Ingestion & Normalisation"]
        KI["Knowledge Indexing"]
        AX["Analyst Experience"]
    end

    subgraph GENERIC["🔧 Generic Domain"]
        direction TB
        PO["Platform Operations"]
    end

    IN --> M1["Normalised call data"]
    M1 --> CAC
    KI --> M2["Semantic vectors"]
    M2 --> CAC
    CAC --> M3["Insights and intents"]
    M3 --> AX
    AX --> M4["Analyst feedback"]
    M4 --> CAC

    PO --> M5["Storage and logging"]
    M5 --> IN
    PO --> M6["Vector infrastructure"]
    M6 --> KI
    PO --> M7["Cached datasets"]
    M7 --> AX
```

Ingestion hands normalised call data into the call analytics core, indexing pushes semantic vectors there, the analytics page surfaces insights and loops analyst feedback back in, and platform operations keeps storage, logging, and vector infrastructure alive for every other domain.

### Relationships Between Data and Services

The entity-relationship diagram shows how the central Pydantic model connects to downstream services and persistence layers.

```{mermaid}
erDiagram
    CALL_RECORD ||--o{ TRANSCRIPTION_RESULT : has
    CALL_RECORD ||--o{ EMBEDDING : generates
    CALL_RECORD ||--o{ LABELING_RESULT : produces
    
    CALL_RECORD {
        string call_id 
        datetime start_time
        float duration_seconds
        string transcript
        string agent_id
        string call_type
        string outcome
    }
    
    TRANSCRIPTION_RESULT {
        string transcript
        float duration_seconds
        string language
        float confidence
    }
    
    LABELING_RESULT {
        string call_type
        string outcome
        string connection_status
    }
    
    EMBEDDING {
        string doc_id
        list vector
    }
    
    STORAGE_MANAGER ||--o{ CALL_RECORD : manages
    INDEXER ||--o{ EMBEDDING : stores
    RETRIEVER ||--o{ EMBEDDING : queries
```

I rely on this diagram when reasoning about how `CallRecord` is persisted in `src/core/storage_manager.py:110`, how transcripts become embeddings inside `src/vectordb/indexer.py:128`, and how semantic lookups consume both through `src/analysis/semantic_search.py:42`.

**Implementation Notes**
- `agent_id`, `campaign`, and other optional fields are part of the `CallRecord` schema rather than standalone entities (`src/core/data_schema.py:45`).
- Labels for connection status, call type, and outcome are derived by the `LabelingEngine` workflow (`src/core/labeling_engine.py:100`).
- Embedding metadata is stored alongside vectors in the Chroma-backed repositories (`src/vectordb/indexer.py:141`).
