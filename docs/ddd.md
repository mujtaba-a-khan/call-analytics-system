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

## Visuals

### Event-Storming

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

### Core Domain, Supporting Capabilities, and Infrastructure

This chart highlights which features embody business-specific behaviour versus reusable plumbing.

```{mermaid}
flowchart TB
    subgraph CORE["🎯 CORE DOMAIN - Unique Business Logic"]
        direction TB
        C1["📋 Call Labeling<br/>Engine"]
        C2["🔍 Semantic Search<br/>Logic"]
        C3["📊 Call Analytics<br/>& KPIs"]
        C4["🎙️ Transcription<br/>Pipeline"]
        C5["💬 Query<br/>Interpretation"]
    end
    
    subgraph SUPPORTING["⚙️ SUPPORTING DOMAIN - Technical Enablers"]
        direction TB
        S1["🗄️ Vector Store<br/>Management"]
        S2["🧮 Embedding<br/>Generation"]
        S3["📈 Data<br/>Aggregation"]
        S4["💾 Storage<br/>Management"]
    end
    
    subgraph GENERIC["🔧 GENERIC DOMAIN - Infrastructure"]
        direction TB
        G1["🎵 Audio<br/>Processing"]
        G2["📄 CSV<br/>Processing"]
        G3["🖥️ UI<br/>Framework"]
        G4["📁 File<br/>Storage"]
        G5["📝 Logging"]
    end
    
    %% Core to Supporting relationships
    C1 ==>|requires| S4
    C2 ==>|uses| S1
    C2 ==>|uses| S2
    C3 ==>|needs| S3
    C5 ==>|relies on| S3
    
    %% Core to Generic relationships
    C4 ==>|utilizes| G1
    
    %% Supporting to Generic relationships
    S1 ==>|stores in| G4
    S2 ==>|persists to| G4
    S4 ==>|uses| G4
    
    %% Professional Corporate Theme - Core Domain (Dark Blue)
    style C1 fill:#1E88E5,stroke:#0D47A1,stroke-width:3px,color:#FFFFFF,font-size:15px,font-weight:bold
    style C2 fill:#1E88E5,stroke:#0D47A1,stroke-width:3px,color:#FFFFFF,font-size:15px,font-weight:bold
    style C3 fill:#1E88E5,stroke:#0D47A1,stroke-width:3px,color:#FFFFFF,font-size:15px,font-weight:bold
    style C4 fill:#1E88E5,stroke:#0D47A1,stroke-width:3px,color:#FFFFFF,font-size:15px,font-weight:bold
    style C5 fill:#1E88E5,stroke:#0D47A1,stroke-width:3px,color:#FFFFFF,font-size:15px,font-weight:bold
    
    %% Supporting Domain (Teal)
    style S1 fill:#26A69A,stroke:#00695C,stroke-width:3px,color:#FFFFFF,font-size:14px,font-weight:bold
    style S2 fill:#26A69A,stroke:#00695C,stroke-width:3px,color:#FFFFFF,font-size:14px,font-weight:bold
    style S3 fill:#26A69A,stroke:#00695C,stroke-width:3px,color:#FFFFFF,font-size:14px,font-weight:bold
    style S4 fill:#26A69A,stroke:#00695C,stroke-width:3px,color:#FFFFFF,font-size:14px,font-weight:bold
    
    %% Generic Domain (Gray)
    style G1 fill:#78909C,stroke:#455A64,stroke-width:3px,color:#FFFFFF,font-size:14px,font-weight:bold
    style G2 fill:#78909C,stroke:#455A64,stroke-width:3px,color:#FFFFFF,font-size:14px,font-weight:bold
    style G3 fill:#78909C,stroke:#455A64,stroke-width:3px,color:#FFFFFF,font-size:14px,font-weight:bold
    style G4 fill:#78909C,stroke:#455A64,stroke-width:3px,color:#FFFFFF,font-size:14px,font-weight:bold
    style G5 fill:#78909C,stroke:#455A64,stroke-width:3px,color:#FFFFFF,font-size:14px,font-weight:bold
    
    %% Subgraphs
    style CORE fill:#F5F5F5,stroke:#0D47A1,stroke-width:2px,color:#0D47A1,font-size:16px,font-weight:bold
    style SUPPORTING fill:#F5F5F5,stroke:#00695C,stroke-width:2px,color:#00695C,font-size:16px,font-weight:bold
    style GENERIC fill:#F5F5F5,stroke:#455A64,stroke-width:2px,color:#455A64,font-size:16px,font-weight:bold
    
    linkStyle default stroke:#37474F,stroke-width:3px
```

I think of the dark-blue boxes as the bespoke logic in `src/core/labeling_engine.py:24`, `src/analysis/semantic_search.py:16`, and `src/analysis/aggregations.py:1`, while the teal and grey layers correspond to reusable helpers like `src/core/storage_manager.py:19` and the framework glue in `src/ui/pages/analysis.py:74`.

### Relationships Between Data and Services

The entity-relationship diagram shows how the central Pydantic model connects to downstream services and persistence layers.

```{mermaid}
erDiagram
    CALL_RECORD ||--o{ TRANSCRIPTION_RESULT : has
    CALL_RECORD ||--o{ EMBEDDING : generates
    CALL_RECORD ||--o{ LABELING_RESULT : produces
    
    CALL_RECORD {
        string call_id PK
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
