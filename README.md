# 🕸️ graphnlp-intel

![PyPI - Version](https://img.shields.io/pypi/v/graphnlp-intel?style=flat-square&color=00c896)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)
![Docs](https://img.shields.io/badge/docs-latest-blue?style=flat-square)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)

**graphnlp-intel** is an open-source Python library and REST API that transforms unstructured documents into rich, interactive knowledge graphs using state-of-the-art NLP, relationship extraction, and GNN-based sentiment propagation.

## 🚀 Quickstart

Install the library and download the required spaCy model:

```bash
pip install graphnlp-intel
python -m spacy download en_core_web_sm
```

Run the pipeline in 6 lines of code:

```python
from graphnlp import Pipeline

pipe = Pipeline(domain="finance")
result = pipe.run(["Goldman Sachs acquired a 5% stake in Microsoft for $2.3 billion."])

# Visualize, export, and summarize
result.graph.visualize("output.html") # Generates a Pyvis interactive HTML graph
result.export_json("output.json")    # Exports D3 compatible JSON
print(result.summary())              # Output stats on nodes, edges, sentiment, and communities
```

## 🧠 How it works

The system processes unstructured text through a 5-stage pipeline:

```text
 📄 Ingestion      🔍 Extraction         🕸️ Graph Build         🧠 GNN              📈 Output
 DocumentLoader → NERExtractor       → GraphBuilder       → GraphGNN          → Pyvis HTML /
 TextChunker      RelationExtractor    CommunityDetector                        D3 JSON /
 EmailParser      EmbeddingExtractor                                            Neo4j / Redis
```

### Standalone Extractor Usage
```python
from graphnlp.extraction.ner import NERExtractor
from graphnlp.extraction.relations import RelationExtractor

ner = NERExtractor()
entities = ner.extract("Apple Inc reported revenue of $120 billion.")

rel_ext = RelationExtractor()
triples = rel_ext.extract("Apple Inc reported revenue of $120 billion.")
```

### Standalone Graph Construction Usage
```python
from graphnlp.graph.builder import GraphBuilder
from graphnlp.graph.community import CommunityDetector
import networkx as nx

builder = GraphBuilder()
graph = builder.build(triples, entities, embeddings_dict)

detector = CommunityDetector()
communities = detector.detect(graph)
```

## 🧩 Domain adapters

Domain adapters supply contextual logic like schema mappings, preprocessing, and post-processing steps tailored to specific industries.

| Adapter | Entity Types | Use Case |
|---|---|---|
| `finance` | `TICKER`, `ORG`, `AMOUNT`, `DATE` | Parse fund records, expand ticker syms, build `COMPETITOR_OF` graphs |
| `email` | `PERSON`, `MERCHANT`, `MONEY` | Strip HTML/headers, parse invoices, generate `PAID_TO` expense clusters |
| `feedback` | `PRODUCT`, `SCORE`, `FEATURE` | Normalize 5-star ratings, cluster feature complaints, link reviews |
| `incidents` | `SERVICE`, `ERROR`, `SEV` | Standardize P0/P1 flags, deduplicate logs, build `AFFECTS` topological graphs |

### Using the Email Adapter
```python
from graphnlp.adapters.base import get_adapter
from graphnlp.adapters.email import EmailAdapter
import networkx as nx

adapter = get_adapter("email")
clean_text = adapter.preprocess(raw_email_string)

# Graph integration
g = nx.DiGraph()
g.add_edge("$234.56", "Amazon", predicate="paid_to")
spend_clusters = EmailAdapter.monthly_spend_summary(g)
```

### Custom Adapter Implementation
```python
from graphnlp.adapters.base import DomainAdapter

class HealthcareAdapter(DomainAdapter):
    @property
    def domain(self) -> str:
        return "healthcare"
        
    @property
    def entity_types(self) -> list[str]:
        return ["PATIENT", "SYMPTOM", "DRUG"]
        
    def preprocess(self, text: str) -> str:
        return text.replace("Pt.", "Patient")
```

## ⚡ API Server

Deploy the multi-tenant REST API via Docker:
```bash
make docker-up
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Check service health and system status. |
| `POST` | `/v1/analyze` | Submit documents for processing (sync or async). |
| `GET` | `/v1/analyze/{job_id}` | Poll status of an async analysis job. |
| `GET` | `/v1/graph/{graph_id}` | Retrieve D3.js compatible graph JSON by ID. |
| `GET` | `/v1/graph/{graph_id}/summary` | Retrieve summarized stats of the graph. |
| `POST` | `/v1/webhooks` | Register a new webhook endpoint for async complete events. |
| `GET` | `/v1/webhooks` | List registered webhooks for the given tenant. |

### Auth, Submit, and Poll

```bash
# Submit Sync
curl -X POST http://localhost:8000/v1/analyze \
  -H "Authorization: Bearer sk-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Invoice 123 for $500 to AWS"], "domain": "finance", "async": false}'

# Submit Async
curl -X POST http://localhost:8000/v1/analyze \
  -H "Authorization: Bearer sk-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Massive batch 1...", "Massive batch 2..."], "async": true}'

# Poll Async Status
curl -X GET http://localhost:8000/v1/analyze/job-1234 \
  -H "Authorization: Bearer sk-your-api-key"
```

## 📦 SDK Integration

### Python SDK
```bash
pip install graphnlp-client
```

```python
from graphnlp_client.client import GraphNLPClient

client = GraphNLPClient(api_key="sk-your-api-key", base_url="http://localhost:8000")

# Sync
result = client.analyze(["Azure bill $300"], domain="email")
print(result["graph_id"])

# Get Graph data
graph = client.get_graph(result["graph_id"])
```

### TypeScript / JavaScript SDK
```bash
npm install graphnlp-client
```

```typescript
import { GraphNLPClient } from 'graphnlp-client';

const client = new GraphNLPClient({ apiKey: 'sk-your-api-key' });

async function analyze() {
  const result = await client.analyze(['Q4 earnings were up 12%'], { domain: 'finance' });
  const graph = await client.getGraph(result.graph_id);
  console.log(graph.nodes);
}
```

## 🪝 Webhooks

Register webhooks to receive JSON payloads upon async task completion.

```bash
curl -X POST http://localhost:8000/v1/webhooks \
  -H "Authorization: Bearer sk-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://yourapp.com/hook", "events": ["graph.ready"], "secret": "wh_sec_123"}'
```

**Webhook Payload Example**
```json
{
  "event": "graph.ready",
  "job_id": "job-1234",
  "graph_id": "graph-5678",
  "tenant_id": "tenant-abc",
  "timestamp": "2026-04-18T10:00:00Z",
  "signature": "sha256=d2b8b9a..."
}
```

## ⚙️ Configuration

Configure the platform using `config/default.yaml` or environment variables:

```yaml
# config/default.yaml
environment: production
neo4j:
  uri: bolt://localhost:7687
redis:
  url: redis://localhost:6379
api:
  rate_limit_per_minute: 100
nlp:
  ner_model: en_core_web_sm
  embedding_model: all-MiniLM-L6-v2
```

```bash
# .env
GRAPHNLP_ENVIRONMENT=production
GRAPHNLP_NEO4J_URI=bolt://neo4j:7687
GRAPHNLP_NEO4J_USER=neo4j
GRAPHNLP_NEO4J_PASSWORD=supersecret
GRAPHNLP_REDIS_URL=redis://redis:6379
```

## 🛠️ CLI Reference

Manage the platform using the built-in Typer CLI:

- `graphnlp run --domain finance --file data.csv` : Run pipeline on a local file.
- `graphnlp serve --port 8000 --reload` : Start the FastAPI server.
- `graphnlp worker --concurrency 4` : Start the Celery async worker.
- `graphnlp generate-key -t my-tenant` : Generate a new API key for the specified tenant.

## 🏗️ Architecture

```text
graphnlp-intel/
├── graphnlp/
│   ├── config.py              # Pydantic Settings
│   ├── pipeline.py            # Main Orchestrator
│   ├── ingestion/             # Loaders, Chunkers, Email Parsers
│   ├── extraction/            # NER, Relations, SBERT Embeddings
│   ├── graph/                 # NetworkX Builder, PyG GNN, Diff, Louvain
│   ├── adapters/              # Domain-specific logic
│   ├── storage/               # Neo4j & Redis handlers
│   ├── api/                   # FastAPI routes, Auth, Tenant Middleware
│   ├── queue/                 # Celery workers & tasks
│   └── webhooks/              # HMAC Dispatcher
├── tests/
│   ├── unit/                  # Isolated logic blocks
│   ├── integration/           # E2E API tests
│   └── fixtures/              # CSV/JSON samples
├── sdk/
│   ├── python/                # PyPI API wrapper
│   └── js/                    # NPM API wrapper
├── docker/
│   ├── docker-compose.yml     # Local orchestration
│   ├── Dockerfile             # API Container
│   └── Dockerfile.worker      # Celery Container
└── pyproject.toml             # Dependencies & metadata
```

## 📚 Open Source Stack

We stand on the shoulders of giants.

| Component | Library |
|---|---|
| NLP Base | `spacy` |
| Deep Learning | `torch` |
| Graph Neural Nets | `torch-geometric` |
| Language Models | `transformers` |
| Sentence Embeddings | `sentence-transformers` |
| Graph Analytics | `networkx` |
| Async Queue | `celery` |
| Web Framework | `fastapi` |
| Configuration | `pydantic` |
| Caching & Rate Limits | `redis.asyncio` |
| Graph Persistence | `neo4j` (async driver) |
| CLI Generation | `typer` |

## 🗺️ Roadmap

| Phase | Milestone | Expected |
|---|---|---|
| **Phase 1** | Streaming Engine (Kafka integration, real-time diffing) | Q3 2026 |
| **Phase 2** | Custom Model Fine-Tuning (LoRA automated pipeline) | Q4 2026 |
| **Phase 3** | Visual Graph Dashboard (React SPA for interactive analytics) | Q1 2027 |

## 💼 Custom Builds & Enterprise

| Tier | Price | Features |
|---|---|---|
| **Open Source** | Free | Apache 2.0 · Self-hosted · All adapters · CLI |
| **Custom NER** | $800–2,000 | Fine-tune NER · HF model delivery · Eval report |
| **Hosted API** | $2,500 + $400/mo | **FEATURED** · AWS/GCP/Azure deploy · Docker + TF · SDK |
| **Enterprise** | $8,000+ | Streaming · Dashboard · Alerting SLA · White-label |

Interested in Hosted API or Enterprise tiers? [Get a quote](#) on our site.

## 🤝 Contributing

We welcome contributions! 

```bash
git clone https://github.com/samvardhan03/GraphNLP-Intel.git
cd GraphNLP-Intel
./setup_dev.sh
make test
```

## 📄 License

This project is licensed under the **Apache License 2.0**.

```bibtex
@software{graphnlpintel2026,
  author = {GraphNLP Team},
  title = {graphnlp-intel: Hybrid Graph-NLP Intelligence Platform},
  year = {2026},
  url = {https://github.com/samvardhan03/GraphNLP-Intel}
}
```
