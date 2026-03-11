# Word Embeddings & Semantic Search
### by Kavyasree Nunna

A semantic similarity search engine built using sentence-level word embeddings, demonstrating how transformer-based models understand meaning beyond keyword matching.

---

## What It Does

This notebook builds a mini semantic search system over a custom multi-domain corpus. Given a natural language query, it retrieves the most contextually relevant sentences — even when no exact words match — using cosine similarity over dense vector embeddings.

**Example:**
> Query: *"How can I build a 3D map from 2D images?"*
> Top Match: *"The 2D-to-3D reconstruction pipeline uses depth estimation."* → Score: 0.453

---

## How It Works

1. **Model**: Loads `all-MiniLM-L6-v2` from Sentence Transformers — a lightweight BERT-based model optimized for semantic similarity tasks
2. **Corpus**: A 50-sentence dataset spanning three distinct domains:
   - Robotics & Computer Vision (VL-Maps, NeRF, SLAM, Gaussian Splatting)
   - Carnatic Classical Music (Ragas, Tala, Gamaka)
   - University / Campus life (hackathons, internships, GDG)
3. **Encoding**: All corpus sentences are encoded into 384-dimensional dense vectors
4. **Retrieval**: Cosine similarity is computed between the query embedding and all corpus embeddings using `util.cos_sim()`
5. **Output**: Top-k most similar sentences are returned with their similarity scores

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `sentence-transformers` | Pre-trained transformer model for embedding |
| `all-MiniLM-L6-v2` | Lightweight BERT model (384-dim embeddings) |
| `PyTorch` | Tensor operations and `torch.topk` for ranking |
| `pandas` | Data structuring |
| Google Colab | Runtime environment |

---

## Sample Results

```
Query: What is the best Raga for a beginner?
 -> [0.661] Mayamalavagowla is often the first Raga taught to beginners.
 -> [0.348] Raga Shankarabharanam corresponds to the major scale in Western music.
 -> [0.338] A Melakarta Raga is a parent scale in the Carnatic system.

Query: How many beats in a standard music cycle?
 -> [0.542] The Adi Tala consists of eight beats in a specific cycle.
 -> [0.333] Mridangam is the primary percussion instrument in a concert.
 -> [0.293] Katcheris usually begin with a Varnam to set the tempo.
```

---

## Run It Yourself

1. Open the notebook in Google Colab
2. Run the first cell to install dependencies:
```bash
pip install sentence-transformers
```
3. Run the second cell to load the model, encode the corpus, and test queries
4. Modify `test_queries` to search with your own questions

---

## Key Concepts Demonstrated

- **Semantic search** vs keyword search — finds meaning, not just matching words
- **Dense vector embeddings** — sentences encoded as numerical vectors in high-dimensional space
- **Cosine similarity** — measures angular distance between vectors as a proxy for semantic closeness
- **Transformer-based NLP** — leveraging pre-trained BERT architecture for downstream retrieval tasks

---

## About the Corpus Design

The corpus intentionally spans unrelated domains (robotics, music, campus life) to demonstrate the model's ability to isolate semantic meaning within a noisy, mixed-topic dataset — a key challenge in real-world retrieval-augmented generation (RAG) systems.

---

*Part of an ongoing exploration into NLP foundations and retrieval systems.*
