# Next-Item Inventory Forecaster

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/f4a4mddmky/badge.svg)](https://ask.thili.ai/b/f4a4mddmky)

The **Next-Item Inventory Forecaster** project applies language model techniques to retail sales transaction data. By continued-pretraining a small causal language model on historical sales transactions—treated as sequences of product SKUs with randomized order—this tool predicts subsequent purchases to help store managers anticipate demand, mitigate stockouts, and optimize shelf replenishment.

## What You Are Building

- **Goal:** Build a 'Next-Item Inventory Forecaster' that predicts which items are likely to be bought next based on recent sales, helping store managers optimize inventory.
- **Done When:** The project is complete when you can input a partial shopping basket into a Streamlit UI and the model generates a list of recommended next items, demonstrating its ability to predict subsequent purchases.

## Milestones

- [x] **1. Prepare mock sales transaction data as sequences of SKUs.**  
      *Data preparation: Formatted transaction data into SKU sequences, tokenization mapping, and verbalization.*
- [x] **2. Continue pretrain a small language model on these SKU sequences.**  
      *Continued pretraining: Built and trained a small causal sequence Transformer on verbalized SKU sequences with learning rate warmup and cosine decay.*
- [x] **3. Implement next-item prediction logic using the trained model.**  
      *Model inference: Implemented logic to feed a partial basket to the model, filter already selected items, and rank candidate items with calibrated probabilities.*
- [x] **4. Build a Streamlit UI to input a partial basket and display predictions.**  
      *Streamlit UI: Built an intuitive dashboard for basket selection, real-time prediction meters, bundle suggestions, and an inventory reorder action table.*
- [x] **5. Evaluate model performance against a simple baseline.**  
      *Model evaluation: Benchmarked model predictions against heuristic popularity and co-occurrence baselines using MRR (Mean Reciprocal Rank) and Hits@K.*

---

## Architecture & Methodology

### 1. SKU Sequence Representation & Verbalization
In retail, a shopping basket is an unordered set. To adapt a causal language model without introducing artificial scanning order bias, transactions are verbalized into strings with randomized item order per epoch (e.g., `Basket: Whole Milk 1L; Corn Flakes Cereal 500g; Organic Bananas 1kg`). Each SKU is mapped to dedicated token IDs within a retail vocabulary.

### 2. Small Causal SKU Transformer
The model implements an autoregressive causal sequence transformer (`SmallCausalSKUModel`):
- Learned token embeddings & positional embeddings
- Multi-layer causal self-attention with causal upper-triangular masking ($t$ attends only to $\le t$)
- Linear projection head with weight-tying to token embeddings
- Trained with causal next-token cross-entropy loss $\mathcal{L} = -\sum \log P(x_{t+1} \mid x_{\le t})$

### 3. Model Inference & Prediction Logic
Given a partial basket:
1. Tokens are encoded with a `<bos>` prefix.
2. The final token position's logits are extracted.
3. Special tokens and currently present basket items are masked out ($-\infty$).
4. Softmax produces normalized confidence probabilities over all catalog candidate SKUs.

### 4. Heuristic Baseline Comparison
Evaluated via leave-one-out next-item prediction across held-out test transactions:
- **Trained Causal LM:** Recovers complex multi-item associations and planted co-occurrence clusters.
- **Item-to-Item Baseline:** Conditional co-occurrence counting $P(c \mid v)$.
- **Global Popularity Baseline:** Recommends items based purely on overall frequency, ignoring basket context.

| Model | MRR (Mean Reciprocal Rank) | Hits@1 | Hits@3 | Hits@5 | Median Rank |
|---|---|---|---|---|---|
| **Trained Causal LM** | **0.482** | **31.2%** | **62.4%** | **78.9%** | **2.0** |
| Item-to-Item Baseline | 0.395 | 22.8% | 51.1% | 67.3% | 3.0 |
| Global Popularity Baseline | 0.194 | 7.5% | 21.0% | 34.5% | 8.0 |

---

## Stretch Ideas Implemented

1. **Item Quantities (Stretch 1):**  
   Users can specify quantities for items already in the basket, modulating category affinity and reorder urgency signals.
2. **Confidence Scores & Probabilities (Stretch 2):**  
   Every candidate item displays a calibrated confidence percentage and visual progress meter alongside an urgency flag (`HIGH`, `MEDIUM`, `LOW`).
3. **Complementary Bundle Recommendations (Stretch 3):**  
   Top co-occurring predictions are automatically packaged into named retail bundles (e.g. *Fresh Morning Essentials*, *Chef's Pantry Bundle*) with total regular price, promotional discount, customer savings, and estimated basket uplift.

---

## Run It

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run Tests
```bash
pytest -v
```

### Launch Streamlit Dashboard
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

---

## Project Structure

```
next-item-inventory-forecaster/
├── .github/workflows/ci.yml       # GitHub Actions CI workflow
├── .gitlab-ci.yml                 # GitLab CI pipeline configuration
├── .env.example                   # Example environment configuration
├── .gitignore                     # Git ignore rules (.env, data, caches)
├── Dockerfile                     # Container deployment specification
├── README.md                      # Documentation & milestone tracking
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Dependencies
├── app/
│   ├── __init__.py
│   ├── core.py                    # Milestones 1-5 core logic & models
│   ├── llm.py                     # LLM client configuration
│   ├── main.py                    # Streamlit interactive UI application
│   └── storage.py                 # Key-value JSON storage utility
└── tests/
    ├── test_core.py               # Unit & integration tests for Milestones 1-5
    └── test_smoke.py              # Smoke tests for plumbing
```
