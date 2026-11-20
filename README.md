# Next-Item Inventory Forecaster

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/f4a4mddmky/badge.svg)](https://ask.thili.ai/b/f4a4mddmky)

The 'Next-Item Inventory Forecaster' project applies language model techniques to retail sales data. By continued-pretraining a small language model on mock historical sales transactions, treated as sequences of product SKUs, this tool predicts subsequent purchases. This helps store managers anticipate demand and optimize inventory reordering.

## What you are building

**Goal.** You will build a 'Next-Item Inventory Forecaster' that predicts which items are likely to be bought next based on recent sales, helping store managers optimize inventory.

**Done when.** The project is complete when you can input a partial shopping basket into a Streamlit UI and the model generates a list of recommended next items, demonstrating its ability to predict subsequent purchases.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare mock sales transaction data as sequences of SKUs.**  
      Hint: Data preparation: Format transaction data into SKU sequences.
- [ ] **2. Continue pretrain a small language model on these SKU sequences.**  
      Hint: Continued pretraining: Use the lfretail-lfmarketbasket techniques on your SKU sequences.
- [ ] **3. Implement next-item prediction logic using the trained model.**  
      Hint: Model inference: Implement logic to feed a partial basket to the trained model and get predictions.
- [ ] **4. Build a Streamlit UI to input a partial basket and display predictions.**  
      Hint: Streamlit UI: Design an interface for basket input and prediction display.
- [ ] **5. Evaluate model performance against a simple baseline.**  
      Hint: Model evaluation: Compare your model's predictions against a simple heuristic baseline.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to specify quantities for items already in the basket, if the model can incorporate this information.
- Display confidence scores or probabilities alongside the predicted next items.
- Implement a feature to suggest entire complementary bundles based on the top predicted items.
