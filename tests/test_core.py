"""Tests for Next-Item Inventory Forecaster milestones."""
import numpy as np
import pytest

from app.core import (
    DEFAULT_CATALOG,
    PLANTED_BUNDLES,
    SKUItem,
    Tokenizer,
    generate_mock_transactions,
    milestone_1,
    parse_verbalized_basket,
    prepare_sku_sequences,
    split_dataset,
    verbalize_basket,
)


# Milestone 1: Prepare mock sales transaction data as sequences of SKUs.
def test_milestone_1():
    res = milestone_1()
    assert res["status"] == "completed"
    assert res["catalog_size"] >= 20
    assert res["total_transactions"] >= 1000
    assert res["train_count"] > res["val_count"]
    assert res["vocab_size"] > res["catalog_size"]
    assert len(res["sample_verbalized"]) == 3
    assert all(text.startswith("Basket: ") for text in res["sample_verbalized"])

    # Test catalog and SKU properties
    assert isinstance(DEFAULT_CATALOG[0], SKUItem)
    assert DEFAULT_CATALOG[0].sku_id.startswith("SKU-")
    assert DEFAULT_CATALOG[0].base_price > 0

    # Test Tokenizer
    tok = Tokenizer(DEFAULT_CATALOG)
    encoded = tok.encode(["Whole Milk 1L", "Corn Flakes Cereal 500g"], add_special_tokens=True)
    assert encoded[0] == tok.bos_token_id
    assert encoded[-1] == tok.eos_token_id
    decoded = tok.decode(encoded, skip_special=True)
    assert decoded == ["Whole Milk 1L", "Corn Flakes Cereal 500g"]

    # Test verbalize and parse
    basket = ["Whole Milk 1L", "Organic Bananas 1kg"]
    verb = verbalize_basket(basket, shuffle=False)
    assert verb == "Basket: Organic Bananas 1kg; Whole Milk 1L"
    parsed = parse_verbalized_basket(verb)
    assert set(parsed) == set(basket)

    # Test sequence preparation shapes
    inputs, targets, texts = prepare_sku_sequences([basket], tok, max_len=8)
    assert inputs.shape == (1, 8)
    assert targets.shape == (1, 8)
    assert len(texts) == 1

    # Test dataset splitting
    dummy_baskets = [["A", "B"]] * 100
    tr, va, te = split_dataset(dummy_baskets, train_ratio=0.8, val_ratio=0.1)
    assert len(tr) == 80
    assert len(va) == 10
    assert len(te) == 10


# Milestone 2: Continue pretrain a small language model on these SKU sequences.
def test_milestone_2(tmp_path):
    import torch
    from app.core import (
        SmallCausalSKUModel,
        Tokenizer,
        DEFAULT_CATALOG,
        load_model_checkpoint,
        milestone_2,
        save_model_checkpoint,
    )

    # Test model initialization and forward shape
    tok = Tokenizer(DEFAULT_CATALOG)
    model = SmallCausalSKUModel(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=1, max_seq_len=16)
    dummy_input = torch.tensor([[tok.bos_token_id, 5, 6, tok.pad_token_id]], dtype=torch.long)
    logits = model(dummy_input)
    assert logits.shape == (1, 4, tok.vocab_size)

    # Test loss computation
    dummy_target = torch.tensor([[5, 6, tok.eos_token_id, tok.pad_token_id]], dtype=torch.long)
    loss = model.compute_loss(logits, dummy_target)
    assert loss.item() > 0.0

    # Test saving and loading weights
    ckpt = tmp_path / "test_model.pt"
    save_model_checkpoint(model, ckpt)
    assert ckpt.exists()
    loaded_model = SmallCausalSKUModel(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=1, max_seq_len=16)
    load_model_checkpoint(loaded_model, ckpt)

    # Test full milestone 2 pretraining execution (quick 3 epochs)
    res = milestone_2(epochs=3)
    assert res["status"] == "completed"
    assert res["loss_reduced"] is True
    assert res["final_loss"] < res["initial_loss"]
    assert res["model_parameters"] > 0
    assert len(res["history"]) == 3


# Milestone 3: Implement next-item prediction logic using the trained model.
def test_milestone_3():
    from app.core import (
        DEFAULT_CATALOG,
        Tokenizer,
        milestone_3,
        predict_next_items,
        suggest_complementary_bundle,
    )

    res = milestone_3()
    assert res["status"] == "completed"
    assert len(res["sample_forecasts"]) == 3
    assert len(res["quantity_test_predictions"]) == 3

    # Check structure of predictions
    for basket_text, payload in res["sample_forecasts"].items():
        assert "partial_basket" in payload
        preds = payload["top_predictions"]
        assert len(preds) > 0
        assert preds[0]["confidence"] > 0
        assert preds[0]["reorder_urgency"] in {"HIGH", "MEDIUM", "LOW"}
        assert "%" in preds[0]["confidence_pct"]

        # Check bundle
        bundle = payload["bundle_suggestion"]
        assert "bundle_name" in bundle
        assert bundle["total_price"] >= bundle["discounted_price"]
        assert bundle["savings"] >= 0

    # Test exclusion of current items
    catalog = DEFAULT_CATALOG
    tok = Tokenizer(catalog)
    # create model and test exclusions
    from app.core import SmallCausalSKUModel
    model = SmallCausalSKUModel(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=1, max_seq_len=16)
    preds = predict_next_items(model, tok, ["Whole Milk 1L"], top_k=5, exclude_current=True)
    pred_names = [p.name for p in preds]
    assert "Whole Milk 1L" not in pred_names


# Milestone 4: Build a Streamlit UI to input a partial basket and display predictions.
def test_milestone_4():
    import py_compile
    from app.core import (
        DEFAULT_CATALOG,
        PRESET_BASKETS,
        calculate_restock_recommendation,
        get_catalog_names,
        get_sku_details,
        milestone_4,
    )

    # Test py_compile of app/main.py
    py_compile.compile("app/main.py", doraise=True)

    # Test UI helper functions
    names = get_catalog_names(DEFAULT_CATALOG)
    assert len(names) == len(DEFAULT_CATALOG)
    assert "Whole Milk 1L" in names

    sku_obj = get_sku_details("Whole Milk 1L")
    assert sku_obj is not None
    assert sku_obj.category == "Dairy"

    assert len(PRESET_BASKETS) >= 4
    assert "Breakfast Essentials" in PRESET_BASKETS

    # Test milestone 4 workflow execution
    res = milestone_4()
    assert res["status"] == "completed"
    assert res["catalog_item_count"] >= 20
    assert res["preset_count"] >= 4
    assert res["predictions_generated"] > 0
    assert res["restock_table_rows"] > 0


# Milestone 5: Evaluate model performance against a simple baseline.
def test_milestone_5():
    from app.core import (
        DEFAULT_CATALOG,
        ItemToItemBaseline,
        PopularityBaseline,
        evaluate_ranking,
        make_eval_cases,
        milestone_5,
    )

    # Test metric calculations
    dummy_ranks = [1, 2, 4, 1]
    metrics = evaluate_ranking(dummy_ranks)
    assert metrics["mrr"] == round((1.0 + 0.5 + 0.25 + 1.0) / 4.0, 4)
    assert metrics["hits@1"] == 0.5
    assert metrics["hits@3"] == 0.75

    # Test baseline candidate scoring
    catalog = DEFAULT_CATALOG
    train_baskets = [["Whole Milk 1L", "Corn Flakes Cereal 500g"]] * 10
    pop = PopularityBaseline(train_baskets, catalog)
    scores = pop.score_candidates(["Whole Milk 1L"], [item.name for item in catalog])
    assert scores["Whole Milk 1L"] == -float("inf")
    assert scores["Corn Flakes Cereal 500g"] > 0

    item_base = ItemToItemBaseline(train_baskets, catalog)
    i_scores = item_base.score_candidates(["Whole Milk 1L"], [item.name for item in catalog])
    assert i_scores["Whole Milk 1L"] == -float("inf")
    assert i_scores["Corn Flakes Cereal 500g"] > 0

    # Test eval cases generation
    cases = make_eval_cases([["A", "B", "C"], ["D", "E"]], n_cases=5)
    assert len(cases) == 2
    assert "visible" in cases[0] and "target" in cases[0]

    # Test full milestone 5 execution (quick 40 cases)
    res = milestone_5(n_cases=40)
    assert res["status"] == "completed"
    assert res["evaluation_cases_evaluated"] > 0
    assert "mrr" in res["trained_model_metrics"]
    assert "mrr" in res["popularity_baseline_metrics"]
    assert res["trained_model_outperforms_baseline"] is True
    assert res["mrr_improvement_over_popularity"] > 0
    assert len(res["comparison_table"]) == 3
