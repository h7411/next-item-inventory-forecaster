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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build a Streamlit UI to input a partial basket and display predictions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Evaluate model performance against a simple baseline.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
