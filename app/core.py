"""Core logic for Next-Item Inventory Forecaster.

Milestone 1: Prepare mock sales transaction data as sequences of SKUs.
Milestone 2: Continue pretrain a small language model on these SKU sequences.
Milestone 3: Implement next-item prediction logic using the trained model.
Milestone 4: Build a Streamlit UI to input a partial basket and display predictions.
Milestone 5: Evaluate model performance against a simple baseline.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ============================================================================
# Milestone 1: Prepare Mock Sales Transaction Data as Sequences of SKUs
# ============================================================================

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<sep>": 3,
    "<unk>": 4,
}

PREFIX = "Basket: "
SEPARATOR = "; "


@dataclass(frozen=True)
class SKUItem:
    sku_id: str
    name: str
    category: str
    base_price: float
    unit: str

    def to_dict(self) -> dict:
        return asdict(self)


# Default Retail SKU Catalog
DEFAULT_CATALOG: List[SKUItem] = [
    # Dairy & Breakfast
    SKUItem("SKU-DAI-01", "Whole Milk 1L", "Dairy", 1.49, "1L"),
    SKUItem("SKU-DAI-02", "Oat Milk Barista 1L", "Dairy", 2.29, "1L"),
    SKUItem("SKU-DAI-03", "Greek Yogurt 500g", "Dairy", 2.19, "500g"),
    SKUItem("SKU-DAI-04", "Salted Butter 250g", "Dairy", 2.49, "250g"),
    SKUItem("SKU-DAI-05", "Cheddar Cheese 200g", "Dairy", 2.99, "200g"),
    SKUItem("SKU-DAI-06", "Parmesan Wedge 150g", "Dairy", 3.89, "150g"),
    # Bakery
    SKUItem("SKU-BAK-01", "Sourdough Bread 500g", "Bakery", 2.89, "500g"),
    SKUItem("SKU-BAK-02", "Whole Wheat Toast 400g", "Bakery", 1.79, "400g"),
    SKUItem("SKU-BAK-03", "Butter Croissant 2pk", "Bakery", 2.49, "pack"),
    SKUItem("SKU-BAK-04", "Garlic Baguette 250g", "Bakery", 1.99, "250g"),
    # Produce
    SKUItem("SKU-PRD-01", "Organic Bananas 1kg", "Produce", 1.69, "1kg"),
    SKUItem("SKU-PRD-02", "Gala Apples 1kg", "Produce", 2.49, "1kg"),
    SKUItem("SKU-PRD-03", "Yellow Onions 1kg", "Produce", 1.29, "1kg"),
    SKUItem("SKU-PRD-04", "Baby Spinach 200g", "Produce", 1.89, "200g"),
    SKUItem("SKU-PRD-05", "Roma Tomatoes 500g", "Produce", 1.99, "500g"),
    SKUItem("SKU-PRD-06", "Hass Avocados 2pk", "Produce", 2.79, "pack"),
    # Pantry & Grains
    SKUItem("SKU-PAN-01", "Corn Flakes Cereal 500g", "Pantry", 3.19, "500g"),
    SKUItem("SKU-PAN-02", "Rolled Oats 1kg", "Pantry", 2.29, "1kg"),
    SKUItem("SKU-PAN-03", "Penne Rigate Pasta 500g", "Pantry", 1.59, "500g"),
    SKUItem("SKU-PAN-04", "Classic Marinara Sauce 400g", "Pantry", 2.49, "400g"),
    SKUItem("SKU-PAN-05", "Extra Virgin Olive Oil 500ml", "Pantry", 5.99, "500ml"),
    SKUItem("SKU-PAN-06", "Brown Sugar 500g", "Pantry", 1.39, "500g"),
    # Beverages & Snacks
    SKUItem("SKU-BEV-01", "Dark Roast Coffee Beans 250g", "Beverages", 4.99, "250g"),
    SKUItem("SKU-BEV-02", "Orange Juice Fresh 1L", "Beverages", 2.89, "1L"),
    SKUItem("SKU-BEV-03", "Sparkling Spring Water 1L", "Beverages", 0.99, "1L"),
    SKUItem("SKU-SNK-01", "Sea Salt Tortilla Chips 200g", "Snacks", 2.19, "200g"),
    SKUItem("SKU-SNK-02", "Chunky Tomato Salsa 300g", "Snacks", 2.59, "300g"),
    SKUItem("SKU-SNK-03", "Dark Chocolate Cookies 150g", "Snacks", 2.69, "150g"),
]

# Planted Frequent Co-Occurrence Bundles (Retail Market Basket Association Rules)
PLANTED_BUNDLES = [
    # Breakfast Essentials
    ("Whole Milk 1L", "Corn Flakes Cereal 500g", "Organic Bananas 1kg"),
    # Italian Pasta Feast
    ("Penne Rigate Pasta 500g", "Classic Marinara Sauce 400g", "Parmesan Wedge 150g", "Garlic Baguette 250g"),
    # Morning Coffee & Pastry
    ("Dark Roast Coffee Beans 250g", "Oat Milk Barista 1L", "Butter Croissant 2pk"),
    # Gourmet Sandwich
    ("Sourdough Bread 500g", "Cheddar Cheese 200g", "Salted Butter 250g", "Roma Tomatoes 500g"),
    # Chips & Dip Party
    ("Sea Salt Tortilla Chips 200g", "Chunky Tomato Salsa 300g", "Hass Avocados 2pk", "Sparkling Spring Water 1L"),
    # Healthy Oats Bowl
    ("Rolled Oats 1kg", "Greek Yogurt 500g", "Organic Bananas 1kg", "Gala Apples 1kg"),
]


class Tokenizer:
    """SKU-level sequence tokenizer for retail transaction modeling."""

    def __init__(self, catalog: List[SKUItem]):
        self.catalog = list(catalog)
        self.name_to_sku = {item.name: item for item in catalog}
        self.sku_to_item = {item.sku_id: item for item in catalog}

        # Build vocabulary
        self.token_to_id: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

        curr_id = len(SPECIAL_TOKENS)
        for item in self.catalog:
            self.token_to_id[item.name] = curr_id
            self.id_to_token[curr_id] = item.name
            curr_id += 1

        self.vocab_size = len(self.token_to_id)
        self.pad_token_id = SPECIAL_TOKENS["<pad>"]
        self.bos_token_id = SPECIAL_TOKENS["<bos>"]
        self.eos_token_id = SPECIAL_TOKENS["<eos>"]
        self.sep_token_id = SPECIAL_TOKENS["<sep>"]
        self.unk_token_id = SPECIAL_TOKENS["<unk>"]

    def encode(self, items: Sequence[str], add_special_tokens: bool = True) -> List[int]:
        """Convert a sequence of SKU names to token IDs."""
        tokens = [self.token_to_id.get(item, self.unk_token_id) for item in items]
        if add_special_tokens:
            return [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, token_ids: Sequence[int], skip_special: bool = True) -> List[str]:
        """Convert token IDs back to SKU names."""
        items = []
        special_ids = set(SPECIAL_TOKENS.values())
        for tid in token_ids:
            if skip_special and tid in special_ids:
                continue
            token = self.id_to_token.get(tid, "<unk>")
            items.append(token)
        return items


def verbalize_basket(items: Sequence[str], seed: Optional[int] = None, shuffle: bool = True) -> str:
    """Format a basket of SKU names as a verbalized text string.
    
    Order is shuffled per verbalization call to avoid artificial scanning order bias.
    """
    unique_items = sorted(set(items))
    if shuffle:
        rng = random.Random(seed)
        shuffled = list(unique_items)
        rng.shuffle(shuffled)
        items_to_format = shuffled
    else:
        items_to_format = unique_items
    return PREFIX + SEPARATOR.join(items_to_format)


def parse_verbalized_basket(text: str) -> List[str]:
    """Parse a verbalized basket string back into individual SKU names."""
    text = text.strip()
    if text.startswith(PREFIX):
        text = text[len(PREFIX):]
    if not text:
        return []
    return [item.strip() for item in text.split(SEPARATOR.strip()) if item.strip()]


def generate_mock_transactions(
    n_transactions: int = 2500,
    catalog: Optional[List[SKUItem]] = None,
    planted_bundles: Optional[List[Tuple[str, ...]]] = None,
    planted_prob: float = 0.55,
    seed: int = 42,
) -> List[List[str]]:
    """Generate realistic sales transaction baskets.
    
    Combines background random purchases with planted high-affinity retail bundles.
    """
    rng = random.Random(seed)
    cat = catalog or DEFAULT_CATALOG
    bundles = planted_bundles or PLANTED_BUNDLES
    all_names = [item.name for item in cat]

    # Category-based background popularity weights (e.g., staples bought more often)
    category_multipliers = {
        "Dairy": 2.2,
        "Bakery": 2.0,
        "Produce": 1.8,
        "Pantry": 1.3,
        "Beverages": 1.4,
        "Snacks": 1.2,
    }
    weights = [category_multipliers.get(item.category, 1.0) for item in cat]
    total_w = sum(weights)
    item_probs = [w / total_w for w in weights]

    baskets: List[List[str]] = []
    for _ in range(n_transactions):
        basket: Set[str] = set()

        # Decide whether to plant an association rule bundle
        if rng.random() < planted_prob:
            bundle = rng.choice(bundles)
            # Pick a subset or full bundle
            subset_size = rng.randint(max(2, len(bundle) - 1), len(bundle))
            picked_bundle_items = rng.sample(bundle, subset_size)
            basket.update(picked_bundle_items)

        # Add background items (1 to 4 extra items)
        bg_count = rng.randint(1, 3)
        extra = rng.choices(all_names, weights=item_probs, k=bg_count)
        basket.update(extra)

        if len(basket) < 2:
            # ensure at least 2 items for next-item prediction learning
            extra_two = rng.sample(all_names, 2)
            basket.update(extra_two)

        baskets.append(sorted(basket))

    return baskets


def prepare_sku_sequences(
    baskets: List[List[str]],
    tokenizer: Tokenizer,
    max_len: int = 16,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert transaction baskets into padded input/target SKU token arrays and verbalized texts."""
    rng = random.Random(seed)
    encoded_list = []
    verbalized_texts = []

    for basket in baskets:
        # Shuffled order for verbalization and sequence input
        shuffled = list(basket)
        rng.shuffle(shuffled)
        verbalized = verbalize_basket(shuffled, shuffle=False)
        verbalized_texts.append(verbalized)

        # Numerical tokens: <bos> + items + <eos>
        token_ids = tokenizer.encode(shuffled, add_special_tokens=True)
        encoded_list.append(token_ids)

    # Pad sequences
    n_samples = len(encoded_list)
    padded_inputs = np.full((n_samples, max_len), tokenizer.pad_token_id, dtype=np.int64)
    padded_targets = np.full((n_samples, max_len), tokenizer.pad_token_id, dtype=np.int64)

    for i, seq in enumerate(encoded_list):
        seq = seq[:max_len]
        length = len(seq)
        padded_inputs[i, :length] = seq
        # targets shifted by 1 for causal language modeling
        padded_targets[i, :length - 1] = seq[1:length]

    return padded_inputs, padded_targets, verbalized_texts


def split_dataset(
    baskets: List[List[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """Split transaction baskets into Train, Validation, and Test sets."""
    rng = random.Random(seed)
    shuffled = list(baskets)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def milestone_1() -> dict[str, Any]:
    """Milestone 1: Prepare mock sales transaction data as sequences of SKUs."""
    catalog = DEFAULT_CATALOG
    tokenizer = Tokenizer(catalog)
    transactions = generate_mock_transactions(n_transactions=1500, catalog=catalog, seed=42)
    train_b, val_b, test_b = split_dataset(transactions, seed=42)

    train_inputs, train_targets, train_texts = prepare_sku_sequences(train_b, tokenizer, seed=42)
    val_inputs, val_targets, val_texts = prepare_sku_sequences(val_b, tokenizer, seed=42)

    return {
        "status": "completed",
        "catalog_size": len(catalog),
        "total_transactions": len(transactions),
        "train_count": len(train_b),
        "val_count": len(val_b),
        "test_count": len(test_b),
        "vocab_size": tokenizer.vocab_size,
        "sample_verbalized": train_texts[:3],
        "input_tensor_shape": train_inputs.shape,
    }


# ============================================================================
# Stubs for Milestones 2-5 (Implemented in subsequent steps)
# ============================================================================

def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
