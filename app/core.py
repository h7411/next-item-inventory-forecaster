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
# Milestone 2: Continue Pretrain a Small Language Model on SKU Sequences
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCausalSKUModel(nn.Module):
    """Small Causal Transformer Language Model for retail SKU sequence prediction."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        max_seq_len: int = 32,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying between token embeddings and projection head
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through causal sequence model.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        device = input_ids.device
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # Embeddings + Positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Causal mask: position t can only attend to <= t
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.bool).to(device)
        padding_mask = (input_ids == self.pad_idx)

        # Pass through causal transformer
        hidden = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)
        hidden = self.layer_norm(hidden)

        logits = self.lm_head(hidden)
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Causal next-token prediction loss ignoring pad tokens."""
        # Shift logits and targets so token t predicts t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
            ignore_index=self.pad_idx,
        )
        return loss


def train_epoch(
    model: SmallCausalSKUModel,
    optimizer: torch.optim.Optimizer,
    baskets: List[List[str]],
    tokenizer: Tokenizer,
    batch_size: int = 32,
    max_len: int = 16,
    rng_seed: int = 0,
) -> float:
    """Train for one epoch with newly permuted basket item orders."""
    model.train()
    rng = random.Random(rng_seed)

    # Permute order of items in each basket (lesson 3 verbalization principle)
    permuted_sequences = []
    for b in baskets:
        shuffled = list(b)
        rng.shuffle(shuffled)
        permuted_sequences.append(shuffled)

    # Shuffle dataset order
    rng.shuffle(permuted_sequences)

    total_loss = 0.0
    num_batches = 0

    for i in range(0, len(permuted_sequences), batch_size):
        batch_baskets = permuted_sequences[i : i + batch_size]
        inputs, targets, _ = prepare_sku_sequences(batch_baskets, tokenizer, max_len=max_len, seed=rng_seed + i)

        input_tensor = torch.tensor(inputs, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(input_tensor)
        loss = model.compute_loss(logits, target_tensor)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate_loss(
    model: SmallCausalSKUModel,
    baskets: List[List[str]],
    tokenizer: Tokenizer,
    batch_size: int = 32,
    max_len: int = 16,
) -> float:
    """Evaluate cross-entropy validation loss on held-out baskets."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(baskets), batch_size):
            batch_baskets = baskets[i : i + batch_size]
            inputs, targets, _ = prepare_sku_sequences(batch_baskets, tokenizer, max_len=max_len, seed=i)
            input_tensor = torch.tensor(inputs, dtype=torch.long)
            target_tensor = torch.tensor(targets, dtype=torch.long)

            logits = model(input_tensor)
            loss = model.compute_loss(logits, target_tensor)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


def pretrain_small_lm(
    model: SmallCausalSKUModel,
    tokenizer: Tokenizer,
    train_baskets: List[List[str]],
    val_baskets: Optional[List[List[str]]] = None,
    epochs: int = 6,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_len: int = 16,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Continued pretraining loop adapting the small causal LM to retail SKU sequences."""
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history = []
    for epoch in range(1, epochs + 1):
        # Linear warmup then cosine decay
        if epoch <= 2:
            current_lr = lr * (epoch / 2.0)
        else:
            progress = (epoch - 2) / max(1, epochs - 2)
            current_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        train_loss = train_epoch(
            model, optimizer, train_baskets, tokenizer,
            batch_size=batch_size, max_len=max_len, rng_seed=seed + epoch,
        )

        val_loss = (
            evaluate_loss(model, val_baskets, tokenizer, batch_size=batch_size, max_len=max_len)
            if val_baskets is not None
            else None
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "lr": round(current_lr, 6),
        })

    return history


def save_model_checkpoint(model: SmallCausalSKUModel, path: str | Path) -> None:
    """Save model state dictionary to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def load_model_checkpoint(model: SmallCausalSKUModel, path: str | Path) -> SmallCausalSKUModel:
    """Load model weights from disk."""
    path = Path(path)
    model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
    model.eval()
    return model


def milestone_2(epochs: int = 6) -> dict[str, Any]:
    """Milestone 2: Continue pretrain a small language model on these SKU sequences."""
    catalog = DEFAULT_CATALOG
    tokenizer = Tokenizer(catalog)
    transactions = generate_mock_transactions(n_transactions=1200, catalog=catalog, seed=42)
    train_b, val_b, _ = split_dataset(transactions, seed=42)

    model = SmallCausalSKUModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        max_seq_len=24,
        dropout=0.05,
    )

    history = pretrain_small_lm(
        model=model,
        tokenizer=tokenizer,
        train_baskets=train_b,
        val_baskets=val_b,
        epochs=epochs,
        batch_size=32,
        lr=2e-3,
        seed=42,
    )

    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]

    # Save model checkpoint
    checkpoint_path = Path("data/forecaster_lm.pt")
    save_model_checkpoint(model, checkpoint_path)

    return {
        "status": "completed",
        "epochs": epochs,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduced": final_loss < initial_loss,
        "history": history,
        "checkpoint_saved": checkpoint_path.exists(),
        "model_parameters": sum(p.numel() for p in model.parameters()),
    }


# ============================================================================
# Milestone 3: Next-Item Prediction Logic Using the Trained Model
# ============================================================================

@dataclass
class NextItemPrediction:
    sku_id: str
    name: str
    category: str
    base_price: float
    unit: str
    confidence: float
    confidence_pct: str
    reorder_urgency: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComplementaryBundle:
    bundle_name: str
    items: List[str]
    total_price: float
    discounted_price: float
    savings: float
    expected_basket_uplift: str

    def to_dict(self) -> dict:
        return asdict(self)


def determine_reorder_urgency(confidence: float, category: str) -> str:
    """Categorize inventory reorder urgency based on model confidence and category."""
    fast_moving = {"Dairy", "Bakery", "Produce"}
    if confidence >= 0.20 or (confidence >= 0.12 and category in fast_moving):
        return "HIGH"
    elif confidence >= 0.08:
        return "MEDIUM"
    return "LOW"


def predict_next_items(
    model: SmallCausalSKUModel,
    tokenizer: Tokenizer,
    basket: Sequence[str] | str,
    top_k: int = 5,
    exclude_current: bool = True,
    quantities: Optional[Dict[str, int]] = None,
    temperature: float = 1.0,
) -> List[NextItemPrediction]:
    """Predict the most likely next items given a partial shopping basket.
    
    Supports:
    - Partial basket as list of names or verbalized string.
    - Filtering out already selected items.
    - Confidence scores & calibrated probabilities (Stretch Idea 2).
    - Incorporating item quantities (Stretch Idea 1).
    """
    model.eval()

    # Parse input basket if provided as string
    if isinstance(basket, str):
        items = parse_verbalized_basket(basket) if basket.startswith(PREFIX) else [
            x.strip() for x in basket.split(",") if x.strip()
        ]
    else:
        items = list(basket)

    # Filter to known items
    valid_items = [name for name in items if name in tokenizer.name_to_sku]

    # If basket is completely empty, use BOS token alone
    if not valid_items:
        prefix_tokens = [tokenizer.bos_token_id]
    else:
        prefix_tokens = [tokenizer.bos_token_id] + [tokenizer.token_to_id[name] for name in valid_items]

    # Truncate to max sequence length if needed
    prefix_tokens = prefix_tokens[-model.max_seq_len :]

    with torch.no_grad():
        input_tensor = torch.tensor([prefix_tokens], dtype=torch.long)
        logits = model(input_tensor)
        # Logits at the final position
        last_logits = logits[0, -1, :].clone()

        # Mask out special tokens
        special_ids = [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ]
        for sid in special_ids:
            last_logits[sid] = -float("inf")

        # Exclude items already in the basket (cross-sell constraint)
        if exclude_current:
            for name in valid_items:
                token_id = tokenizer.token_to_id.get(name)
                if token_id is not None:
                    last_logits[token_id] = -float("inf")

        # Stretch Idea 1: incorporate quantities (boost probability of complementary items)
        if quantities:
            for item_name, qty in quantities.items():
                if qty > 1 and item_name in tokenizer.name_to_sku:
                    item_cat = tokenizer.name_to_sku[item_name].category
                    # boost items in complementary categories
                    for cat_item in tokenizer.catalog:
                        if cat_item.category == item_cat and cat_item.name != item_name:
                            tid = tokenizer.token_to_id[cat_item.name]
                            if last_logits[tid] != -float("inf"):
                                last_logits[tid] += 0.15 * math.log(qty)

        # Scale by temperature and apply softmax
        scaled_logits = last_logits / max(1e-4, temperature)
        probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()

    # Extract catalog item probabilities
    item_scores: List[Tuple[str, float]] = []
    for item in tokenizer.catalog:
        tid = tokenizer.token_to_id[item.name]
        score = float(probs[tid])
        item_scores.append((item.name, score))

    # Sort descending
    item_scores.sort(key=lambda x: x[1], reverse=True)

    predictions: List[NextItemPrediction] = []
    for name, confidence in item_scores[:top_k]:
        sku_obj = tokenizer.name_to_sku[name]
        urgency = determine_reorder_urgency(confidence, sku_obj.category)
        predictions.append(
            NextItemPrediction(
                sku_id=sku_obj.sku_id,
                name=sku_obj.name,
                category=sku_obj.category,
                base_price=sku_obj.base_price,
                unit=sku_obj.unit,
                confidence=round(confidence, 4),
                confidence_pct=f"{confidence * 100:.1f}%",
                reorder_urgency=urgency,
            )
        )

    return predictions


def suggest_complementary_bundle(
    predictions: List[NextItemPrediction],
    basket: Optional[Sequence[str]] = None,
    discount_pct: float = 0.10,
) -> ComplementaryBundle:
    """Suggest an optimized complementary retail bundle from top predictions (Stretch Idea 3)."""
    if not predictions:
        return ComplementaryBundle(
            bundle_name="Store Favourites",
            items=[],
            total_price=0.0,
            discounted_price=0.0,
            savings=0.0,
            expected_basket_uplift="0%",
        )

    # Pick top 2 or 3 predicted items for the bundle
    bundle_items = predictions[:3]
    item_names = [p.name for p in bundle_items]
    raw_total = sum(p.base_price for p in bundle_items)
    discounted = raw_total * (1.0 - discount_pct)
    savings = raw_total - discounted

    # Determine theme name based on dominant category
    categories = [p.category for p in bundle_items]
    if "Dairy" in categories and "Bakery" in categories:
        theme = "Fresh Morning Essentials"
    elif "Pantry" in categories:
        theme = "Chef's Pantry Bundle"
    elif "Snacks" in categories or "Beverages" in categories:
        theme = "Party & Snack Combo"
    else:
        theme = "Smart Restock Bundle"

    return ComplementaryBundle(
        bundle_name=theme,
        items=item_names,
        total_price=round(raw_total, 2),
        discounted_price=round(discounted, 2),
        savings=round(savings, 2),
        expected_basket_uplift=f"+{round(discount_pct * 150, 0):.0f}%",
    )


def milestone_3(checkpoint_path: str = "data/forecaster_lm.pt") -> dict[str, Any]:
    """Milestone 3: Implement next-item prediction logic using the trained model."""
    catalog = DEFAULT_CATALOG
    tokenizer = Tokenizer(catalog)

    model = SmallCausalSKUModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        max_seq_len=24,
    )

    path = Path(checkpoint_path)
    if path.exists():
        load_model_checkpoint(model, path)
    else:
        # Fallback to train briefly if checkpoint doesn't exist
        m2_res = milestone_2(epochs=3)
        load_model_checkpoint(model, path)

    test_baskets = [
        ["Whole Milk 1L", "Corn Flakes Cereal 500g"],
        ["Penne Rigate Pasta 500g", "Classic Marinara Sauce 400g"],
        ["Dark Roast Coffee Beans 250g"],
    ]

    forecast_results = {}
    for b in test_baskets:
        preds = predict_next_items(model, tokenizer, b, top_k=4)
        bundle = suggest_complementary_bundle(preds, basket=b)
        forecast_results[verbalize_basket(b, shuffle=False)] = {
            "partial_basket": b,
            "top_predictions": [p.to_dict() for p in preds],
            "bundle_suggestion": bundle.to_dict(),
        }

    # Test with quantities (Stretch Idea 1)
    preds_with_qty = predict_next_items(
        model, tokenizer, ["Whole Milk 1L"], top_k=3, quantities={"Whole Milk 1L": 4}
    )

    return {
        "status": "completed",
        "sample_forecasts": forecast_results,
        "quantity_test_predictions": [p.to_dict() for p in preds_with_qty],
    }


# ============================================================================
# Stubs for Milestones 4-5 (Implemented in subsequent steps)
# ============================================================================

def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
