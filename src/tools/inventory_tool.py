import json
from pathlib import Path
from pydantic import BaseModel, Field
from openai import pydantic_function_tool

INVENTORY_PATH = Path(__file__).parent / "inventory.json"

with open(INVENTORY_PATH) as f:
    INVENTORY = json.load(f)


class CheckAvailability(BaseModel):
    """Check the current price and stock level for a specific TrailPeak product, by its SKU or by its name."""
    sku: str = Field(..., description="The product SKU, e.g. TP-JKT-001. If you do not know the SKU, give the product name instead, e.g. NightWatch Headlamp")


# Words that carry no information when matching a product name ("the headlamp" -> "headlamp").
_FILLER = {"the", "a", "an", "and", "of", "for", "in", "your", "our", "my", "please", "any"}


def _words(text):
    """Lower-case words of `text`, without filler, with a trailing plural 's' removed ("poles" -> "pole")."""
    words = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _FILLER}


def find_product(text):
    """Look a product up by SKU or by name. Returns (sku, product, problem): exactly one of
    (sku, product) or `problem` is filled in, so the caller never has to guess.

    An exact SKU wins. Otherwise the name matches when all the words of one side appear in the other,
    and it must match exactly ONE product - if it matches several, or none, `problem` says so."""
    text = (text or "").strip()
    if text.upper() in INVENTORY:
        return text.upper(), INVENTORY[text.upper()], None

    wanted = _words(text)
    matches = []
    if wanted:
        for sku, product in INVENTORY.items():
            name = _words(product["name"])
            if wanted <= name or name <= wanted:
                matches.append((sku, product))
    if len(matches) == 1:
        return matches[0][0], matches[0][1], None
    if not matches:
        return None, None, f"No product found for '{text}'."
    options = "; ".join(f"{product['name']} (SKU {sku})" for sku, product in matches)
    return None, None, f"Several products match '{text}': {options}. Ask which one, or use its SKU."


def check_availability(sku: str) -> str:
    found_sku, product, problem = find_product(sku)
    if problem:
        return problem
    stock_status = "In stock" if product["stock"] > 0 else "Out of stock"
    return (
        f"{product['name']} (SKU {found_sku}): ${product['price']:.2f}, "
        f"{stock_status} ({product['stock']} units), "
        f"available sizes: {', '.join(product['sizes'])}"
    )


def get_product(sku: str):
    """Same lookup as check_availability, but returns data instead of a sentence,
    so the UI can draw a product card from it."""
    found_sku, product, problem = find_product(sku)
    if problem:
        return None
    return {"sku": found_sku, **product}


# Self-registration - rag/pipeline.py aggregates these rather than hand-
# assembling a tools list itself. A new tool file following this same
# TOOL_SCHEMA/TOOL_FUNCTION pattern is all that's needed to be picked up -
# relevant starting Day 16 (agents), where more tools are likely to appear.
TOOL_SCHEMA = pydantic_function_tool(CheckAvailability)
TOOL_FUNCTION = {"CheckAvailability": check_availability}
