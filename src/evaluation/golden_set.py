# Each question lists the chunks a good answer MUST use, as "<file>#<position>".
# See the labels with: python -m src.evaluation.inspect_chunks [file]
# Labels must be re-checked whenever a file in data/ changes (positions shift).
# Prices and stock are not in the documents (they come from the inventory tool),
# so for price questions the relevant chunk is the one that names the product.

GOLDEN_SET = [
    {
        "query": "What is your return policy?",
        "expected_sources": ["returns_policy.txt"],
        "expected_chunks": ["returns_policy.txt#0", "returns_policy.txt#1"],
    },
    {
        "query": "Is the SummitCarry backpack in stock, and what does it cost?",
        "expected_sources": ["product_catalog.txt"],
        "expected_chunks": ["product_catalog.txt#4"],
    },
    {
        "query": "What is your warranty policy?",
        "expected_sources": ["warranty_policy.txt"],
        "expected_chunks": ["warranty_policy.txt#0", "warranty_policy.txt#1"],
    },
    {
        "query": "What tent works for winter camping?",
        "expected_sources": ["product_catalog.txt"],
        "expected_chunks": ["product_catalog.txt#2", "product_catalog.txt#3"],
    },
    {
        "query": "How much would the StormShield jacket and TrekLight poles cost together?",
        "expected_sources": ["product_catalog.txt"],
        "expected_chunks": ["product_catalog.txt#0", "product_catalog.txt#10"],
    },
]
