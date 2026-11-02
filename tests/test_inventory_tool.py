"""Tests for src/tools/inventory_tool.py: finding a product by SKU or by name.
The model may not know a SKU (the search did not retrieve it), so the tool resolves names itself."""
from src.tools import inventory_tool as tool


# ---------- the real inventory ---------------------------------------------------

def test_an_exact_sku_is_found_in_any_case():
    assert tool.find_product("TP-LAMP-009")[0] == "TP-LAMP-009"
    assert tool.find_product("tp-lamp-009")[0] == "TP-LAMP-009"


def test_a_product_is_found_by_its_full_name():
    sku, product, problem = tool.find_product("NightWatch Headlamp")
    assert (sku, product["name"], problem) == ("TP-LAMP-009", "NightWatch Headlamp", None)


def test_a_product_is_found_by_a_part_of_its_name_plural_or_not():
    for text in ("headlamp", "the headlamp", "Headlamps", "a NIGHTWATCH headlamp"):
        assert tool.find_product(text)[0] == "TP-LAMP-009", text
    assert tool.find_product("trekking poles")[0] == "TP-POLE-006"
    assert tool.find_product("the StormShield jacket")[0] == "TP-JKT-001"
    assert tool.find_product("StormShield Rain Jacket in M")[0] == "TP-JKT-001"      # the whole name plus extra words


def test_every_product_can_be_found_by_its_own_name():
    for sku, product in tool.INVENTORY.items():
        assert tool.find_product(product["name"])[0] == sku


def test_an_unknown_product_is_reported_not_guessed():
    for text in ("NOPE-1", "flux capacitor", "", "   ", None):
        sku, product, problem = tool.find_product(text)
        assert sku is None and product is None and "No product found" in problem


def test_check_availability_accepts_a_name_and_reports_the_real_sku():
    reply = tool.check_availability("headlamp")
    assert "NightWatch Headlamp (SKU TP-LAMP-009)" in reply and "$34.00" in reply


def test_check_availability_tells_the_model_when_it_cannot_find_the_product():
    assert tool.check_availability("TP-LMP-004").startswith("No product found")


def test_the_product_card_works_for_a_name_and_gives_the_canonical_sku():
    card = tool.get_product("poles")
    assert card["sku"] == "TP-POLE-006" and card["name"] == "TrekLight Trekking Poles"
    assert tool.get_product("flux capacitor") is None


# ---------- when two products could match ------------------------------------------

def test_several_matches_are_listed_instead_of_picking_one(monkeypatch):
    monkeypatch.setattr(tool, "INVENTORY", {
        "A-1": {"name": "Trail Jacket Light", "price": 1.0, "stock": 1, "sizes": ["M"]},
        "A-2": {"name": "Trail Jacket Heavy", "price": 2.0, "stock": 1, "sizes": ["M"]},
    })
    sku, product, problem = tool.find_product("trail jacket")
    assert sku is None and product is None
    assert "Several products match" in problem and "A-1" in problem and "A-2" in problem
    assert tool.get_product("trail jacket") is None
    assert tool.find_product("Trail Jacket Heavy")[0] == "A-2"      # a more exact name still resolves
