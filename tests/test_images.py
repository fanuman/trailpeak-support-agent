"""Tests for src/core/images.py: telling a real photo from something pretending to be one.
No image library is needed: the checks read the first bytes of the decoded file."""
import base64

import pytest

from src.core.images import read_image_data_url, sniff_image_type

TYPES = ("image/jpeg", "image/png", "image/webp")
JPEG = b"\xff\xd8\xff\xe0" + b"fake jpeg body"
PNG = b"\x89PNG\r\n\x1a\n" + b"fake png body"
WEBP = b"RIFF\x24\x00\x00\x00WEBP" + b"fake webp body"
PDF = b"%PDF-1.7 not a picture"


def url(data, kind="image/jpeg"):
    return f"data:{kind};base64," + base64.b64encode(data).decode()


# ---------- sniff_image_type: what the bytes really are -----------------------------

def test_the_three_allowed_types_are_recognised_by_their_first_bytes():
    assert sniff_image_type(JPEG) == "image/jpeg"
    assert sniff_image_type(PNG) == "image/png"
    assert sniff_image_type(WEBP) == "image/webp"


def test_anything_else_is_not_an_image():
    for data in (PDF, b"GIF89a....", b"<svg xmlns='http://www.w3.org/2000/svg'/>", b"RIFF\x00\x00\x00\x00WAVEfmt ", b"", b"\xff\xd8"):
        assert sniff_image_type(data) is None, data


# ---------- read_image_data_url -----------------------------------------------------

def test_a_real_image_is_accepted_and_decoded():
    assert read_image_data_url(url(JPEG), 1000, TYPES) == ("image/jpeg", JPEG)
    assert read_image_data_url(url(PNG, "image/png"), 1000, TYPES) == ("image/png", PNG)
    assert read_image_data_url(url(WEBP, "image/webp"), 1000, TYPES) == ("image/webp", WEBP)


def test_the_declared_type_is_not_case_sensitive():
    assert read_image_data_url(url(JPEG, "IMAGE/JPEG"), 1000, TYPES)[0] == "image/jpeg"


def test_text_that_is_not_a_base64_data_url_is_rejected():
    for bad in ("", "hello", "https://example.com/photo.jpg", "data:image/jpeg,abc", "image/jpeg;base64,abc", "data:;base64,"):
        with pytest.raises(ValueError, match="data URL|must be one of"):
            read_image_data_url(bad, 1000, TYPES)


def test_a_type_that_is_not_allowed_is_rejected_even_if_the_bytes_look_fine():
    with pytest.raises(ValueError, match="must be one of"):
        read_image_data_url(url(b"GIF89a....", "image/gif"), 1000, TYPES)
    with pytest.raises(ValueError, match="must be one of"):
        read_image_data_url(url(JPEG, "image/jpeg"), 1000, ("image/png",))


def test_broken_base64_and_an_empty_payload_are_rejected():
    with pytest.raises(ValueError, match="not valid base64"):
        read_image_data_url("data:image/png;base64,!!!not base64!!!", 1000, TYPES)
    with pytest.raises(ValueError, match="empty"):
        read_image_data_url("data:image/png;base64,", 1000, TYPES)


def test_the_size_limit_counts_decoded_bytes():
    assert read_image_data_url(url(JPEG), len(JPEG), TYPES)[1] == JPEG          # exactly the limit is fine
    with pytest.raises(ValueError, match="larger than"):
        read_image_data_url(url(JPEG), len(JPEG) - 1, TYPES)


def test_a_file_pretending_to_be_an_image_is_rejected():
    with pytest.raises(ValueError, match="not an image"):
        read_image_data_url(url(PDF, "image/jpeg"), 1000, TYPES)                 # a PDF renamed as a photo
    with pytest.raises(ValueError, match="does not match"):
        read_image_data_url(url(PNG, "image/jpeg"), 1000, TYPES)                 # a PNG that says it is a JPEG


def test_error_messages_never_contain_the_photo():
    payload = base64.b64encode(PDF).decode()
    with pytest.raises(ValueError) as error:
        read_image_data_url(f"data:image/jpeg;base64,{payload}", 1000, TYPES)
    assert payload not in str(error.value)
