# src/core/images.py
#
# Checking an uploaded photo. The browser sends it as a "data URL":
#     data:image/jpeg;base64,/9j/4AAQ...
# Never trust what the sender says about it. This file decodes the text and looks at the first
# bytes of the real file, so a script or a renamed PDF can't get through as "image/jpeg".
# Plain functions, no third-party packages: easy to test.
import base64
import binascii


def sniff_image_type(data):
    """The real type of an image, read from its first bytes: 'image/jpeg', 'image/png',
    'image/webp', or None if it is none of those."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def read_image_data_url(value, max_bytes, allowed_types):
    """Check a data URL and return (type, bytes). Raises ValueError, with a message that is safe to
    show, if it is not a base64 image of an allowed type that fits in `max_bytes` once decoded."""
    header, comma, payload = value.partition(",")
    if not comma or not header.startswith("data:") or not header.endswith(";base64"):
        raise ValueError("the image must be a base64 data URL (data:image/jpeg;base64,...)")
    claimed = header[len("data:"):-len(";base64")].lower()
    if claimed not in allowed_types:
        raise ValueError(f"the image must be one of: {', '.join(allowed_types)}")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("the image is not valid base64") from None
    if not data:
        raise ValueError("the image is empty")
    if len(data) > max_bytes:
        raise ValueError(f"the image is larger than {max_bytes} bytes")
    actual = sniff_image_type(data)
    if actual is None or actual not in allowed_types:
        raise ValueError(f"the file is not an image of an allowed type ({', '.join(allowed_types)})")
    if actual != claimed:
        raise ValueError("the image's content does not match its declared type")
    return actual, data
