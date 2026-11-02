# src/rag/events.py
# What the UI's "trail" says. Each builder returns one (event_name, data) pair for answer_stream().


# tool name -> (label while running, label when finished), for the trail the UI shows
TOOL_LABELS = {
    "CheckAvailability": ("Checking live stock", "Checked live stock"),
    "CalculateTotal": ("Adding the prices", "Added the prices"),
}


IMAGE_STEP_DETAIL_CHARS = 90   # how much of the photo's description the trail shows


def search_step(label, detail="", status="done"):
    return "step", {"id": "search", "label": label, "detail": detail, "status": status}


def rewrite_step(detail="", status="done"):
    """Shown only for a follow-up: the question is rewritten so it makes sense without the conversation."""
    label = "Reading your follow-up" if status == "running" else "Read your follow-up"
    return "step", {"id": "rewrite", "label": label, "detail": detail, "status": status}


def image_step(detail="", status="done"):
    """Shown only when a photo was sent: the photo is read and put into words, and those words
    are used to search. The trail shows a short version of them."""
    label = "Reading your photo" if status == "running" else "Read your photo"
    if len(detail) > IMAGE_STEP_DETAIL_CHARS:
        detail = detail[:IMAGE_STEP_DETAIL_CHARS].rstrip(" .,;:-") + "..."
    return "step", {"id": "image", "label": label, "detail": detail, "status": status}


def tool_step(request, args, status, result=""):
    running, finished = TOOL_LABELS.get(request.name, (request.name, request.name))
    if request.name == "CheckAvailability":
        detail = args.get("sku", "")
    else:
        detail = result if status == "done" else ""
    return "step", {"id": request.id, "label": running if status == "running" else finished,
                    "detail": detail, "status": status}
