def load_image_ids(description, filepath):
    """Load image IDs from a text file (one filename per line) and return a dict
    of those IDs to their values in description (description.get(image_id))."""
    des = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            k = line.rstrip("\n")
            if len(k) < 2:
                continue
            image_id = k.split(".", 1)[0]
            des[image_id] = description.get(image_id)
    return des

