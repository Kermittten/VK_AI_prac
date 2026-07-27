import json


def extract_to_json(raw_text):

    try:
        return json.loads(raw_text)

    except Exception:
        return None