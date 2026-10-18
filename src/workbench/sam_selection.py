"""Keep explicit SAM counts; make an all-instances selection actually plural."""
import re


def detection_prompt(text, index):
    if index != -1:
        return text
    parts = [part.strip() for part in text.split(',') if part.strip()]
    # SAM's tokenizer treats an unqualified phrase as max_detections=1.
    return ', '.join(part if re.search(r':\s*[\d.]+\s*$', part)
                     else part + ':64' for part in parts)
