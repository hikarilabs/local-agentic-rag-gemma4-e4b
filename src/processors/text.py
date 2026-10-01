import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_into_sentences_with_offsets(text: str) -> list[dict]:
    paragraphs = re.split(r"\n\s*\n", text)
    sentences_data = []
    search_start_pointer = 0

    for paragraph in paragraphs:
        cleaned_paragraph = paragraph.replace("\n", " ")
        cleaned_paragraph = re.sub(r"[ \t]+", " ", cleaned_paragraph).strip()
        if not cleaned_paragraph:
            continue

        raw_sentences = _SENTENCE_BOUNDARY.split(cleaned_paragraph)
        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            match_start = text.find(s_clean[:15], search_start_pointer)
            if match_start != -1:
                match_end = match_start + len(s_clean)
                search_start_pointer = match_end
            else:
                match_start = search_start_pointer
                match_end = search_start_pointer + len(s_clean)

            sentences_data.append(
                {"text": s_clean, "start": match_start, "end": match_end}
            )

    return sentences_data
