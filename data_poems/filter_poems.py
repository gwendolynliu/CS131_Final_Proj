import csv
import re
import statistics
from collections import Counter

# ---- file paths ----
CSV_IN = "data_poems/PoetryFoundationData.csv"
CSV_OUT = "data_poems/filtered_poems.csv"

# word length limits -- we decided 10-250 words is a reasonable range
MIN_WORDS = 10
MAX_WORDS = 250

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines)


def clean_title(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def count_words(text: str) -> int:
    return len(text.split())


def parse_tags(tag_string: str) -> list[str]:
    parts = [p.strip() for p in tag_string.split(",")]
    merged: list[str] = []
    for part in parts:
        if part.startswith("&") and merged:
            merged[-1] = merged[-1] + " " + part
        elif part:
            merged.append(part)
    return merged



def is_layout_heavy(text: str) -> bool:
    """
    Returns True if more than 50% of lines are very short (< 3 words).
    This catches concrete/visual poetry where the layout is the point --
    those poems don't make sense as plain text for our purposes.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    short_lines = [line for line in lines if len(line.split()) < 3]
    return len(short_lines) / len(lines) > 0.5


# ---- load data ----

rows: list[dict] = []
with open(CSV_IN, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Loaded {len(rows)} poems.")


# ---- basic stats ----

print("\n" + "=" * 60)
print("BASIC STATS")
print("=" * 60)
print(f"Total poems: {len(rows)}")

# collect all unique tags across the whole dataset
all_tags: list[str] = []
for r in rows:
    for tag in parse_tags(r["Tags"]):
        all_tags.append(tag)

unique_tags = sorted(set(all_tags))
print(f"All unique tags ({len(unique_tags)} total):")
for tag in unique_tags:
    print(f"  {tag}")


# ---- filtering ----

removed_counts: dict[str, int] = {
    "too_long": 0,
    "too_short": 0,
    "duplicate": 0,
    "layout_heavy": 0,
}

kept: list[dict] = []
seen: set[tuple[str, str]] = set()

for r in rows:
    title = clean_title(r["Title"])
    text = clean_text(r["Poem"])
    poet = r["Poet"].strip()
    tags = r["Tags"].strip()
    wc = count_words(text)

    # filter 1: too long
    if wc > MAX_WORDS:
        removed_counts["too_long"] += 1
        continue

    # filter 2: too short (probably a fragment or bad data)
    if wc < MIN_WORDS:
        removed_counts["too_short"] += 1
        continue

    # filter 3: same title by the same poet = true duplicate
    dupe_key = (title.lower(), poet.lower())
    if dupe_key in seen:
        removed_counts["duplicate"] += 1
        continue
    seen.add(dupe_key)

    # filter 4: layout-dependent / concrete poetry
    if is_layout_heavy(text):
        removed_counts["layout_heavy"] += 1
        continue

    kept.append({
        "title": title,
        "author": poet,
        "text": text,
        "tags": tags,
        "word_count": wc,
    })


# ---- save output ----

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["id", "title", "author", "text", "tags", "word_count"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, poem in enumerate(kept):
        writer.writerow({"id": i, **poem})

print(f"Saved {len(kept)} poems to {CSV_OUT}")


# ---- final stats ----

total_removed = sum(removed_counts.values())
word_counts = [p["word_count"] for p in kept]

print("\n" + "=" * 60)
print("FILTER RESULTS")
print("=" * 60)
print(f"Poems kept   : {len(kept)}")
print(f"Poems removed: {total_removed}")
print()
print("Breakdown of removed poems:")
for reason, count in removed_counts.items():
    print(f"  {reason:<20}: {count}")

print("\nWord count distribution (kept poems):")
print(f"  Min    : {min(word_counts)}")
print(f"  Max    : {max(word_counts)}")
print(f"  Median : {statistics.median(word_counts):.1f}")
print(f"  Mean   : {statistics.mean(word_counts):.1f}")

# count tags in the filtered corpus
kept_tags: list[str] = []
for p in kept:
    for tag in parse_tags(p["tags"]):
        kept_tags.append(tag)

print("\nTop 20 most common tags (filtered corpus):")
for tag, count in Counter(kept_tags).most_common(20):
    print(f"  {count:>5}  {tag}")
