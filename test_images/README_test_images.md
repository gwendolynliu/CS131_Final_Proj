# Test Images

## Naming Convention

Files follow the pattern `{id}_{short_name}.jpg` (or `.jpeg`), where `id` is a 1-based integer matching the corresponding entry in `test_images_metadata.json`.

Example: `7_room_cozy.jpg` corresponds to entry `id: 7` in the metadata file.

## Metadata

Each image has a corresponding entry in `test_images_metadata.json` with the following fields:

- `id` — integer identifier matching the filename prefix
- `filename` — exact filename in this directory (use this, not a reconstructed path)
- `category` — broad visual category (Landscape, Urban, Interior Space, Human, Object, Scene)
- `description` — short plain-language description of the image content
- `expected_moods` — list of mood/affect labels used for evaluation
- `scene_type` — either `dominant_subject` (one clear focal element) or `multi_object` (several elements of roughly equal visual weight)
