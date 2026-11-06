# Counting steel scaffold planks in photos

Test task for Ready to Bloom. Short plain-language report: [`REPORT.md`](REPORT.md). 

## Results (planks per image, front stacks only)
| Image | Correct | Idea 1 | Idea 2 (B2) | Idea 3 (conf 0.25) | Idea 3 (conf 0.15) |
|---|---|---|---|---|---|
| 1 | ~57 | 47 | 46 | 55 | 57 |
| 2 | ~23 | 23 | 33 | 14 | 21 |
| 3 | 72 | 70 | 70 | 44 | 59 |
| 4 | ~15 | 14 | 14 | 13 | 15 |
| 5 | 60 | 65 | 61 | 61 | 61 |
| **average miss** | | 3.6 | 5.0 | 8.4 | 3.2 |


## Run it
```bash
pip install -r requirements.txt
python src/prepare_images.py /folder/with/the/5/originals   # makes images/img1.png ... img5.png
python src/baseline.py          # Idea 1
python src/dp_counter.py B2     # Idea 2 (also try B1, B3)
python src/line_variant.py      # counting along a thin line down each stack
```
Run these from the repo root. Stack regions are hand-chosen in `src/baseline.py` (`ROIS`). For Idea 3, open the notebook in Google Colab (GPU), upload a zip made of `images/` plus the label files in `data_labels/` (copied into a `labels/` folder), and run the cells in order.

## Confidentiality
The test images are confidential. They are **not** in this repo, and `.gitignore` blocks `images/`, `private_results/` and any `out_*.png` (these show the photos). Keep the GitHub repo **private** and share it only with Kotone. The marked-up result pictures are in `private_results/`, for sending to her directly (for example in the zip).
