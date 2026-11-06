# Counting steel scaffold planks in photos

Test task for Ready to Bloom. Short plain-language report: [`REPORT.md`](REPORT.md). Long technical version: [`docs/REPORT_DETAILED.md`](docs/REPORT_DETAILED.md).

## In plain words
- **The job:** look at a photo of stacked steel planks and say how many planks are in each stack.
- **What the photos show:** the *ends* of flat steel planks piled like pancakes. Each end is a flat rectangle with a small hole. So the count is "how many pancakes in the pile", not "how many round pipes". (If pipes were meant, please send pipe photos.)
- **Idea 1, no learning (`src/baseline.py`):** look down a pile from top to bottom. Every time one plank ends and the next begins there is a thin bright/dark edge. Count the edges, plus one.
- **Idea 2, smarter edge counting (`src/dp_counter.py`):** planks lower in a photo look thinner because of perspective. This version lets the plank thickness change smoothly while it counts.
- **Idea 3, learning (`notebooks/scaffold_yolo_colab.ipynb`):** teach a small neural network (YOLOv8n) to draw a box around every plank end, then count the boxes. With only 5 photos it was tested by hiding one photo, training on the other four, and counting the hidden one. Done five times.
- **iPhone:** the trained network was exported to Core ML (`models/best.mlpackage`, 6.2 MB). It has **not** been run on a phone yet.

## Results (planks per image, front stacks only)
| Image | Correct | Idea 1 | Idea 2 (B2) | Idea 3 (conf 0.25) | Idea 3 (conf 0.15) |
|---|---|---|---|---|---|
| 1 | ~57 | 47 | 46 | 55 | 57 |
| 2 | ~23 | 23 | 33 | 14 | 21 |
| 3 | 72 | 70 | 70 | 44 | 59 |
| 4 | ~15 | 14 | 14 | 13 | 15 |
| 5 | 60 | 65 | 61 | 61 | 61 |
| **average miss** | | 3.6 | 5.0 | 8.4 | 3.2 |

Read this honestly: the simple edge counter (Idea 1) is as good as the neural network, the clean photos (3, 4, 5) work and the angled or staggered ones (1, 2) do not, and the 0.15 column was chosen after seeing the results on the same five photos, so it is optimistic. The network was trained on box labels that were **not hand-corrected** (they came from Idea 2). Details, failures and the iPhone plan are in `REPORT.md`.

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
