# Scaffold Counting Test Task: Draft Report

## 1. Task interpretation
The test images  show stacks of galvanized steel scaffold planks seen from their end faces, not tubes. I therefore count **planks (layers) per front stack**. Far-background and heavily occluded stacks are excluded.


## 2. Approach (no training data needed)
Each stack's end face is selected as a region of interest (in the app this would be a user tap or a detector box). For each region:
1. Blur, then take the vertical Sobel gradient and average across the width. This gives a row-by-row profile of horizontal-edge strength.
2. **Method A, peak picking:** find peaks with a minimum spacing taken from the profile's autocorrelation period. Count = peaks + 1.
3. **Method B, perspective-aware segmentation:** choose the set of boundaries by dynamic programming. Each boundary must sit on a strong edge, and plank height may only change smoothly from one plank to the next (layers get thinner toward the bottom because of perspective).

Code: `baseline.py` (A), `dp_counter.py` (B). Both run in about 3 to 6 ms (A) and 0.1 to 2.5 s (B, unoptimised Python) per image on CPU.

## 3. Results

Ground truth is hand-counted from zoomed crops of the images, counted twice independently . "~" marks images where stacks are cropped or partly hidden, so the count is approximate. Counts are for the **front stacks only**. Image 1 is the least certain: the first count gave about 50, the check gave about 57, and the bottoms of the stacks are cropped by the frame.

| Image | Correct count (front stacks) | A: peaks | B1: DP, loose | B2: DP, period-constrained | B3: DP, gradual |
|---|---|---|---|---|---|
| 1 | ~57 (about 15, 15, 14, 14; bottoms cropped, ±5) | 47 (8, 12, 14, 13) | 56 | 46 | 40 |
| 2 | ~23 (8, 8, 7) | 23 (4, 10, 9) | 34 | 33 | 33 |
| 3 | 72 (18 per column) | 70 (18, 17, 17, 18) | 82 | 70 | 75 |
| 4 | ~15 (5, 5, 5) | 14 (4, 5, 5) | 37 | 14 | 13 |
| 5 | 60 (20, 20, 20) | 65 (21, 22, 22) | 98 | 61 | 79 |

Image 3 was double-checked on a 3x zoom: about 18 clear planks per column, plus partly visible ones at the top and bottom edges, so 18 to 20 per column depending on how partial planks are treated. Image 2's total of 23 for method A is a coincidence: the per-stack counts (4, 10, 9) do not match the true (8, 8, 7).

### 3b. Line-based variant (matches the reference workflow)
Same two methods, but using only a thin vertical line (12% of the stack width, down the centre, away from the holes) instead of the full end face. Script: `line_variant.py`.

| Image | Correct | Line + peaks | Line + DP (B2 settings) |
|---|---|---|---|
| 1 | ~57 | 54 (13, 14, 14, 13) | 52 |
| 2 | ~23 | 27 (4, 15, 8) | 34 |
| 3 | 72 | 62 (14, 16, 18, 14) | 67 |
| 4 | ~15 | 14 | 13 |
| 5 | 60 | 64 | 60 (20, 20, 20) |

The line improves image 1 (47 to 54) and gives an exact total on image 5, but makes image 3 worse (70 to 62 or 67). Results are mixed, so the full-width profile stays as the main method and the line variant is reported alongside it.

## 4. What worked and what did not
- **Total counts from method B2 are within about 1 to 2 planks on images 3, 4 and 5** (70 against 72, 14 against about 15, 61 against 60). Method A is within about 1 to 5 on the same images. This is better than expected from a method with no training.
- **Per-stack counts are much worse on the hard images.** On image 1, stack 1 reads 8 against about 12, and on image 2 the stacks read 4, 10, 9 against 8, 8, 7. So the total can look right while individual stacks are wrong.
- **The looser perspective-aware segmentation (B1, B3) double-counts**: it splits each plank into its top edge and its front edge, so image 5 gives up to about 40 per stack and image 4 gives 37 against 15. The period-constrained version (B2) removes this but is no better than plain peak picking (A) on image 3.
- **Method B2 is only marginally better than the simple method A** (closer on image 5, equal on images 3 and 4, worse on image 2). The added complexity bought little on this data, and the looser variants (B1, B3) are worse. Both A and B2 undercount image 1 by about 10 planks.
- **Images 1, 2 and 4 are the error cases**: oblique angle, staggered or partly hidden stacks, cropped bottoms, and long plank tops that dominate the view. The edge profile is not periodic there, and the hand-drawn regions include non-end-face texture.
- **Limits of this evaluation**: 5 images, hand-chosen regions, and parameters adjusted while looking at the same images. Treat the numbers as a diagnosis, not a benchmark.

## 5. Learned detector: YOLOv8n, leave-one-image-out

**Setup.** YOLOv8n (3.0 M parameters, pretrained on COCO), one class (`plank`), input 800 px. For each image, the model was trained on the other four (each repeated 16 times per epoch) for 60 epochs with mosaic, scale, rotation, perspective and colour augmentation, then used to predict the held-out image (confidence 0.25, NMS IoU 0.5). Only detections inside the front-stack regions were counted. These regions are slightly larger than those used for the classical methods (they are listed in the notebook), so the detector and classical counts are not on exactly identical regions. A final model was trained on all five images for export.

**Labels.** The box labels started as pre-labels generated by method B2 (stack region plus detected layer boundaries). The training log shows the label files for images 1 and 2 still had the original pre-label box counts (44 and 32), so those labels were **not corrected by hand**. Image 2's pre-labels were known to be wrong (32 boxes for about 23 planks). The detector was therefore trained partly on the classical method's own mistakes, and this is a limitation of the results below.

| Image | Correct | Method A | Method B2 | YOLOv8n (held out) | YOLO error |
|---|---|---|---|---|---|
| 1 | ~57 | 47 | 46 | 55 | -2 |
| 2 | ~23 | 23 | 33 | 14 | -9 |
| 3 | 72 | 70 | 70 | 44 | -28 |
| 4 | ~15 | 14 | 14 | 13 | -2 |
| 5 | 60 | 65 | 61 | 61 | +1 |
| **Mean absolute error** | | **3.6** | **5.0** | **8.4** | |

Inference time for YOLOv8n on a Tesla T4 GPU: 11 to 15 ms per image at 800 px (after warm-up). On the Colab CPU (Intel Xeon @ 2.00 GHz, PyTorch, one photo at 800 px, median of 10 runs) it took 174.5 ms per image. Core ML export succeeded (`best.mlpackage`, 6.2 MB, NMS included); it has not been run on a device, so phone speed is unmeasured.

**Confidence-threshold sensitivity (same held-out models, one threshold for all images).**

| Image | Correct | conf 0.10 | conf 0.15 | conf 0.25 (default) |
|---|---|---|---|---|
| 1 | ~57 | 59 (+2) | 57 (0) | 55 (-2) |
| 2 | ~23 | 23 (0) | 21 (-2) | 14 (-9) |
| 3 | 72 | 65 (-7) | 59 (-13) | 44 (-28) |
| 4 | ~15 | 19 (+4) | 15 (0) | 13 (-2) |
| 5 | 60 | 66 (+6) | 61 (+1) | 61 (+1) |
| **Mean absolute error** | | 3.8 | **3.2** | 8.4 |

The default threshold of 0.25 was fixed before any results were seen. The sensitivity run was added after seeing that image 3 was badly undercounted, and the best value (0.15) is chosen on the same five images it is scored on, so 3.2 is optimistic and is not a fair benchmark. With five images the gap between 3.2 (detector at 0.15) and 3.6 (method A) is too small to call either method better.

**Reading the result.**
- **At the default threshold the detector is clearly worse than the classical method** (mean error 8.4 against 3.6), mostly because of image 3 (44 against 72) and image 2 (14 against about 23).
- **The undercounting is largely a confidence problem, not a detection problem.** Lowering the threshold recovers most of the missing planks on images 2 and 3, so the model finds many thin or staggered planks but scores them low. At 0.15 the detector is exact on images 1 and 4 and within 1 on image 5.
- **No single threshold fits every image.** Lowering it improves images 2 and 3 but starts to overcount images 4 and 5 (at 0.10: +4 and +6). Image 3 is still 7 to 13 planks short even at 0.10 to 0.15.
- **The detector helps most where the classical methods fail:** image 1, where both classical methods undercount by about 10 and the detector at 0.15 is exact. It is no better than them on the clean images (4, 5).
- The mAP values printed during training were computed on a training image, not on the held-out one, so they are not reported.

**What would improve it (not done).** Correct the labels by hand (images 1 and 2 first), add more varied photos, tile the large images, and check the confidence threshold on image 3 as a sensitivity test (it was not tuned per image).

## 5b. Why YOLO, in simple terms
- **It counts by drawing a box on every plank end, then counting the boxes.** That is also how the reference app works for individual items.
- **Five photos is not as little as it sounds.** The five photos show about 230 plank ends in total, so the network sees hundreds of examples of what a plank end looks like, not just five pictures.
- **It starts smart.** YOLOv8n was already trained on millions of everyday photos, so it already knows edges, metal and rectangles. We only had to teach it one new thing: "this is a plank end".
- **It is small and quick.** About 3 million numbers (6.2 MB). It took 11 to 15 ms per image on the Colab GPU, and it exported to a Core ML file for iPhone with one command.
- **Other options were less suitable.** A network that guesses the total number straight from the whole photo would need far more photos. Bigger detectors would be slower on a phone with no clear gain here. The simple edge counter needs no training, and it was built as the yardstick YOLO had to beat.
- **It was not the winner.** On our five photos YOLO was about as good as the simple edge counter, not better, and it needed a lower confidence setting to find thin planks.
- **A possible problem (untested).** Planks in a stack are thin and their boxes overlap. The step that removes duplicate boxes may delete real neighbouring planks, which could explain the undercount on image 3. This was not tested.
- **Licence note.** As far as I know, Ultralytics YOLOv8 is released under the AGPL-3.0 licence, with a separate paid licence for commercial use. Before putting it in a commercial app, check the current terms, or use a detector with a more permissive licence.

## 5c. How the small amount of data was handled, in simple terms
- **A pretrained model:** the network started from what it had already learned, instead of from zero.
- **Augmentation:** every time a training photo was shown, it was randomly changed (zoomed, shifted, tilted a little, brighter or darker, mirrored, or stitched together with other photos), so the network saw many versions of the same few photos. Each photo was also shown 16 times per round so there were enough training steps.
- **A fair test:** each photo was counted by a network that had never seen it (train on four, count the hidden fifth, repeat five times).
- **A yardstick without training:** the edge-counting method needs no labels, and it showed how good the network really was.
- **Cheap labelling:** first-draft boxes were generated automatically from the edge counter to avoid drawing every box by hand. **They were not corrected afterwards**, so the network partly learned the edge counter's mistakes. This is the biggest weakness of the data approach.
- **Not tried (next steps):** cutting large photos into tiles, making fake stacks, collecting more photos, hand-correcting labels, and using the app's tap-to-correct feature to feed fixes back into training.

## 6. iOS plan
- Export the trained detector with `coremltools` (or the Ultralytics Core ML export), FP16 or int8, with non-maximum suppression included in the model. Run it through the Vision framework.
- Offer the reference app's interaction: the user drags a line or region over a stack, and the model counts inside it (a good fit for the line-based method above, and a way to avoid counting background stacks).
- Use tiling for high-resolution photos, and show the live count as an overlay.
- To work in real conditions, collect a few hundred varied site photos (angles, lighting, weather, rust, occlusion). Give users a capture guide (shoot square-on to the end faces). Add a confidence score and a tap-to-add/remove correction UI, and feed the corrections back into training.
- Measure inference time on a real device (Neural Engine against GPU) before committing to a model size.

## 7. Confidentiality
The test images are confidential and were used only for this task.
