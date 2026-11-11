# Task 4 report

I wanted better results on my phone and to see which changes helped. I used the Task 3 training loop with the pretrained Task 2 MobileNetV3-small baseline, keeping the 96 × 96 RGB input and ImageNet normalization.

First, I compared more data and augmentation in 12 runs with seeds 0, 1 and 2. Removing duplicates and conflicting labels left 649 training images instead of 107.

| Training variant | Historical test accuracy, mean ± SD (%) |
| --- | ---: |
| Original data, new runs | 47.22 ± 2.83 |
| Additional data | 86.11 ± 3.36 |
| Additional data + geometry | 87.50 ± 0.00 |
| Additional data + geometry + ColorJitter | 84.44 ± 3.87 |

The original Task 2 model scored 50.00%. Adding data made the biggest difference, though it also gave the model more training steps during the same 15 epochs. Geometric augmentation helped a little; ColorJitter did not clearly improve the results. After this comparison, I stopped using the historical test to judge new changes.

Next, I compared networks and training options in 21 runs using the same 649 images, geometric augmentation, 15 epochs and seeds 0, 1 and 2. I selected the best epoch and model on development data only.

| Model / change | Development score, mean ± SD (%) |
| --- | ---: |
| Task 3 LeafNet | 68.46 ± 1.48 |
| LeafNet + BatchNorm | 73.68 ± 0.66 |
| LeafNet + BatchNorm + global average pooling | 69.99 ± 3.16 |
| MobileNet, full fine-tuning | 80.35 ± 0.80 |
| MobileNet, last layer only | 79.48 ± 0.29 |
| MobileNet + cosine scheduler | 81.36 ± 0.41 |
| MobileNet + label smoothing 0.05 | 83.14 ± 0.66 |

The development score averages balanced accuracy across the fixed clean and camera-like development views. The ± values show variation between seeds, not uncertainty on new phone photos. BatchNorm helped LeafNet. Global average pooling made it smaller but lowered its score. MobileNet performed better, but it also had the advantage of pretraining. Training only its last layer saved training work without making inference cheaper. Label smoothing improved the average score by 2.79 points over full fine-tuning, more than the variation between these seeds.

I chose MobileNet with label smoothing because it had the best average development score. The selected model used seed 0 and epoch 14, scored 82.76%, and takes about 6.1 MB in ONNX format. Full results are in [NETWORK_RESULTS.md](NETWORK_RESULTS.md) and [TRAINING_RESULTS.md](TRAINING_RESULTS.md).

I tested the app in Safari on my iPhone 13 mini with iOS 27.2. I used the rear camera to view 15 leaf photos on my laptop screen, choosing pictures the model had not seen. The top class was correct for 14/15 photos (93.3%). One chestnut was classified as apple. This small trial used photos on a screen, so it does not estimate accuracy on physical leaves. High probability did not guarantee correctness.

Camera permission, stopping and restarting worked. The first load was slow, but recognition felt immediate afterwards. Later openings also felt fine; I did not measure the times. Having the result directly below the camera made the app easier to use. My next step would be to try physical leaves in different lighting.

I also loaded my ONNX model in the course classifier and tested `check_image.jpg`; it predicted apple. My own app passed the reference comparison to three decimals (maximum error 4.122e-6). Phone screenshots and the reference values are in [REVIEW.md](REVIEW.md).
