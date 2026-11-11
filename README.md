# LEAF/LENS · MLR EX08

**App: https://kafcoo.github.io/mlr-ex08/**

A leaf classifier in one `index.html`, using plain JavaScript and ONNX Runtime Web. Open the app, allow the rear camera and centre a leaf, or use **Choose photo**. All four class probabilities are shown. Inference and QR generation run locally; images are not uploaded.

## Model and results

The app uses **MobileNetV3-small with geometric augmentation and label smoothing 0.05**, seed 0, best epoch 14. Its ONNX file is about 6.1 MB. The variant scored **83.14 ± 0.66%** across three seeds; the selected checkpoint scored **82.76%**. These are development balanced accuracies, not independent phone-test accuracy.

I tested it in Safari on an **iPhone 13 mini, iOS 27.2**, using the rear camera to view 15 leaf photos on a laptop screen. The top class was correct for 14/15. Camera permission, stop and restart worked. The reference check passed to three decimals, with maximum error **4.122e-6**. This small trial does not measure accuracy on physical leaves outdoors.

- [Report](REPORT.md) and [review with phone screenshots](REVIEW.md)
- [Network comparison: 21 runs](NETWORK_RESULTS.md)
- [Original data/augmentation study: 12 runs](TRAINING_RESULTS.md)
- [Model provenance and SHA-256](release_info.json)

## Preprocessing and privacy

[model_info.json](model_info.json) specifies classes `apple, cherry, chestnut, maple`, input size 96, mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. The app crops a central square, resizes to 96 × 96, divides RGB by 255, normalizes each channel and forms a float32 tensor `[1,3,96,96]`. The same metadata is embedded in [leaf_model.onnx](leaf_model.onnx).

ONNX Runtime Web 1.20.1, QR generator 1.4.4 and the optional jSquash JPEG decoder 1.6.0 load from `cdn.jsdelivr.net`. Model and reference files load from this Pages site. JPEG decoding is local and shared across browsers to avoid native decoding differences. The QR points to this app's URL. The uncertainty flag is a heuristic, not calibrated confidence or an extra class.

## Reproduction

See [training/README.md](training/README.md) for data preparation, seeds, training and checks. The original 12 runs and the separate 21-run study are preserved. The historical test has already been examined; the newer study selects using fixed development views only.

The local `ex08_student.ipynb` contains the checked outputs. Task 2 exports to `exports/task2/`, keeping the website model separate. Raw images, checkpoints, notebook backups and internal audit notes remain local and ignored by Git. Published measurements alone are not resumable checkpoints.

## Submission status

The public repository is [kafcoo/mlr-ex08](https://github.com/kafcoo/mlr-ex08). Pages serves the root of `main`; `MLRteacher` has a pending collaborator invitation.

The required [REVIEW.md](REVIEW.md) and [REPORT.md](REPORT.md) are present. The submission includes the phone screenshots and additional laptop screenshots with all four probabilities. The assignment deadline is **48 hours after the lab**. The notebook itself is not a required hand-in file.
