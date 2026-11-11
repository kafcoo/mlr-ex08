# Leaf lens · MLR EX08

**Open the app: https://kafcoo.github.io/mlr-ex08/**

A single-page, plain JavaScript leaf classifier using the supplied ONNX model. Open the HTTPS link in Safari on an iPhone, tap **Start rear camera**, allow camera access, and hold a leaf in the centre of the square preview. Probabilities for apple, cherry, chestnut and maple update continuously. **Choose photo** provides a fallback when camera access is unavailable. The camera stops when the page goes into the background; tap Start rear camera to resume.

The labelled QR code at the bottom is generated in the browser with `qrcode-generator` and encodes this app's own public URL, also shown as a clickable link. No external QR image service is used. Photos and camera frames are processed on the device and are never uploaded.

## Model and preprocessing

[index.html](index.html) loads [model_info.json](model_info.json) and [leaf_model.onnx](leaf_model.onnx). For every frame or uploaded photo it:

1. Crops the central square using the shorter image dimension.
2. Resizes that square to **96 × 96**, as specified by `input_size`.
3. Reads RGB channels, divides pixels by 255, and applies `(channel - mean) / std` using the metadata's mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
4. Creates a float32 tensor with shape **[1, 3, 96, 96]**, with contiguous R, G and B planes (NCHW).
5. Runs the model and applies numerically stable softmax to its four output logits, preserving metadata class order.

ONNX Runtime Web **1.20.1** loads from jsDelivr. Single-threaded WebAssembly avoids requiring cross-origin isolation or SharedArrayBuffer on GitHub Pages. Inline, muted video, a user-initiated camera request and the `environment` facing-mode preference support Safari on iPhone. The QR library is also version-pinned. An internet connection is needed to load these CDN dependencies and the model initially.

## Reference check

Tap **Check reference image** to classify [check_image.jpg](check_image.jpg) and compare every output against [check_expected.json](check_expected.json). The page shows actual and expected probabilities, the maximum absolute error, and a PASS/FAIL result. It uses an absolute tolerance of **0.0005** per probability to allow small browser JPEG-decoder and floating-point differences. This is 0.05 percentage points, not 0.05 probability.

The supplied image is already 96 × 96. A browser reference run produced:

| Class | Browser probability | Expected probability |
| --- | ---: | ---: |
| apple | 0.00122140 | 0.001221379498 |
| cherry | 0.99863268 | 0.998632729053 |
| chestnut | 0.00013527 | 0.000135266993 |
| maple | 0.00001065 | 0.000010653736 |

The model always predicts among its four known classes, even for an unrelated image; probabilities are not a guarantee of identification.

## Hosting

GitHub Pages serves the repository root from the **main** branch. There is no build step. Keep the HTML, model, metadata and two reference-check files together. `.nojekyll` enables direct static serving.

The repository intentionally excludes the training dataset, notebook, model-training checkpoint, source ZIP and local verification tooling.
