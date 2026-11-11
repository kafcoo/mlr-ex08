# Leaf lens · MLR EX08

**Open the app: https://kafcoo.github.io/mlr-ex08/**

A single-page, plain JavaScript leaf classifier using an ONNX MobileNetV3-small model. Open the HTTPS link in Safari on an iPhone, tap **Start rear camera**, allow camera access, and hold a leaf in the centre of the square preview. Probabilities for apple, cherry, chestnut and maple update continuously. **Choose photo** provides a fallback. The camera stops when the page goes into the background; tap Start rear camera to resume.

## Improved model

The deployed model was selected after **12 training runs**: four controlled data/augmentation variants, each with three seeds. Cleaning the original and historical photos yielded **649 training images**, compared with the original 107, with separate development data and a final 120-image historical test. The selected variant is **`expanded_color`**, seed **2**.

On the same held-out historical test, the original deployed model scored **50.00%** and the selected checkpoint scored **79.17%**. This is a dataset result, not a measured physical-phone accuracy. Selection used development scores before looking at test results.

See [the full technical comparison](TRAINING_RESULTS.md) for all variants, seed spread, confusion matrix, cleaning decisions and limitations. See [training instructions](training/README.md) to reproduce the runs. This keeps the original notebook's MobileNet/PyTorch approach and adds no training framework.

## Camera, confidence and QR

The browser flags uncertainty when the highest probability is below **0.65**, or the gap between the top two is below **0.15**. It suggests another angle while still displaying every probability. This is a heuristic, not a trained unknown class: an unfamiliar species or unrelated object can still receive a high score.

The labelled QR code at the bottom is generated in the browser with `qrcode-generator` and encodes **https://kafcoo.github.io/mlr-ex08/**, also shown as a clickable link. No external QR image service is used. Photos and camera frames are processed locally and are never uploaded.

## Model and preprocessing

[index.html](index.html) loads [model_info.json](model_info.json) and [leaf_model.onnx](leaf_model.onnx). For every frame or uploaded photo it:

1. Crops the central square using the shorter image dimension.
2. Resizes that square to **96 × 96**, as specified by `input_size`.
3. Reads RGB channels, divides pixels by 255, and applies `(channel - mean) / std` using mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]` from the metadata.
4. Creates a float32 tensor with shape **[1, 3, 96, 96]**, with contiguous R, G and B planes (NCHW).
5. Runs the model and applies numerically stable softmax to its four output logits, preserving metadata class order.

ONNX Runtime Web **1.20.1** loads from jsDelivr. Single-threaded WebAssembly avoids requiring cross-origin isolation or SharedArrayBuffer on GitHub Pages. Inline, muted video, a user-initiated request and the `environment` facing-mode preference support Safari on iPhone. The QR library is also version-pinned. An internet connection is needed to load CDN dependencies and the model initially. The model uses ONNX opset **17** and is **6,097,434 bytes**.

## Reference check and browser verification

Tap **Check reference image** to classify [check_image.jpg](check_image.jpg) and compare every output against [check_expected.json](check_expected.json). Actual and expected probabilities, maximum absolute error and PASS/FAIL are shown. The native-JPEG absolute tolerance is **0.002** per probability (0.2 percentage points). A separate identical-RGB-pixels check uses a strict **0.00001** tolerance.

The reference JPEG is unchanged. Its expected probabilities were regenerated from the selected PyTorch checkpoint and checked against ONNX Runtime. The old model and its old expected values remain in Git history. Reference probabilities change when the model changes.

| Class | Chrome probability | WebKit probability | Expected probability |
| --- | ---: | ---: | ---: |
| apple | 0.81026398 | 0.81144060 | 0.8102642298 |
| cherry | 0.12601460 | 0.12568796 | 0.1260144562 |
| chestnut | 0.02314719 | 0.02289184 | 0.0231471751 |
| maple | 0.04057423 | 0.03997961 | 0.0405741669 |

Maximum native-JPEG absolute errors were **2.49e-07** in Chrome and **0.00118** in WebKit. Safari decodes the JPEG into slightly different RGB pixels than Pillow/Chrome. When both browsers receive exactly the same decoded RGB pixels as PyTorch, their probability error is just **2.50 × 10⁻⁷**. This isolates the larger WebKit difference to JPEG decoding. Both the reference button and actual file upload were checked. See [browser validation measurements](training/results/browser_validation.json).

Automated checks used Playwright WebKit with an iPhone 13 viewport and Chrome. They verified RGB channel order and normalization, central-square cropping, NCHW dimensions, all four probabilities, photo upload, camera-denial fallback and responsive layout. A simulated camera stream in Chrome verified repeated inference, the rear-camera preference and camera-track cleanup. The rendered QR was independently decoded with `jsQR` and matched the public app URL. A physical iPhone camera was not available for testing.

## Hosting

GitHub Pages serves the repository root from **main**. There is no build step. Keep the HTML, model, metadata and two reference-check files together. `.nojekyll` enables direct static serving.

Raw photos, training checkpoints, notebooks, ZIP archives, Python environments and local verification tools are excluded from Git. The public training scripts, audit and metrics document how the model was built.
