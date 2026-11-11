# LEAF/LENS · MLR EX08

**Open: https://kafcoo.github.io/mlr-ex08/**

One `index.html`, plain JavaScript, device-local ONNX inference. Start the rear camera or choose a photo; all four class probabilities update on the page. The QR is generated locally and points to this app's own public URL. No photograph is sent to a server.

## Current model

**MobileNetV3-small + geometric augmentation + label smoothing 0.05**, seed **0**, best epoch **14**. Variant mean dev score: **83.14 ± 0.66%** over seeds 0/1/2. Selected checkpoint: **82.76%**. These are development balanced accuracies, **not independent test or phone-camera accuracies**. The fixed 120-image historical test was not reopened for this new study. Physical-phone testing remains pending.

- [New networks: measured scores, model size, WASM speed and selection rationale](NETWORK_RESULTS.md)
- [Original 12-run data/augmentation study — archived](TRAINING_RESULTS.md)
- [Current release provenance and SHA-256](release_info.json)
- [Audit and remaining hand-in tasks](AUDIT.md)
- [Report draft](REPORT_DRAFT.md) · [Review draft](REVIEW_DRAFT.md)

Seven new variants were trained on identical data for 15 epochs: Task 3 LeafNet, BatchNorm, GAP, and MobileNet with full fine-tuning, frozen last-layer-only training, cosine scheduling, or smoothing. MobileNet had higher dev scores; custom networks were smaller. Browser timings are from a desktop Mac, not a phone. The original 12 measurements remain unchanged; their historical 79.17% test result belongs to a previous checkpoint.

## Camera and preprocessing

Open the HTTPS link in Safari on iPhone, tap **Start rear camera**, allow access and centre one leaf. **Choose photo** is the fallback. The camera stops in the background; tap Start to resume.

The winning class, its model score and uncertainty flag sit **directly below the viewfinder**. On small phones, starting the camera or choosing a photo enters a compact input view and scrolls to it once. The square preview scales with the available viewport height and safe areas, so the frame, top result and controls can be seen together. This addresses user-reported scrolling problems on iPhone 13 mini. [Layout tests](training/results/mobile_layout_checks.json) cover portrait with reduced Safari-sized heights and landscape; confirmation on the user's physical phone is pending. It is a usability improvement, with no model or accuracy change.

[model_info.json](model_info.json) defines classes `apple, cherry, chestnut, maple`, size **96**, mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`. The page crops the central square, resizes it to 96 × 96, divides RGB by 255 and standardizes each channel. Contiguous R/G/B planes form **[1,3,96,96] float32**. Stable softmax converts logits to probabilities. Metadata is also embedded in [leaf_model.onnx](leaf_model.onnx), as in Task 2e.

The uncertain flag uses top probability below 0.65 or top-two margin below 0.15. It is a heuristic, not a trained unknown class or calibrated confidence. The supplied reference is ambiguous for the new model: a numerical check may PASS while the display correctly says Uncertain.

## JPEG fixed

JPEG uploads now use the same pinned **`@jsquash/jpeg@1.6.0` MozJPEG/WASM decoder** in every browser, with EXIF orientation enabled. Tests compare exact Pillow pixels and all eight EXIF orientations. PNG/other supported formats use native decoding. Live camera frames are already decoded and do not need the JPEG codec.

The original [check_image.jpg](check_image.jpg) is unchanged. [check_expected.json](check_expected.json) comes from the newly selected PyTorch model and was checked against ONNX. **Reference & diagnostics** now requires absolute error below **0.00001** and actual **three-decimal agreement**. Both WebKit and Chrome pass; observed error is about **4.12 × 10⁻⁶**. There is no relaxed 0.002 tolerance or browser-specific expected result.

A 12MP JPEG upload was also checked; its display preview is capped at 1024 pixels while inference uses the original source. Large-image Canvas resize filters can still differ between browsers; exact reference parity applies to the supplied 96 × 96 JPEG. See the detailed limitations in the network report.

ONNX Runtime Web **1.20.1**, QR generator **1.4.4**, and the optional JPEG codec load from jsDelivr. Code/WASM downloads; images stay local. A single WASM thread, inline muted video and a user-initiated camera request support GitHub Pages/Safari. Model and metadata URLs carry a model revision to avoid mixing cached releases.

## Notebook and checks

The user-authorized corrections are minimal: Task 3d uses `aug_train_loader`; train loaders use fixed generators; the custom dataset filters image extensions. Task 2 writes safely into `exports/task2/`. The original notebook has a local backup. The corrected local notebook has fresh outputs from **30/30 code cells and 15/15 semantic checks**.

A separate read-only Task 4 cell loads the currently released PyTorch checkpoint and checks its reference probabilities against the website files, keeping it distinct from the original Task 2 demonstration model. Graphs and the confusion matrix are embedded in the executed notebook.

- [Notebook execution](training/results/notebook_audit.json)
- [Original study/current export integrity](training/results/integrity_checks.json)
- [New 21-run checks](training/runs/networks_v1/results/integrity_checks.json)
- [Task 3 network parity and frozen BN checks](training/runs/networks_v1/results/network_checks.json)
- [Browser/QR/privacy/layout checks](training/results/web_audit.json)
- [JPEG/EXIF tests](training/results/jpeg_decoder_checks.json)

Browser tests cover uploads, strict probabilities, denied-camera fallback, a simulated live stream, crop/RGB/NCHW, uncertainty, stale inference races, widths 320/390/768/1366, independent QR decoding and absence of uploads. A physical iPhone test is still for the student to perform.

## Reproduce and submit

See [training instructions](training/README.md). Training uses small PyTorch loops, copied Task 3 networks and no new training framework. Raw data, checkpoints, local notebooks, environments and ZIPs stay out of Git. The current local `ex08_for_agent.zip` contains the matching model, JSON and reference files.

Pages serves **main**, with no build step. `MLRteacher` has been invited. The student still needs a real phone screenshot and their own final `REVIEW.md` and `REPORT.md`, pushed within **48 hours of the lab**. The labelled drafts contain technical assistance and unfilled phone observations.
