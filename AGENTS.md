# Project instructions

Keep our simple, readable code. Start from the Task 3 pipeline and add or change only what a variant needs — no rewrite, no new framework, classes or helper layers that are not necessary, no improvements nobody asked for. Every change must be small enough for the student to read and explain in their report. A fixed Task 2 pretrained baseline is also allowed by the experiment requirements; clearly identify which baseline is being compared.

- Preserve the student's original notebook. Review Task 3 and explain defects; do not silently replace the student's own solution. Put agent-generated technical findings in `AUDIT.md`, not in the student's `REVIEW.md` or `REPORT.md`.
- Keep the app as one `index.html` with plain JavaScript and CDN ONNX Runtime Web. Maintain rear-camera support, photo fallback, all four probabilities, device-local inference and the locally generated QR pointing to `https://kafcoo.github.io/mlr-ex08/`.
- Keep original data and raw historical data intact. Remove exact decoded-pixel duplicates across train/development/test and exclude ambiguous label groups rather than guessing labels.
- Compare one change at a time on the same fixed data and seeds, except when additional data is the factor being tested. Repeat important rows with seeds 0, 1 and 2. Save the best development checkpoint.
- Keep the original 12-run measurements intact when starting another experiment. Use a distinct run directory and record the data/protocol fingerprint. Published metrics alone are not a resumable checkpoint.
- The 120-image historical test has already been examined. Do not present it as a fresh independent test for future experiments. Select using development data; reserve a new phone/session test for final confirmation.
- Distinguish class probabilities from calibrated confidence, synthetic camera-like tests from physical-phone tests, and a JPEG tolerance check from exact agreement to three decimals. Do not report a test as passed unless its actual criterion passed.
- Export class/preprocessing metadata into ONNX as in Task 2e, and keep the ONNX file, JSON metadata and reference probabilities together. Verify graph validity and PyTorch/ONNX/browser agreement before deployment.
- Run the original notebook in a separate working directory: its Task 2 export cells overwrite root model/reference files if executed from the repository root.
