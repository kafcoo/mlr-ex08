# Návrh podkladů pro REVIEW.md

**Není to hotové REVIEW.md. Doplň vlastní skutečný telefonní test a formuluj závěry vlastními slovy. Zadání chce v review čtyři konkrétní kontroly; experimentální výsledky patří hlavně do reportu.**

## 1. Funguje kamera na telefonu?

- URL: https://kafcoo.github.io/mlr-ex08/
- **DOPLNIT:** telefon/OS, spuštění zadní kamery, povolení, reakce na list, zastavení a opětovné spuštění.
- **DOPLNIT skutečný screenshot telefonu**, ne obrázek z emulovaného desktopového WebKitu.
- Agent ověřil Chrome se simulovaným živým streamem a WebKit s fotografiemi; to nenahrazuje moje telefonní pozorování.

## 2. Stejné pravděpodobnosti jako v notebooku?

- Vyber `check_image.jpg` přes Choose photo a otevři Reference & diagnostics.
- **DOPLNIT:** tabulku všech čtyř hodnot ze svého telefonu vedle `check_expected.json`, zaokrouhlení na tři desetinná místa a maximální odchylku.
- Problém byl v různých nativních JPEG dekodérech Safari a Pillow. Oprava používá stejný MozJPEG/WASM dekodér na zařízení. Původní JPEG se neměnil a žádné hodnoty se nepřepisují podle prohlížeče.
- Automatické testy nyní vyžadují odchylku pod 1e-5 i třídesetinnou shodu v Chrome i WebKitu. Ověřena je také orientace fotografií EXIF 1–8.
- Očekávané hodnoty patří ke konkrétnímu aktuálnímu modelu; po novém tréninku se legitimně změnily. Neopisuj staré hodnoty z předchozího reportu.

## 3. Odpovídá preprocessing modelu?

- V `index.html` najdi `preprocess`: kratší rozměr určuje centrální čtverec, `drawImage` ho převzorkuje na 96 × 96, RGB se dělí 255 a standardizuje mean/std.
- Data se skládají po celých rovinách R, G, B do `[1,3,96,96]`. Porovnej s `model_info.json` a transform v notebooku.
- JPEG se nejprve lokálně dekóduje a zachová EXIF orientaci; kamera už poskytuje dekódované pixely a JPEG větev nepoužívá.
- **DOPLNIT vlastní vysvětlení**, co znamenají mean/std a proč nelze prohodit pořadí tříd nebo RGB/BGR.

## 4. Nic z obrazu neopouští zařízení?

- Ze své Pages domény se načítají model, JSONy a referenční obrázek. Z `cdn.jsdelivr.net` se načítají ONNX Runtime Web, QR knihovna a při JPEG nahrávání také jSquash/MozJPEG JS/WASM.
- Zvolený soubor se čte lokálně přes blob URL; kamera zůstává v MediaStream. Inference běží v lokálním WASM. QR vzniká v prohlížeči a odkazuje na vlastní Pages URL, nikoli na externí QR službu.
- Automatická síťová kontrola při analýze nenašla uploadové požadavky. **DOPLNIT, co jsem sám ověřil v kódu / síti.**

## Co jsme navíc zkusili

Pro report je připraveno 12 původních a 21 nových běhů. Nejvíc pomohla další data. ColorJitter neměl jednoznačný přínos. Nově pomohl BatchNorm vlastní síti; GAP šetřil velikost za cenu skóre. Předtrénovaný MobileNet byl přesnější, frozen-head varianta levnější na trénování, cosine měl menší a smoothing větší validační přínos. Přesná čísla a omezení jsou v REPORT_DRAFT.md a NETWORK_RESULTS.md; dev skóre nepovažuj za telefonní accuracy.

Finální REVIEW.md vytvoř až po svém ověření a přidej screenshot do veřejného repozitáře bez soukromých údajů. Deadline je 48 hodin od cvičení podle pokynů vyučujícího.
