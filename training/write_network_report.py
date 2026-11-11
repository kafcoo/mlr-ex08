"""Measured architecture comparison and clearly labelled Czech report/review drafts."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".local-tools/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path("training/runs/networks_v1")
MOBILE_USABILITY = """
\n## Zlepšení použitelnosti: výsledek přímo u hledáčku

Uživatel při skutečném používání na **iPhonu 13 mini** upozornil, že pro přečtení vítězné třídy musel odscrollovat od kamery, a tím ztrácel přehled o záběru. Hlavní výsledek jsme proto přesunuli přímo pod hledáček: jméno nejvýše skórující třídy, procentní skóre a označení nejistoty se aktualizují společně s původními čtyřmi pravděpodobnostmi.

Mobilní režim po spuštění kamery nebo výběru fotografie skryje velký úvod a přesune pohled na vstupní panel. Čtvercový hledáček se vejde do dostupné výšky pomocí svh a safe-area insets, ovládání zachovává dotykové cíle a při otočení telefonu se rozložení přizpůsobí. Podrobné pravděpodobnosti a diagnostika zůstaly níže.

WebKit a Chrome prošly kontrolou rozměrů 375 × 812, 375 × 650, 375 × 600 a landscape 812 × 375; test ověřuje současnou viditelnost hledáčku, vítězné třídy a ovládání, přepínání vítěze i skutečně běžící simulovaný stream v Chrome. [Záznam](training/results/mobile_layout_checks.json). To není fyzické potvrzení opravy na telefonu: **DOPLNIT vlastní ověření nového rozložení a screenshot na iPhonu 13 mini**.

Jde o zlepšení použitelnosti, nikoli přesnosti. Síť, váhy, preprocessing a vypočtená skóre se touto úpravou nemění.
"""
LABELS = {"leaf_geo": "Task 3 LeafNet + geometrie", "leaf_bn_geo": "LeafNet + BatchNorm + geometrie",
          "leaf_gap_geo": "LeafNetBN + GAP + geometrie", "mobile_geo": "MobileNet: plný fine-tuning",
          "mobile_frozen": "MobileNet: jen poslední vrstva", "mobile_cosine": "MobileNet + cosine scheduler",
          "mobile_smoothing": "MobileNet + label smoothing 0,05"}


def main():
    selection = json.loads((RUN / "results/selection.json").read_text())
    evaluation = json.loads((RUN / "results/evaluation.json").read_text())
    profiles = json.loads((RUN / "results/profiles.json").read_text())
    browser = json.loads((RUN / "results/browser_profiles.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((RUN / "results").glob("*_seed*.json"))]
    chosen = next(r for r in evaluation["models"] if r["variant"] == selection["selected_variant"] and r["seed"] == selection["selected_seed"])
    chosen_run = next(r for r in rows if r["variant"] == chosen["variant"] and r["seed"] == chosen["seed"])
    assert len(rows) == 21 and all(set(r["metrics"]) == {"original_val", "historical_dev", "camera_original_val", "camera_historical_dev"} for r in evaluation["models"])
    table = []
    for profile in profiles:
        speed = next(r for r in browser["results"] if r["variant"] == profile["variant"] and r["browser"] == "webkit")
        score = f"{profile['mean_dev_score']*100:.2f} ± {profile['std_dev_score']*100:.2f}".replace(".", ",")
        parameters = f"{profile['parameters']:,}".replace(",", " ")
        trainable = f"{profile['trainable_parameters']:,}".replace(",", " ")
        size = f"{profile['onnx_bytes']/1e6:.3f}".replace(".", ",")
        timing = f"{speed['medianMs']:.1f}".replace(".", ",")
        table.append(f"| {LABELS[profile['variant']]} | {score} | {parameters} | {trainable} | {size} | {timing} |")
    table_text = "\n".join(table)
    ordered = sorted(profiles, key=lambda row: row["mean_dev_score"], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    y = np.arange(len(ordered))
    colours = ["#285a42" if p["variant"] == selection["selected_variant"] else "#91a58c" for p in ordered]
    axes[0].barh(y, [p["mean_dev_score"]*100 for p in ordered], xerr=[p["std_dev_score"]*100 for p in ordered], capsize=3, color=colours)
    axes[0].set_yticks(y, [LABELS[p["variant"]] for p in ordered])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Dev balanced accuracy (%) · mean ± SD, 3 seeds")
    axes[0].set_title("Fixed development data; no new test claim")
    axes[1].barh(y, [p["onnx_bytes"]/1e6 for p in ordered], color=colours)
    axes[1].set_xlabel("ONNX size (MB, decimal)")
    axes[1].set_title("Model size matters on a phone")
    for axis in axes:
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
    fig.savefig(RUN / "results/comparison.png", dpi=170)
    plt.close(fig)

    report = f"""# Nové srovnání sítí a trénovacích variant

Technická měření agenta, 11. 11. 2026. **21 nových běhů, 7 variant, 3 seedy. Nejde o nový nezávislý test ani o měření telefonu.** Původních 12 běhů zůstává v [archivním reportu](TRAINING_RESULTS.md).

## Protokol a férovost

Všechny nové varianty dostaly stejných 649 trénovacích fotografií, 15 epoch, batch 32, Adam s LR 0,001 a stejnou geometrickou augmentaci: crop 0,7–1,0, horizontální/vertikální flip a rotaci ±20°. ColorJitter je v této nové studii vypnutý, aby neměnil další faktor. Každá varianta začíná znovu pro seedy 0, 1, 2. Udržuje se nejlepší dev checkpoint.

Dev skóre je rovnoměrný průměr čtyř balanced accuracies: původní validace 29 snímků, historický dev 137 snímků a předem vytvořená camera-like kopie každé. Tyto pohledy nejsou nezávislé. Fotografický obsah je napříč variantami stejný; inicializace a transformace jsou řízené seedem. [Protokol a hash dat](training/runs/networks_v1/protocol.json) a [detailní metriky](training/runs/networks_v1/results/evaluation.json) jsou uložené.

Historická sada 120 snímků je již známá z první studie. Nový evaluator běžel s `--dev-only`: nová měření vůbec neobsahují `test` ani `camera_test`. **Pro závěrečné ověření nového modelu je nutná nová telefonní/session sada.**

## Výsledky a cena modelu

| Varianta | Dev skóre % (průměr ± SD) | Parametrů celkem | Trénovatelných | ONNX MB | WASM medián ms |
| --- | ---: | ---: | ---: | ---: | ---: |
{table_text}

![Srovnání dev skóre a velikosti](training/runs/networks_v1/results/comparison.png)

WASM časy jsou z desktopového Macu ve WebKitu, jeden thread, 5 warm-upů a 20 měření, nikoli z iPhonu. WebKit má hrubší rozlišení hodin; hodnoty kolem 2–3 ms se nesmí interpretovat jako přesný telefonní benchmark. [Chrome i WebKit profily](training/runs/networks_v1/results/browser_profiles.json) navíc ověřují shodu každého modelu s PyTorch do 1e-5 na referenčním vstupu. Zmrazený MobileNet má stejný objem a inferenční práci jako plně trénovaný; šetří trénink, ne stažení modelu.

## Co přinesla která změna

- **BatchNorm** proti stejnému vlastnímu LeafNetu zvedl průměrný dev výsledek přibližně o 5,21 procentního bodu. Toto porovnání mění jen BN a vychází ze stejných Task 3 konvolučních bloků.
- **GAP** snížil počet parametrů z 208 644 na 65 284 a ONNX z přibližně 0,836 na 0,262 MB, ale zhoršil dev skóre oproti LeafNetBN asi o 3,69 bodu. V tomto rozpočtu není nejpřesnější. Většina konvoluční práce zůstala, takže zmenšení souboru neznamená stejně velké zrychlení inference.
- **Předtrénovaný MobileNet** překonal vlastní sítě trénované od nuly. Není to čistý test samotné architektury: současně využívá znalosti z ImageNetu. Praktický závěr platí pro tyto dvě celé trénovací strategie, ne pro tvrzení, že vlastní konvoluční síť nikdy nemůže vyhrát.
- **Zmrazení** trénuje jen 4 100 parametrů poslední vrstvy; všechny ostatní váhy, BN statistiky i dropout zůstávají v eval režimu. Proti plnému fine-tuningu je průměr asi o 0,87 bodu nižší. To může být užitečný kompromis při omezeném tréninkovém výkonu.
- **Cosine scheduler** proti konstantnímu LR přidal asi 1,00 bodu. Je to menší efekt než zisk z dat v první studii. TMax=15 a počet epoch zůstávají stejné; neměnili jsme současně délku tréninku.
- **Label smoothing 0,05** proti základnímu fine-tuningu přidal asi 2,79 bodu, při stejné architektuře a velikosti souboru. Rozdíl převyšuje zaznamenané SD mezi těmito třemi seedy, ale to samo není test statistické významnosti ani důkaz lepší kalibrace pravděpodobností. Trénovací loss se smoothingem není přímo srovnatelná s nesmoothovanou loss; eval loss zůstala stejná.

## Volba pro web

Podle průměru dev skóre byla vybrána **{selection['selected_variant']}** a podle předem určeného mediánu seed **{selection['selected_seed']}**, nejlepší epocha **{chosen_run['best_epoch']}**. Jeho konkrétní dev skóre je **{chosen['dev_score']*100:.2f}%**; průměr celé varianty je **83,14 %**. Dosud nasazený model měl na stejné dev definici **{selection['baseline_dev_score']*100:.2f}%**. Nenahrazujeme tato čísla historickou testovou accuracy 79,17 % z jiného checkpointu.

MobileNet ponechává rozumný přibližně 6,1MB soubor a ve zdejším desktopovém WASM benchmarku nevytváří zásadní latenci. Vlastní BN/GAP síť by byla vhodný kandidát, kdyby prioritou byla velikost a nové telefonní měření ukázalo přijatelnou přesnost. Před další změnou volby je potřeba skutečný telefonní test, ne vybírání podle již známého historického testu.

## Notebook a JPEG

Uživatelem schválená oprava Task 3d používá `aug_train_loader`, oba train loadery mají vlastní pevný generátor a vlastní Dataset filtruje obrázkové přípony. Modely z Task 3 jsou v [networks.py](training/networks.py) přímo a čitelně zkopírované; [behaviorální kontrola](training/runs/networks_v1/results/network_checks.json) porovnává jejich výstupy se skutečnými třídami z notebooku. Původní notebook je lokálně zálohován. Opravený notebook prošel 30/30 buňkami a 15/15 věcnými kontrolami; Task 2 export je oddělen do `exports/task2/`.

JPEG fotografie dekóduje jednotný lokální WASM [jSquash/MozJPEG](https://github.com/jamsinclair/jSquash/tree/main/packages/jpeg) verze 1.6.0. Původní JPEG a referenční pixely se nemění; nepoužíváme browser-specifické očekávané hodnoty. Reference prošla přísnou tolerancí 1e-5 a shodou na tři desetinná místa ve WebKitu i Chrome. Všech osm EXIF orientací navíc dává přesné stejné pixely jako Pillow + exif_transpose. Kamera tuto JPEG větev nepotřebuje. Knihovna i její WASM se načítají z CDN; samotné fotografie se nikam neposílají.

Ověřen je i 12MP JPEG upload. Náhled je omezen na 1024 × 1024 pro úsporu paměti; inference dál používá původní obraz a centrální 96px výřez. U větších fotografií stále probíhá převzorkování přes Canvas: jeho high-quality filtr se může mezi prohlížeči lišit. Shoda dekódovaných JPEG pixelů proto není tvrzením, že libovolná zmenšená fotografie musí mít identické skóre napříč prohlížeči. Přísná reference 96 × 96 tento další resize nepotřebuje. Sjednocení resamplingu je případný další samostatně měřený krok, ne skrytá změna nynějších experimentů.

## Reprodukce

```sh
python training/train.py --run-dir training/runs/networks_v1 --variants mobile_geo mobile_frozen mobile_cosine mobile_smoothing leaf_geo leaf_bn_geo leaf_gap_geo --seeds 0 1 2 --epochs 15
python training/evaluate.py --run-dir training/runs/networks_v1 --baseline training/runs/networks_v1/checkpoints/baseline_before_networks.onnx --baseline-sha256 22bde880732116f01725c7b7d9bd58f14d7a5be62be593d3ad71657ebd9ad64a --dev-only
python training/export_model.py --run-dir training/runs/networks_v1 --output-dir exports/candidate
```

Pro čerstvou reprodukci použij nový run-dir. Původní baseline této studie lze obnovit z commitu `0303611e72702491b5e38114620f30c72a9ea7c5`. Checkpointy nejsou publikované; samotné veřejné JSONy trénink nenahrazují. MPS nemusí být bitově reprodukovatelné na jiném hardware. Automatické technické podklady: [návrh reportu](REPORT_DRAFT.md), [návrh review](REVIEW_DRAFT.md), [aktuální model](release_info.json).
"""
    Path("NETWORK_RESULTS.md").write_text(report + MOBILE_USABILITY)

    draft = f"""# Návrh REPORT.md — Task 4

**Technický návrh připravený agentem, ne hotové studentské odevzdání. Projdi a uprav text vlastními slovy a doplň telefonní pozorování. Bez potvrzení telefonu není závěrečný provozní výsledek hotový.**

## Cíl a základní pipeline

Cílem bylo zlepšit klasifikaci listů na telefonu a rozlišit, co pomohlo. Základ tvoří stejný trénovací loop jako v Task 3 a předtrénovaný MobileNetV3-small z Task 2 jako povolený baseline. Zachovali jsme vstup 96 × 96 RGB, dělení 255, ImageNet mean/std a čtyři výstupní logity. Přidávali jsme vždy jednu změnu; aplikace zůstala jedním HTML souborem s JavaScriptem a ONNX Runtime Web.

## První studie: data a augmentace

První studie měla 12 běhů, čtyři varianty a tři seedy. Po odstranění přesných duplicit a vyřazení sporných popisků bylo k dispozici 649 trénovacích snímků namísto původních 107. Tabulka zachycuje původní historické testové výsledky; tento test po jejím zhlédnutí už nepoužíváme jako nový test dalších změn.

| Varianta | Historický test % | Camera-like historický test % |
| --- | ---: | ---: |
| Původní nasazený Task 2 checkpoint | 50,00 | 50,00 |
| Nové běhy pouze s původními daty | 47,22 ± 2,83 | 47,22 ± 4,10 |
| Přidání historických trénovacích dat | 86,11 ± 3,36 | 84,72 ± 1,42 |
| Navíc geometrické augmentace | 87,50 ± 0,00 | 86,11 ± 0,39 |
| Navíc ColorJitter | 84,44 ± 3,87 | 86,39 ± 0,79 |

Největší praktický přínos mělo přidání dat. Při stejných 15 epochách ovšem větší dataset dostal také více aktualizací vah (60 vs 315), takže tento experiment sám neodděluje efekt rozmanitosti dat od výpočetního rozpočtu. ColorJitter nepřinesl jednoznačné zlepšení. Původně zvolený konkrétní checkpoint měl historickou testovou accuracy 79,17 %; průměr varianty a konkrétní nasazený seed nejsou totéž.

## Druhá studie: sítě a trénovací recepty

Dalších 21 běhů porovnalo následující varianty na stejných 649 snímcích s geometrickou augmentací, bez ColorJitter, s 15 epochami a seedy 0/1/2. Původní 29snímková validace a 137snímkový historický dev, v čistém i pevně vytvořeném camera-like pohledu, sloužily k výběru. Dev skóre je průměr jejich balanced accuracies, nikoli telefonní testová accuracy.

| Varianta | Dev % ± SD | Parametry | Trénovatelné | ONNX MB | Desktop WASM ms |
| --- | ---: | ---: | ---: | ---: | ---: |
{table_text}

Vlastní LeafNet a LeafNetBN odpovídají Task 3. BatchNorm pomohl, GAP výrazně zmenšil model, ale zde zhoršil skóre. Mobilní předtrénovaná síť byla přesnější; porovnání zahrnuje i efekt předtrénování, ne jen samotnou topologii. Zmrazená síť s tréninkem jen poslední vrstvy byla levnější na trénování, ale nemá menší model ani rychlejší inferenci. Cosine pomohl mírně, label smoothing v tomto nastavení více.

## Šum a omezení

Uváděná SD je rozptyl mezi třemi seedy, ne interval spolehlivosti přes nové fotografie. Jedna chyba původní validace je 3,45 procentního bodu, jedna chyba historického testu 0,83 bodu. Camera-like pohled je transformace týchž fotografií, ne další nezávislá sada. Rozdíl dev skóre ColorJitter/geometrie 0,23 bodu byl menší než seedový rozptyl; nepovažujeme ho za důkaz přínosu barevných změn. U nové studie smoothing zvedl průměr dev skóre asi o 2,79 bodu, ale nezávislý telefonní test ještě chybí.

## Volba modelu

Podle předem popsaného dev pravidla jsme zvolili MobileNetV3-small + geometrickou augmentaci + label smoothing 0,05, seed {selection['selected_seed']}, nejlepší epochu {chosen_run['best_epoch']}. Průměr varianty je 83,14 %, konkrétní seed {chosen['dev_score']*100:.2f} %. Soubor má přibližně 6,1 MB. Velikost i desktopový WASM čas jsou přijatelné, zatímco vlastní menší sítě ztrácely v dev skóre. Smoothing neznamená automaticky kalibrované pravděpodobnosti; upozornění „Uncertain“ je stále heuristika.

## Telefonní zkouška — DOPLNIT STUDENTEM

- Telefon: **iPhone 13 mini** (uvedl uživatel); verze Safari/OS: **DOPLNIT**.
- Datum, druhy listů, počet odlišných listů a světelné podmínky: **DOPLNIT**.
- Výsledek skutečné zadní kamery, nejčastější chyby a praktické zpoždění: **DOPLNIT**.
- Ověření vlastního ONNX v kurzové aplikaci na telefonu: **DOPLNIT**.
- Odkaz na screenshot a závěr, zda bych tento model opravdu ponechal: **DOPLNIT**.

Pro příští potvrzení bych odložil novou telefonní sadu podle listu/rostliny a session. Historický test už je známý. Další změny bych zaváděl odděleně a nejdříve měřil přínos proti současnému dev baseline, ne přidával současně více optimalizací.

Podrobné metriky, grafy a protokol: [NETWORK_RESULTS.md](NETWORK_RESULTS.md) a [TRAINING_RESULTS.md](TRAINING_RESULTS.md). Před odevzdáním text vlastními slovy zkrať a zkontroluj, že odpovídá tomu, co umíš vysvětlit.
"""
    Path("REPORT_DRAFT.md").write_text(draft + MOBILE_USABILITY)

    review = """# Návrh podkladů pro REVIEW.md

**Není to hotové REVIEW.md. Doplň vlastní skutečný telefonní test a formuluj závěry vlastními slovy. Zadání chce v review čtyři konkrétní kontroly; experimentální výsledky patří hlavně do reportu.**

## 1. Funguje kamera na telefonu?

- URL: https://kafcoo.github.io/mlr-ex08/
- Telefon je podle uživatele **iPhone 13 mini**. **DOPLNIT:** OS, spuštění zadní kamery, povolení, reakce na list, zastavení a opětovné spuštění.
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
"""
    Path("REVIEW_DRAFT.md").write_text(review + MOBILE_USABILITY)
    print("Wrote NETWORK_RESULTS.md, REPORT_DRAFT.md and REVIEW_DRAFT.md")


if __name__ == "__main__":
    main()
