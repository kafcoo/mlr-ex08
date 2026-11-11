# Návrh REPORT.md — Task 4

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
| LeafNet + BatchNorm + geometrie | 73,68 ± 0,66 | 208 644 | 208 644 | 0,836 | 2,0 |
| LeafNetBN + GAP + geometrie | 69,99 ± 3,16 | 65 284 | 65 284 | 0,262 | 2,0 |
| Task 3 LeafNet + geometrie | 68,46 ± 1,48 | 208 292 | 208 292 | 0,836 | 2,0 |
| MobileNet + cosine scheduler | 81,36 ± 0,41 | 1 521 956 | 1 521 956 | 6,098 | 3,0 |
| MobileNet: jen poslední vrstva | 79,48 ± 0,29 | 1 521 956 | 4 100 | 6,098 | 3,0 |
| MobileNet: plný fine-tuning | 80,35 ± 0,80 | 1 521 956 | 1 521 956 | 6,098 | 3,0 |
| MobileNet + label smoothing 0,05 | 83,14 ± 0,66 | 1 521 956 | 1 521 956 | 6,098 | 3,0 |

Vlastní LeafNet a LeafNetBN odpovídají Task 3. BatchNorm pomohl, GAP výrazně zmenšil model, ale zde zhoršil skóre. Mobilní předtrénovaná síť byla přesnější; porovnání zahrnuje i efekt předtrénování, ne jen samotnou topologii. Zmrazená síť s tréninkem jen poslední vrstvy byla levnější na trénování, ale nemá menší model ani rychlejší inferenci. Cosine pomohl mírně, label smoothing v tomto nastavení více.

## Šum a omezení

Uváděná SD je rozptyl mezi třemi seedy, ne interval spolehlivosti přes nové fotografie. Jedna chyba původní validace je 3,45 procentního bodu, jedna chyba historického testu 0,83 bodu. Camera-like pohled je transformace týchž fotografií, ne další nezávislá sada. Rozdíl dev skóre ColorJitter/geometrie 0,23 bodu byl menší než seedový rozptyl; nepovažujeme ho za důkaz přínosu barevných změn. U nové studie smoothing zvedl průměr dev skóre asi o 2,79 bodu, ale nezávislý telefonní test ještě chybí.

## Volba modelu

Podle předem popsaného dev pravidla jsme zvolili MobileNetV3-small + geometrickou augmentaci + label smoothing 0,05, seed 0, nejlepší epochu 14. Průměr varianty je 83,14 %, konkrétní seed 82.76 %. Soubor má přibližně 6,1 MB. Velikost i desktopový WASM čas jsou přijatelné, zatímco vlastní menší sítě ztrácely v dev skóre. Smoothing neznamená automaticky kalibrované pravděpodobnosti; upozornění „Uncertain“ je stále heuristika.

## Telefonní zkouška — DOPLNIT STUDENTEM

- Telefon: **iPhone 13 mini** (uvedl uživatel); verze Safari/OS: **DOPLNIT**.
- Datum, druhy listů, počet odlišných listů a světelné podmínky: **DOPLNIT**.
- Výsledek skutečné zadní kamery, nejčastější chyby a praktické zpoždění: **DOPLNIT**.
- Ověření vlastního ONNX v kurzové aplikaci na telefonu: **DOPLNIT**.
- Odkaz na screenshot a závěr, zda bych tento model opravdu ponechal: **DOPLNIT**.

Pro příští potvrzení bych odložil novou telefonní sadu podle listu/rostliny a session. Historický test už je známý. Další změny bych zaváděl odděleně a nejdříve měřil přínos proti současnému dev baseline, ne přidával současně více optimalizací.

Podrobné metriky, grafy a protokol: [NETWORK_RESULTS.md](NETWORK_RESULTS.md) a [TRAINING_RESULTS.md](TRAINING_RESULTS.md). Před odevzdáním text vlastními slovy zkrať a zkontroluj, že odpovídá tomu, co umíš vysvětlit.


## Zlepšení použitelnosti: výsledek přímo u hledáčku

Uživatel při skutečném používání na **iPhonu 13 mini** upozornil, že pro přečtení vítězné třídy musel odscrollovat od kamery, a tím ztrácel přehled o záběru. Hlavní výsledek jsme proto přesunuli přímo pod hledáček: jméno nejvýše skórující třídy, procentní skóre a označení nejistoty se aktualizují společně s původními čtyřmi pravděpodobnostmi.

Mobilní režim po spuštění kamery nebo výběru fotografie skryje velký úvod a přesune pohled na vstupní panel. Čtvercový hledáček se vejde do dostupné výšky pomocí svh a safe-area insets, ovládání zachovává dotykové cíle a při otočení telefonu se rozložení přizpůsobí. Podrobné pravděpodobnosti a diagnostika zůstaly níže.

WebKit a Chrome prošly kontrolou rozměrů 375 × 812, 375 × 650, 375 × 600 a landscape 812 × 375; test ověřuje současnou viditelnost hledáčku, vítězné třídy a ovládání, přepínání vítěze i skutečně běžící simulovaný stream v Chrome. [Záznam](training/results/mobile_layout_checks.json). To není fyzické potvrzení opravy na telefonu: **DOPLNIT vlastní ověření nového rozložení a screenshot na iPhonu 13 mini**.

Jde o zlepšení použitelnosti, nikoli přesnosti. Síť, váhy, preprocessing a vypočtená skóre se touto úpravou nemění.
