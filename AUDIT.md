# Technický audit a plán dokončení

Kontrola z 11. 11. 2026. Tento dokument sepsal agent jako technický podklad. **Nenahrazuje tvoje vlastní `REVIEW.md` ani `REPORT.md`.** Rozhodující je [zadání](https://tomasvicar.github.io/MLR-public/exercises/ex08.html) a text úloh v původním notebooku.

## Verdikt

Funkční web, veřejný repozitář, nasazený model a měřené experimenty máme. Odevzdání zatím **není kompletní**: chybějí oba studentské texty a skutečný test/screenshot z telefonu. Navíc je potřeba opravit a přeměřit Task 3d. WebKit nyní nesplňuje doslovnou shodu referenčního JPEG na tři desetinná místa; tolerance 0,002 tento požadavek nenahrazuje.

### Přehled zadání

| Požadavek | Stav | Důkaz / co zbývá |
| --- | --- | --- |
| Task 1: spustit a rozumět příkladům | Technicky ověřeno | Gradienty, Linear, SGD, batching a uložení/načtení ověřeny novým během. Porozumění musíš doložit ty. |
| Task 2: skupinová data, MobileNet, trénink, confusion matrix, export | Technicky ověřeno | Původní buňky prošly v čistém procesu; velikosti dat 107/29 a ONNX/PyTorch reference souhlasí. |
| Veřejný `mlr-ex08`, Pages, README a čtyři soubory modelu | Hotovo | [Repozitář](https://github.com/kafcoo/mlr-ex08), [aplikace](https://kafcoo.github.io/mlr-ex08/). |
| Pozvánka `MLRteacher` | Hotovo | Při auditu je stále evidována čekající pozvánka s oprávněním write; zadání čekající pozvánku připouští. |
| Rear camera, foto fallback, všechny čtyři pravděpodobnosti | Prohlížečové testy prošly | Chrome: simulovaný živý stream; WebKit: model, fotografie a odmítnutí kamery. Fyzický telefon zbývá. |
| Shoda RGB / mean/std / NCHW | Ověřeno | Syntetická kontrola centrálního výřezu, pořadí kanálů a normalizace; stejná RGB data dávají shodný výsledek. |
| Referenční JPEG na tři desetinná místa | Chrome ano, WebKit ne | Apple: PyTorch 0,810264 → 0,810; WebKit 0,811441 → 0,811. Na webu je nyní oddělená informace o toleranci a zaokrouhlení. |
| Nic z obrazu neopustí zařízení | Kód ověřen, síťové kontroly doplněny | Inference je v lokálním WASM. CDN se načítají kvůli knihovnám, model/JSON/reference ze stránky. Vlastní pozorování napiš do review. |
| QR k vlastní Pages URL | Ověřeno | QR vzniká v prohlížeči; nezávislý dekodér načetl `https://kafcoo.github.io/mlr-ex08/`. |
| Task 3a–c: vlastní dataset, síť, loop | Technické kontroly prošly | Všech 136 položek vlastního Datasetu souhlasí s ImageFolder; LeafNet má 208 292 parametrů a výstup 5 × 4. |
| Task 3d: augmentace + BatchNorm | Chyba k opravě tebou | Vytvoří se `aug_train_loader`, ale loop čte `my_train_loader`. Dosavadní čísla nejsou výsledkem kombinace augmentace + BN. |
| Task 4: pevný baseline a postupné změny | Splněno pro měřené varianty | Task 2 MobileNet baseline je zadáním povolen. Vlastní LeafNet z Task 3 zatím není řádkem těchto 12 běhů. |
| Task 4: 2–3 seedy | Splněno | 4 varianty × seedy 0, 1, 2 = 12 běhů, všechny po 15 epochách. |
| Task 4: camera-like nebo historické fotografie | Splněno v rozsahu datových testů | Pevné camera-like dev sady a historický 120snímkový test. Nejde o měření reálné kamery. |
| Vybraný export v kurzové aplikaci | Ověřen foto fallback v prohlížeči | [Kurzová aplikace](https://tomasvicar.github.io/MLR-public/apps/classifier/), metadata z ONNX a klasifikace fotografie; fyzický telefon stále zbývá. |
| `REVIEW.md` + telefonní screenshot | Chybí | Zadání výslovně požaduje tvůj vlastní text a skutečný screenshot. |
| `REPORT.md` s interpretací a volbou modelu | Chybí | `TRAINING_RESULTS.md` je automatický technický záznam, nikoli studentský report. |
| Termín do 48 hodin po cvičení | Musíš ověřit ty | Doplň skutečný začátek/konec cvičení podle pokynů vyučujícího a konkrétní deadline; čas vytvoření souborů ho nedokazuje. |

## Co skutečně proběhlo při kontrole

- Původní notebook má 65 buněk, z toho **29 kódových**. Všechny se provedly v novém IPython procesu bez běhové chyby, přibližně za **103 sekund** na CPU.
- **13 ze 14** doplňkových věcných kontrol prošlo. Zbývající kontrola potvrdila špatný loader v Task 3d. Bezchybný běh tedy neznamená úplnou shodu se zadáním.
- Původní zdroj notebooku nebyl upraven. Lokální `ex08_checked.ipynb` obsahuje nově získané výstupy; [strojový záznam](training/results/notebook_audit.json) uvádí hash původního zdroje a výsledky kontrol.
- Notebook běžel v oddělené pracovní složce. Jeho Task 2 export tak nepřepsal nasazený model.
- [Kontrola integrity](training/results/integrity_checks.json) ověřila 935 zachovaných zdrojových RGB hashů, počty a oddělení dat, 12 historií po 15 epochách, metriky 13 modelů na 6 sadách, pravidlo výběru modelu, ONNX metadata/reference i obsah exportního ZIPu.
- [Reprodukční smoke test](training/results/reproducibility_checks.json) skutečně provedl krátké tréninky v dočasné složce a ověřil resume, chybějící checkpoint, změnu konfigurace, chybnou baseline i ochranu archivu.
- [Kurzová aplikace](training/results/course_app_check.json) načetla náš vlastní ONNX se správnými třídami a normalizací a klasifikovala referenční fotografii. To není screenshot z fyzického telefonu.

## Důležité nálezy a minimální opravy

### 1. Task 3d nepoužívá augmentovaná data — neopraveno za tebe

V buňce s tréninkem `LeafNetBN` změň vstup smyčky na vytvořený augmentovaný loader a stejně uprav jmenovatel při výpisu průměrné batch loss:

```python
for images, labels in aug_train_loader:
    # Stávající tělo smyčky ponech.
    ...

# Ve stávajícím printu použij len(aug_train_loader).
```

Nejde o požadavek přidat doslova `...`: změň pouze název loaderu v obou příslušných místech svého kódu. Pak znovu inicializuj model, nastav stejný seed a buňku celou spusť. Validaci nech bez augmentací.

Pro skutečné rozlišení efektů porovnej **LeafNet**, **LeafNet + augmentace**, **LeafNetBN bez augmentace**, **LeafNetBN + augmentace**, vše se stejnými daty a seedem. Dosavadní nejlepší validační hodnoty 20/29 versus 22/29 znamenají rozdíl pouhých dvou obrázků; jedna validační chyba je **3,45 procentního bodu**. Z jednoho takového běhu nevyvozuj statistickou jistotu.

Task 3 je podle notebooku samostatná studentská práce bez agenta, proto její implementaci nepřepisuju. Vlastní dataset nyní funguje pro dodané soubory; do budoucna by měl filtrovat přípony obrázků, aby například `.DS_Store` nebyl považován za fotografii. Zbylé komentáře TODO/PLACEHOLDER po vlastní kontrole vyčisti, protože u funkčního řešení působí jako nedokončená práce.

Původní notebook volí jen CUDA nebo CPU, takže na tomto Macu běžel na CPU. Samostatný `train.py` už využívá dostupné MPS. Pokud přidáš MPS i do notebooku, u měření GPU času ošetři také `torch.mps.synchronize()`; jinak lze měřit pouze odeslání asynchronní práce, ne skutečný čas výpočtu. Pro samotné splnění úloh není MPS potřeba.

### 2. Rerun původního notebooku může přepsat aktuální webový model

Task 2e/f zapisuje `leaf_model.onnx`, `model_info.json`, `check_expected.json` a ZIP do aktuální pracovní složky. Po novém tréninku Task 2 tak můžeš nechtěně vrátit web na jiný model. Pro kontrolu jsem použil izolovanou složku. Pro další práci odděl výstup Task 2 od vydaného modelu; nový export nejdřív ověř a až pak ho vědomě nasaď.

### 3. Reprodukce experimentů — opraveno

Původní `train.py` přeskočil běh, pokud našel JSON, i když nebyly k dispozici váhy. To by v čerstvém klonu se zveřejněnými metrikami a ignorovanými checkpointy nefungovalo. Nyní musí existovat i checkpoint a odpovídající protokol. Změna dat/nastavení vyžaduje nový `--run-dir`; původní studie je chráněna před přepsáním.

Původní `evaluate.py` při chybějící záloze kopíroval aktuální root model a nazval ho původní baseline. To mohlo tiše porovnávat jiný model. Nyní vyžaduje existující soubor a ověřuje jeho SHA-256. [Návod k reprodukci](training/README.md) ukazuje i obnovu pevné původní baseline z konkrétního Git commitu.

Příprava dat stále používá společný `data/clean` a zapisuje výchozí datový audit. Při pouhé reprodukci se stejnými zdrojovými fotkami je to v pořádku. Před přidáváním dalšího datasetu je další malý krok oddělit také připravená data a jejich manifest podle experimentu; samotný nový `--run-dir` odděluje výsledky a checkpointy, nikoli tento datový adresář. Neupravuj zdrojová data historické studie a nepřepisuj její audit novým datasetem.

### 4. Metadata ONNX a aktuální ZIP — opraveno

Nový Task 4 export původně obsahoval správný doprovodný JSON, ale chyběl mu vnitřní ONNX klíč `model_info`, který vytváří Task 2e a čte kurzová aplikace. Export jej nyní vkládá a ověřuje. Obnovuje i ZIP ze čtyř aktuálních souborů, takže se nerozcházejí váhy, nastavení a reference. **Neměnily se naučené váhy ani výsledky experimentů**, pouze metadata a exportní balíček.

### 5. Shoda na tři desetinná místa — otevřený bod na WebKitu

Pro aktuální model jsou výsledky původního JPEG:

| Třída | PyTorch | Chrome | WebKit |
| --- | ---: | ---: | ---: |
| apple | 0,810264 | 0,810264 | 0,811441 |
| cherry | 0,126014 | 0,126015 | 0,125688 |
| chestnut | 0,023147 | 0,023147 | 0,022892 |
| maple | 0,040574 | 0,040574 | 0,039980 |

Na přesně stejných RGB pixelech z [bezeztrátové diagnostické reference](training/results/reference_pixels.png) se oba prohlížeče shodují s PyTorch přibližně do **2,5 × 10⁻⁷**. Větší rozdíl tedy vzniká při JPEG dekódování. Zvýšená tolerance ale není náhradou doslovného požadavku na tři desetinná místa. Web nyní ukazuje zvlášť `JPEG tolerance` a `Three-decimal match`.

Do vlastní review napiš skutečná čísla z telefonu a co jsi zjistil. Pokud nesouhlasí, ukaž i diagnostickou PNG referenci a problém transparentně popiš; případnou akceptaci této odchylky potvrzuje vyučující, ne agent. Technická cesta k přesné shodě je sjednotit dekódování v tréninku a prohlížeči nebo po dohodě používat bezeztrátovou referenci. Nenahrazuj potichu původní JPEG ani neupravuj očekávané hodnoty podle jednotlivých prohlížečů.

## Co máme v Task 4 změřeno

| Varianta | Tréninkových fotek | Seedy | Historický test accuracy | Camera-like test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Původně nasazený Task 2 model | 107 | pevný checkpoint | 50,00 % | 50,00 % |
| Stejná architektura, nové běhy / původní data | 107 | 0, 1, 2 | 47,22 ± 2,83 % | 47,22 ± 4,10 % |
| + očištěná historická trénovací data | 649 | 0, 1, 2 | 86,11 ± 3,36 % | 84,72 ± 1,42 % |
| + geometrické augmentace | 649 | 0, 1, 2 | 87,50 ± 0,00 % | 86,11 ± 0,39 % |
| + ColorJitter | 649 | 0, 1, 2 | 84,44 ± 3,87 % | 86,39 ± 0,79 % |

`±` je směrodatná odchylka mezi třemi seedy, nikoli interval spolehlivosti přes fotografie. Nulový rozptyl accuracy geometrické varianty neznamená nulovou nejistotu nebo nutně totožné chyby.

Na telefonu je zatím `expanded_color`, seed 2: **79,17 %** na čistém historickém testu a **85,83 %** na camera-like testu. To jsou čísla konkrétního checkpointu, nikoli průměr řádku. Varianta byla zvolena podle průměrného dev skóre a seed podle předem stanoveného mediánu, před vyhodnocením testu. Nezměnili jsme následně volbu podle nejhezčího testového výsledku.

### Jak výsledky vysvětlit poctivě

- Největší naměřený přínos má přidání historických dat. Pro posouzení jeho efektu porovnávej nové běhy `original_only` s `expanded_only`, ne jen starý finální checkpoint s novým nejlepším checkpointem.
- Při 15 epochách dostává větší dataset také více optimizer kroků: 4 batche/epochu versus 21, tedy přibližně **60 versus 315 aktualizací**. Je to legitimní praktické porovnání dvou datových režimů, ale nezjišťuje čistý efekt diverzity při stejném počtu aktualizací. Jako doplňkovou kontrolu lze rozpočet aktualizací vyrovnat.
- Barevné augmentace mají jen o **0,23 procentního bodu** vyšší průměrné dev skóre než samotná geometrie; to je menší než variabilita mezi seedy. Na čistém testu měly horší průměr. Nemáme důkaz, že se ColorJitter obecně vyplatí.
- Jedna chyba v historickém testu znamená **0,83 procentního bodu**. Testové a camera-like snímky představují dvě zobrazení stejných 120 fotografií, ne 240 nezávislých měření.
- „Historický test“ už je prohlédnutý. Při novém ladění z něj nesmí vzniknout opakovaně používaná skrytá validace. Pro nové závěry odlož novou sadu telefonních snímků.
- U nových dat drž pohromadě tentýž list/rostlinu/fotícího a jednu fotografickou session. Přesné duplikáty jsme odstranili, téměř totožné záběry a příbuzné snímky mohou v archivu zůstat.

## Další experimenty: krátká, vysvětlitelná fronta

Není povinnost vyzkoušet každou položku ze seznamu kandidátů. Nejdřív dokonči povinné důkazy a texty. Potom pokračuj v nových adresářích; dosavadních 12 běhů zůstává archivem.

| Priorita | Jediná nová otázka | Co zůstává stejné | Co měřit / pozor na |
| --- | --- | --- | --- |
| 1 | Jak funguje model na skutečném telefonu? | Zamknutý model a preprocessing | Nové listy, světlo, pozadí, různé vzdálenosti; accuracy, nejčastější záměny, zpoždění a FPS. Nezahrnout tyto testovací snímky do tréninku. |
| 2 | Jen nová poslední vrstva vs fine-tuning celé sítě | Stejná geometrická augmentace, data, 15 epoch, seedy 0/1/2 | Zmrazit i running statistiky BatchNorm; samotné `requires_grad=False` je nezmrazí. Pro „jen poslední vrstvu“ musí být ostatní vrstvy v eval režimu. |
| 3 | Menší konstantní learning rate, například 1e-4 | Stejné epochy, optimizer, augmentace, data, seedy | Mění se pouze LR. Nezavádět současně nový scheduler a další epochy. |
| 4 | Pomůže cosine scheduler? | Stejný počáteční LR a 15 epoch | Porovnat konstantní LR a `CosineAnnealingLR(T_max=15)`; scheduler krokovat po optimizer aktualizacích dané epochy. |
| 5 | Pomůže delší trénink? | Stejná varianta a optimizer | 15 vs 30 epoch nejdřív při konstantním LR; u cosine mění změna T_max i průběh LR, což je potřeba přiznat. |
| 6 | Pomůže label smoothing nebo weight decay? | Stejné data a tréninkové nastavení | Nejprve smoothing 0 vs 0,05. AdamW/weight decay měřit jako samostatný experiment, ne současně se smoothingem. |
| 7 | Zlepší GAP/dropout vlastní Task 3 síť? | Stejný LeafNet baseline a split | U GAP nahradit flatten prostorovým průměrem a upravit vstup fc vrstvy. Dropout měřit zvlášť; MobileNet už dropout i global pooling obsahuje. |

Praktický nový telefonní test: například 10 snímků na druh, několik různých listů a aspoň dvě sessions. Pro test neznámých vstupů přidej jiné druhy a záběry bez listu. Čtyřicet snímků je pořád malý pilot, ne přesný odhad provozní accuracy. Pro úpravy podle těchto fotografií si odděl ještě další nedotčenou finální sadu.

Jako pozdější úpravy webu dávají smysl změřené FPS a jemné vyhlazení několika posledních predikcí; předem zvaž kompromis stabilita vs zpoždění. Kvantizaci má smysl řešit až po změření skutečné rychlosti na telefonu. Trénovaná detekce „neznámé“ třídy potřebuje vhodná negativní data; současný práh skóre/marginu ji nenahrazuje.

## Co musíš před odevzdáním udělat ty

1. V Task 3d opravit loader, znovu inicializovat model a přeměřit požadované kombinace. Dopsat vlastní odpovědi u otázek k confusion matrix a vlivu augmentace/BN.
2. Otevřít [vlastní aplikaci](https://kafcoo.github.io/mlr-ex08/) v Safari na telefonu, vyzkoušet zadní kameru a upload. Pořídit skutečný screenshot s listem a pravděpodobnostmi; před zveřejněním zkontrolovat soukromé údaje a odrazy v obraze.
3. Nahrát referenční JPEG a do vlastní review opsat skutečné hodnoty vedle očekávaných. Rozlišit shodu na tři desetinná místa od obecné JPEG tolerance. Zkusit i diagnostickou PNG referenci a uvést rozdíl.
4. Vyzkoušet vlastní ONNX v [kurzové aplikaci](https://tomasvicar.github.io/MLR-public/apps/classifier/) také na telefonu; aktuální ONNX lze získat z veřejného repozitáře.
5. Napsat vlastní `REVIEW.md`: co jsi zkoušel, screenshot, čísla, preprocessing, kam a proč se načítají síťové soubory, případné neshody a opravy.
6. Napsat vlastní krátký `REPORT.md`: baseline, měněné faktory, tabulka průměrů a rozptylů, výsledky vůči šumu, co nepomohlo, odůvodněná volba modelu a zjištění z telefonu. [TRAINING_RESULTS.md](TRAINING_RESULTS.md) poskytuje čísla a grafy, text interpretuj vlastními slovy.
7. Pushnout oba dokumenty a screenshot do `mlr-ex08` **do 48 hodin po cvičení**. Ověřit, že jsou vidět na GitHubu a je stále platná pozvánka učiteli.

Změny vzhledu, nové grafy ani samotný automatický audit nenahrazují tyto studentské výstupy.
