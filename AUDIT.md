# Aktuální audit a dokončené opravy

Aktualizováno 11. 11. 2026 po výslovném souhlasu uživatele s opravou notebooku, JPEG a dalšími experimenty. Předchozí audit je zachován v Git historii. Technické podklady nenahrazují skutečná telefonní pozorování ani vlastní interpretaci studenta.

## Vyřešené body

| Oblast | Stav a důkaz |
| --- | --- |
| Opravený notebook | 30/30 kódových buněk bez chyby, 15/15 věcných kontrol. Čerstvé výstupy v lokálním `ex08_student.ipynb`; [záznam](training/results/notebook_audit.json). |
| Task 3d | Smyčka i jmenovatel výpisu používají `aug_train_loader`; skutečně se trénuje augmentace + BN. |
| Reprodukovatelnost Task 3 | Vlastní seedované generátory shuffle, filtr přípon obrázků, zachovaná původní záloha. |
| Task 2 export | Model, JSONy, reference a ZIP se ukládají do `exports/task2/`, ne přes vydaný root model. |
| ONNX metadata | `model_info` je uvnitř ONNX i v doprovodném JSONu; export ověřen proti PyTorch. |
| JPEG Safari/Chrome | Jednotný MozJPEG/WASM; strict error < 1e-5 i shoda na tři desetinná místa procházejí v obou prohlížečích. Původní JPEG je zachovaný. |
| EXIF | Osm orientací dává stejné pixely jako Pillow + exif_transpose v obou prohlížečích. [Výsledky](training/results/jpeg_decoder_checks.json). |
| Nové sítě | 21 běhů, sedm variant × tři seedy, stejná data a epochy; [NETWORK_RESULTS.md](NETWORK_RESULTS.md). |
| Férovost | Nová studie používá pouze dev data, žádný historický test. [Fingerprints a kontroly](training/runs/networks_v1/results/integrity_checks.json). |
| Obhajoba modelu | Dev skóre doplněné parametry, velikostí a desktopovým single-thread WASM časem. |
| Historie | Původních 12 měření, jejich datový audit, evaluace a výběr se nemění. |
| Web | Jedno HTML, plain JS, zadní kamera, upload, čtyři pravděpodobnosti, lokální inference a QR na vlastní URL. |
| Podklady | [REPORT_DRAFT.md](REPORT_DRAFT.md) a [REVIEW_DRAFT.md](REVIEW_DRAFT.md), bez vymyšlených telefonních výsledků. |

## Změny notebooku

1. Dataset zachovává pořadí tříd a transformaci, pouze filtruje `.jpg/.jpeg/.png`.
2. Oba trénovací loadery mají `torch.Generator().manual_seed(0)`.
3. Task 3d skutečně používá augmentovaný loader. U dokončeného kódu byly odstraněny zbylé TODO/PLACEHOLDER komentáře.
4. Task 2e/f zapisuje do `exports/task2/`; ZIP obsahuje čtyři jednoduché názvy souborů.
5. Task 4 má rozcestník k novým výsledkům a návrhům. Run All automaticky nespouští novou 21běhovou studii.
6. Samostatná Task 4 buňka pouze načte aktuální vybraný PyTorch checkpoint a ověří jeho pravděpodobnosti proti souborům webu. Tím se nepletou hodnoty původního Task 2 modelu s novým modelem. Grafy a confusion matrix jsou vložené ve vykonaném notebooku.

Staré výsledky Task 3d nepoužívej jako měření „augmentace + BN“. Nové výstupy jsou získané skutečnou augmentací. Jedna chyba původní validace má 3,45 procentního bodu, takže jeden běh není silný statistický důkaz.

## Nové výsledky

| Varianta | Dev balanced skóre: průměr ± SD |
| --- | ---: |
| LeafNet + geometrie | 68,46 ± 1,48 % |
| LeafNetBN + geometrie | 73,68 ± 0,66 % |
| LeafNetBN + GAP + geometrie | 69,99 ± 3,16 % |
| MobileNet, plný fine-tuning | 80,35 ± 0,80 % |
| MobileNet, jen poslední vrstva | 79,48 ± 0,29 % |
| MobileNet + cosine | 81,36 ± 0,41 % |
| MobileNet + smoothing 0,05 | 83,14 ± 0,66 % |

Vybraný je MobileNet se smoothingem, seed 0, nejlepší epocha 14, konkrétní dev skóre 82,76 %. Průměr varianty a konkrétní checkpoint nejsou totožná čísla. Nový model nemá novou nezávislou testovou accuracy; historických 79,17 % patří předchozímu checkpointu.

BN pomohl vlastní síti. GAP šetří velikost, ale zde zhoršil skóre. Frozen-head šetří trénovatelné parametry, ne inferenční práci. MobileNet využívá ImageNet, takže proti vlastní síti od nuly neizolujeme pouze architekturu. SD tří seedů není interval spolehlivosti přes nové listy. Podrobnosti, graf a rozpočty jsou v [NETWORK_RESULTS.md](NETWORK_RESULTS.md).

## Co zbývá skutečně ověřit

Uživatel nově potvrdil používání iPhonu 13 mini a nahlásil nepoužitelné oddělení kamery a vítězné třídy. Hlavní výsledek je nyní přímo pod hledáčkem, mobilní úvod se při práci skryje a hledáček respektuje výšku viewportu/safe areas. Automatické [layout kontroly](training/results/mobile_layout_checks.json) prošly; zbývá potvrdit opravu na jeho skutečném telefonu. Jde o ergonomii, ne o změnu accuracy. Funkce je uvedená také v návrzích reportu a review.

- Telefon/OS, zadní kameru, zastavení/obnovení, upload, světlo, různé listy a pozadí; skutečný screenshot bez soukromých údajů.
- Referenční JPEG na vlastním telefonu a jeho čtyři hodnoty vedle `check_expected.json`. Numerický PASS neznamená jistotu identifikace listu; současná reference může mít hlášku Uncertain.
- Aktuální model i v [kurzové aplikaci](https://tomasvicar.github.io/MLR-public/apps/classifier/). Ta má vlastní JPEG kód; naše oprava mění vlastní web. Pro izolaci jejího dekódování lze použít diagnostickou PNG referenci.
- Novou telefonní/session sadu držet mimo ladění. Historický 120snímkový test už byl viděn a znovu neposkytuje nezávislé potvrzení nové volby.

## Další rozumné kroky

1. Nejdřív sbírat nové chyby telefonu, hlavně apple/cherry a druhy s malým množstvím vlastních fotografií.
2. Změřit skutečný čas/FPS telefonu. Případné vyhlazení několika predikcí porovnat s přidaným zpožděním.
3. Více epoch, menší LR či kombinaci scheduler+smoothing zkoušet odděleně a v novém run-dir; tyto kombinace zde zatím měřené nejsou.
4. Pro skutečné „neznámé druhy“ doplnit negativní data a samostatné vyhodnocení. Práh skóre/marginu není pátá třída ani kalibrace.
5. Před změnou zdrojových dat oddělit také připravené datové adresáře a manifesty podle experimentu. Tato studie znovu použila stejné neměnné `data/clean`.

## Odevzdání

Technické podklady jsou připravené. Zadání požaduje vlastní finální `REVIEW.md` a `REPORT.md`, skutečný telefonní screenshot a push do **48 hodin po cvičení**. Drafty obsahují reálná měření a jasná místa pro telefon. Pozvánka `MLRteacher` už byla odeslána; čekající pozvánka podle zadání stačí.
