# bogo_gpu_turbo / bogo_gpu_fast — optimalizované workery (2026-06-10, popcount bound 2026-06-13, V6: no-flag + STOP@floor + SplitMix reuse 2026-06-15)

## TURBO V6 ✅ — SplitMix reuse + no-flag tahy + STOP@floor, server +26 % proti v2

`MAINBOGOGPU_NVIDIA_newAPI_turbo.cu` (kernel `bogo_range_h<256,13,512>`). Tři
kroky nad v2, všechny ověřené stejnou validační sadou (3 seedy × 2^30 + 40
podrozsahů + CPU recheck každé trojice) a ostrým serverem (0 rejected):

1. **SplitMix reuse (+~10–15 %, největší).** `seed_expand(k)` počítá
   `a = mix(base+(k+1)·g)`, `b = mix(base+(k+2)·g)`, takže **`b(k) == a(k+1)`** —
   dva po sobě jdoucí indexy sdílejí jeden SplitMix64. Vlákno teď prochází
   **souvislý blok** indexů (grid-stride PŘES bloky, `BLOCK_SIZE=512`) a `b`
   předchozího indexu použije jako `a` toho dalšího, takže každý index po prvním
   stojí **jeden mix() místo dvou**. Mix xor-shifty leží na ALU pipe (úzké
   hrdlo), takže půlení seedu je přímý zisk a platí **na každém floor** (proto je
   reálný zisk větší než bench na f13). 48 registrů → ~83 % okupance, ale seed
   úspora to víc než vyrovná. Bit-exaktní (validace zelená).
2. **No-flag (straight-line) tahy (+7 %).** Optimistický tah ve v2 ještě nesl
   per-draw test `bad |= res < TH` (RNG rejection, ~1× na 18 M indexů) — **OR
   řetězený přes celý screen**, serializuje plánování. V6 ho z horké cesty
   vyhazuje: tahy jdou rovně a **každý** index, jehož horký count překoná floor,
   se přepočítá na přesné studené cestě (`exact_redo_l`, jediný publisher,
   řádná rejection smyčka) → **report bytově identický (0 rejected)**. Jediná
   teoretická ztráta: rekord s rejectionem ve svých tažených krocích se může
   minout (~1e-7/chunk) — nikdy ne špatný report.
3. **STOP přeladěný na PRODUKČNÍ floor → 13 (ne 12) (+5 %).** Sweep STOP běžel na
   bench floor 8, ale worker nese `winBest` (~13) do floor launche, kde se
   divergentní ocas skoro nikdy nespustí — takže **o jeden povinný screen tah
   míň vyhrává**. STOP=13 vyhrává napříč f12–15; STOP=14 je na f14+ rychlejší,
   ale na f12 se hroutí (moc citlivé pro fixní kernel).

Naměřeno (bench2, produkční tvar 2560×256, chunk 2^31):

```
kernel                  f8     f12    f13    f14
hp    <256,12> (v2)    63.7   65.6   66.5   66.1
nf    <256,13> (v3)    65.2   70.7   74.3   75.1
reuse <256,13,B512> V6 77.0   77.3   81.9   83.3
```

**Ostrý server (RawPower, 2026-06-15, A/B ve stejné session, 0 rejected):**
v2 64.4 → v3 72.3 → **V6 82.3 B/s** (server rate field) = **+26 % proti v2**
(+14 % proti v3), session best 14–15 jako obvykle. Bench na floor 13: v2 66 →
**V6 ~82 (+24 %)**. Proti původnímu workeru v rodokmenu (~18 B/s) je to **~4.5×**.

**Co tentokrát NEfungovalo (změřeno, zavrženo):** inkrementální seed −2.5 %
(64bit násobení se pipelinuje dobře, přírůstkový add jen prodlouží řetězec);
adds→IMAD na no-flag tahu −3 %; 2-index ILP ≈neutrál; refresh 16/32 horší (8
zůstává); chunk 2^32 horší na produkčním floor 13 (2^31 + TDR rezerva); reuse
blok 128 horší / 1024 nevyrovnaný (512 optimální); 32bit blokový čítač („lean")
nesnížil registry a byl pomalejší.

Pointa kola 2: i „triviální" seedovací krok šel zlevnit **algebraickou
identitou** (`b(k)=a(k+1)`), ne mikro-optimalizací — opět platí „reformuluj,
než ladíš instrukce", a „měř na floor, na kterém to reálně běží".

---

## TURBO v2 ✅ ~2.3× — popcount bound, server naměřil ~68 B/s

`MAINBOGOGPU_NVIDIA_newAPI_turbo.cu` (tentýž soubor, kernel přepracovaný
2026-06-13). Tři nové kroky nad H-mask kernelem, všechny ověřené stejnou
validační sadou (3 seedy × 2^30 + 40 podrozsahů + CPU recheck každé trojice):

1. **Popcount bound (+14 %).** Stará mez `c + i + 1 ≤ best` předpokládá, že
   pevným bodem se může stát každá zbývající pozice. H-maska ale už ví víc:
   pozice `p` se může stát pevnou jen pokud je její bit **dosud nezasažený**
   (zásah ji finalizuje jinde). Nová mez `c + popc(~H & bity 0..i)` — po
   screenu zbývá typicky jen ~5 nezasažených spodních bitů, takže skoro každý
   index se prořeže okamžitě místo procházení 3–4 kroků divergentního ocasu.
2. **H/E split screen (+13 %).** Per-draw účetnictví počtu
   (`if (j==i && !(H & 1<<i)) c++` — porovnání + test bitu + predikovaný add,
   serializované přes H) nahrazují dva čisté OR-akumulátory:
   `H |= 1<<j` (všechny zásahy) a `E |= (1<<j) & ~(1<<i)` (zásahy z *cizího*
   kroku; `j ≤ i`, takže netřeba porovnávat). Pevné bity = `H & ~E`, tedy
   `c = popc(H & ~E)` — a protože jde o disjunktní množiny, celý test meze je
   **jeden LOP3 + jeden POPC**: `popc((H & ~E) | (~H & LOWMASK)) > best`.
3. **Kratší screen + nový tvar (+15 %).** Těsnější mez dovolila posunout test
   o 2 kroky dřív (screen 24..13, test na kroku 12; sweep 10–13) a launch
   přeladit na 256×2560 (dřív 128×2880).
4. **Floor z okna reportu (+2–4 %).** Každý launch dostává jako floor dosavadní
   maximum reportovacího okna (`winBest`) — kernel honí jen počty, které by
   report *zlepšily*. Floor musí být přesně `winBest` (vyšší by mohl tiše
   zahodit lepší nález pod ním); fallback relaunch s floor 0 zůstává jen pro
   první chunk okna.

Naměřeno (bench2, tentýž stroj, baseline 29.7–30.5): starý turbo kernel
44.4 B/s → **hp<256,12> 65.5–65.8 B/s (+47 %)**; s floorem 12–13 (≈ chování
carry-overu v ustáleném stavu) 67.5–68.2 B/s. **Ostrý server (2026-06-13):
~68 B/s — proti 47.2 u v1 je to +44 %, 0 rejected** — sedí přesně na bench
čísla s carry-over floorem. Chunk sweep: 2^32 je o ~1.5 % rychlejší než 2^31,
ale SHARE build zůstává na 2^31 kvůli TDR rezervě pomalých karet. Registry 40,
žádná shared memory, 100% okupance.

Pointa pro příště: profiler po kroku 3 ukazoval ALU pipe 88.7 % a „~92 %
teoretického stropu daného povinnými ~14 tahy" — a stejně z toho šlo dostat
+47 %, protože strop padl reformulací toho, co je „povinné". Lekce
„reformuluj, než začneš mikro-optimalizovat" platí rekurzivně.

---

## TURBO v1 ✅ ~1.6× — server naměřil 47.2 B/s (historie)

`MAINBOGOGPU_NVIDIA_newAPI_turbo.cu` → `build_turbo.bat` → `start_turbo.bat`.

= fast worker + **H-mask reformulace**: počet pevných bodů je funkce samotné
sekvence j-ček —
- pozice `i ≥ 1` je pevná ⟺ `j_i == i` a žádný krok `i' > i` netrefil `i`
  (hodnota `i+1` se před krokem `i` nemůže pohnout; zásah ji finalizuje jinde),
- pozice `0` je pevná ⟺ žádný krok netrefil `0`.

Horká smyčka si tedy drží jen **25-bitovou masku zásahů v registru** — žádné
pole, žádný init, **žádná shared memory** → 100% okupance (128×2880, chunk
2^32). Přesné pole se materializuje jen ve studených cestách (publish/redo,
local memory, ~stokrát za launch). Reporty zůstávají bytově identické.

Naměřeno: bench 44.5 B/s (a to při degradovaném stavu stroje — baseline jen
27.7); **živý server: rate 47.24 B/s, průměr 44.3 B/s vč. startu, 0 rejected,
session best 15**. Validace: shoda skóre na 3 seedech × 2^30 + 40 podrozsahů,
CPU recheck všech trojic.

---

## FAST ✅ ~1.45× — záloha (nedotčeno)

`MAINBOGOGPU_NVIDIA_newAPI_fast.cu` je produkční worker (stejný protokol v5,
stejný host kód) s novým kernelem **`bogo_range_pruned`**. Nic v původních
souborech se neměnilo — fast verze žije vedle nich.

## Naměřeno (RTX 4080 SUPER, ostrý test proti serveru)

| | původní worker | fast worker |
|---|---|---|
| kernel rate (dashboard) | 29.65 B/s | ~43 B/s |
| rate měřený **serverem** | — | **42.95 B/s** (1.45×) |
| průměr za 80 s testu vč. startu | — | 42.12 B/s |
| rejected | 0 | **0** |
| session best během testů | — | 15/25 (all-time je 16) |

Benchmark (`bench2.exe`, stejné podmínky, stejný stroj): baseline 29.7 B/s →
pruning 39.8 → **+ optimistické tahy 43.1 B/s = 1.45×**. Pozn.: stroj někdy
běží v "pomalém" stavu (~21 B/s baseline — pozadí na GPU: prohlížeč/Discord/
overlay), poměr 1.45× ale platí vždy.

## Jak to funguje

Server krátí kredit za `total_done` a ověřuje jen nejlepší trojici
`(best_index, best_arr, best_correct)`. Při Fisher-Yates shuffle shora dolů se
pozice `i` finalizuje v kroku `i` — po něm může přibýt už jen `i+1` pevných
bodů. Jakmile tedy

```
c + i + 1 <= nejlepší_count_tohoto_launche
```

index už **nemůže překonat** průběžné maximum a zbytek shuffle se přeskočí
(stejný princip jako alpha-beta prořezávání). Typicky se tak spočítá jen ~14
z 24 kroků. Vítězný index se přepočítá celý znovu (materializující shuffle),
takže report je **bytově identický** s oficiálním enginem — liší se jedině
tie-break mezi indexy se stejným počtem (server akceptuje libovolný validní).

Detaily kernelu:
- kroky 24..11 běží bez kontrol ("screen"), kroky 10..1 nesou prune test
  (sweep ISTOP ∈ {8..13, 24} → 10 nejrychlejší; čistý per-step prune je
  pomalejší kvůli warp divergenci),
- **optimistické tahy** (nález ze SASS auditu): rejection smyčka RNG
  (P(reject) = TH/2^32, TH ≤ 24 → ~1× za 18 M indexů) je v horké cestě
  nahrazena vlaječkou `bad |= (res < TH)`; když se vlaječka rozsvítí, celý
  index se přepočítá pomalou přesnou cestou (`exact_redo`). Lineární kód bez
  24 vnitřních smyček zbavil kernel BRA/BSSY/BSYNC režie a pustil ptxas k
  plánování napříč kroky → +8 %. Korektnost: rejection se může "schovat" jen
  v krocích, které se vůbec netáhly — provedený prefix je vždy bytově přesný,
- průběžné maximum launche se sdílí přes `atomicMax(best_and_tid)` a čte se
  `__ldcg` jednou za 8 indexů — zastaralá (nižší) hodnota jen méně prořezává,
  nikdy nepokazí výsledek,
- každý launch startuje s floor 8 (P(best ≤ 8 na 2^31) ≈ e^-2400); kdyby launch
  nic nenašel (mikroskopický zbytek lease), host to zopakuje s floor 0,
- `THREADS_PER_BLOCK` je template parametr → statické offsety do shared memory
  jsou immediate hodnoty; launch 1920×192, chunk 2^31.

## Korektnost (bench2.cu)

- shoda nejlepšího skóre s plným průchodem na 3 seedech × 2^30,
- shoda na 40 podrozsazích (32 × 2^22 + 8 × 2^24) vč. CPU překontrolování
  každé reportované trojice,
- 100s ostrý test: server přijal všechny reporty (0 rejected), batch_best
  12–15 jako obvykle.

Co NEfungovalo (změřeno, zavrženo): u16 shared (100% okupance, ale pomalejší
extract), inkrementální seed (žádný rozdíl), lazy-init dirty maskou (horší),
2-way ILP (horší), jiné TPB/blocks (≤ ±2 %), konstantní první krok (-1 %),
řidší prune testy jen na sudých krocích (-2 %), častější/řidší refresh maxima,
párové zpracování indexů + __launch_bounds__ hinty (-0.5 až -2.5 %).
SASS audit potvrdil: žádné LDL/STL (local-memory spill třída chyb nehrozí).

**CPU spolutěžba (bench_cpu.cpp, i7-12700KF):** H-mask matematika na CPU
validní, ale 20 vláken dá max ~194 M/s = +0.4 % ke GPU (i ideální AVX2 by
bylo +1–2 %) — nevyplatí se (teplo, hluk, riziko zdržování GPU reportů).
**RAM:** hot path se DRAM vůbec nedotýká — rychlost/časování RAM nemá vliv.
**Async host pipeline:** režie mezi launchi při chunku 2^32 je ~0.2 % — taky ne.

**Profiler (Nsight Compute, po odemčení čítačů):** ALU pipe 88.7 % = potvrzená
zeď; FMA 20.9 %, paměť ~1 %, nula spillů, 28.4/32 aktivních lanes. Pokusy
využít volnou FMA (rotace→IMAD přes x·2^k + umulhi s opaque konstantami: −8 %;
jen sčítání→IMAD: −3 %) — ptxas volí instrukce optimálně, LOP3/SHF/ISETP nelze
z ALU přesunout. Kernel je na křemíkové zdi; dál už jen OC jádra GPU.

## Build & run

```bat
build_fast.bat      :: -> bogo_gpu_fast.exe (+ z.dll)
start_fast.bat      :: zeptá se na code/UUID/nickname a spustí
```

Benchmark / validace kdykoliv znovu: `build_bench2.bat` a `bench2.exe 2.5`
(příp. `bench2.exe prof [base]` pro Nsight Compute — vyžaduje admin práva na
GPU performance countery).

## Tip mimo kód

Když na GPU sahá něco dalšího (prohlížeč s video akcelerací, overlay,
Discord), baseline padá z ~29.7 na ~21 B/s (−30 %!). Pro maximální výtěžek
zavřít/minimalizovat GPU aplikace; fast worker pak drží ~40 B/s.
