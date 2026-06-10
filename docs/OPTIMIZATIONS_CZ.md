# bogo_gpu_turbo / bogo_gpu_fast — optimalizované workery (2026-06-10)

## TURBO ✅ ~1.6× — server naměřil 47.2 B/s (doporučený default)

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
