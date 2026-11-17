# Test data

Sample files used by the tests in `test/` and the examples in `docs/source/examples/`. They are not part of the
installed package.

## Files provided by SINTEF Ocean

SINTEF Ocean provided these files for the QATS tests, with permission to publish them in this repository. They hold
the same SIMA result, the nodal displacements of a flexible riser (204 series, 200 time steps), in three formats.

| File | Format |
| --- | --- |
| `sintef_asc_noddis.asc`, `key_sintef_asc_noddis.txt` | SIMA/RIFLEX ASCII results and key file |
| `sintef_bin_noddis.bin`, `key_sintef_bin_noddis.txt` | SIMA/RIFLEX binary results and key file |
| `sintef_noddis.h5` | SIMA HDF5 |

## Files from other projects

The OpenFAST files are copied unchanged, apart from their names, from the projects below. They are licensed under the
Apache License 2.0, a copy of which is in [LICENSE-OpenFAST](LICENSE-OpenFAST).

| File | Source |
| --- | --- |
| `openfast_MinimalExample.out` | [OpenFAST/r-test](https://github.com/OpenFAST/r-test) `glue-codes/openfast/MinimalExample/MinimalExample.out` |
| `openfast_MinimalExample.outb` | OpenFAST/r-test `glue-codes/openfast/MinimalExample/MinimalExample.outb` |
| `openfast_5MW_OC4Semi_Linear.outb` | OpenFAST/r-test `glue-codes/openfast/5MW_OC4Semi_Linear/5MW_OC4Semi_Linear.outb` |
| `openfast_5MW_OC3Mnpl_Linear.outb` | OpenFAST/r-test `glue-codes/openfast/5MW_OC3Mnpl_Linear/5MW_OC3Mnpl_Linear.outb` |
| `openfast_bd_driver.out` | OpenFAST/r-test `modules/beamdyn/bd_static_cantilever_beam/bd_driver.out` |
| `openfast_ifw_primary.out` | OpenFAST/r-test `modules/inflowwind/py_ifw_turbsimff/ifw_primary.out` |
| `openfast_md_driver.MD.out` | OpenFAST/r-test `modules/moordyn/md_case5/driver.MD.out` |
| `openfast_FASTOutBin.outb` | [OpenFAST/openfast_toolbox](https://github.com/OpenFAST/openfast_toolbox) `openfast_toolbox/io/tests/example_files/FASTOutBin.outb` |
| `openfast_FASTOut_Hydro.out` | OpenFAST/openfast_toolbox `openfast_toolbox/io/tests/example_files/FASTOut_Hydro.out` |

Copied from OpenFAST/r-test at commit `dd5feaaaa500ba7283140107806300d551cff0a7` and OpenFAST/openfast_toolbox at
commit `54e6f3acb7b04254ccb6e8e4b9bdd06368f40168`.

The Bladed files are copied unchanged, also their names, from the demo runs in DNV's
[`dnv-bladed-results`](https://pypi.org/project/dnv-bladed-results/) 2.2.2 (`UsageExamples/Runs/demo`), licensed under
the MIT License, a copy of which is in [LICENSE-dnv-bladed-results](LICENSE-dnv-bladed-results). Each header
(`.%NN`) names its data file (`.$NN`) and holds statistics of the data, so neither is changed.

| Files | Source | Content |
| --- | --- | --- |
| `powprod.%04`, `powprod.$04` | `powprod/` | Control variables: 9 series, 1,501 time steps |
| `estop.%15`, `estop.$15` | `estop/` | Blade 1 loads at 2 stations: 8 variables × 2 stations, 624 time steps |
| `lin1.%31`, `lin1.$31` | `lin1/` | Blade 1 absolute motion against wind speed: not a time series |

## Realisations of one process

`riser_tension_seeds.ts` and `riser_tension_seeds.key` hold 20 realisations of one stationary process: the effective
tension at the top (end A) of a catenary riser in irregular waves, from 20 simulations of the same sea state with
different wave seeds. They serve the extreme value tests and examples, where the maxima of several realisations are
fitted with a Gumbel distribution.

The QATS maintainers generated them with OrcaFlex 11.6b from the example model A01 Catenary riser, which Orcina
distributes openly with OrcaFlex. The results are published with this repository under its licence.

- Sea state: Hs = 7 m, Tz = 8 s (Tp = 11.26 s), surface current 1 m/s
- Duration 10,800 s (3 hours) per seed, after the build-up period was removed; time step 0.1 s (108,001 points)
- Tension in kN, series `seed01` to `seed20` in the order of the simulation cases
- The largest maxima are short, snatch-like peaks after near-slack, resolved by 2–3 time steps. A coarser time step
  would lower them noticeably, so the file keeps the logging time step of the simulations.

| Series | Wave seed |
| --- | --- |
| `seed01` | -1635867623 |
| `seed02` | -775308176 |
| `seed03` | 10805082 |
| `seed04` | 1938501628 |
| `seed05` | 50778171 |
| `seed06` | 632211351 |
| `seed07` | 1546190736 |
| `seed08` | 1945408073 |
| `seed09` | -1706661756 |
| `seed10` | 1649781963 |
| `seed11` | -1188483806 |
| `seed12` | 1695683714 |
| `seed13` | 433923357 |
| `seed14` | 521770244 |
| `seed15` | 242919145 |
| `seed16` | -340956190 |
| `seed17` | 1217079413 |
| `seed18` | 311680655 |
| `seed19` | 205348630 |
| `seed20` | 1470604236 |
