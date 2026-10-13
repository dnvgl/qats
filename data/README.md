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

The files are renamed. In the key files, the path of the result file on the computer that wrote it is replaced by
the file name; nothing else is changed.

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
