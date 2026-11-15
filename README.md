# Pretrain, Then Convert

Training and evaluation code, analysis scripts, and numerical results for **Pretrain, Then Convert: Compute-Optimal Training of One-Step Image Generators**.

## Contents

- [`pretrain_then_convert/`](pretrain_then_convert/): training and evaluation code (source commit `4f3b405`). See its [README](pretrain_then_convert/README.md).
- [`pretrain_then_convert_paper/`](pretrain_then_convert_paper/): analysis scripts and numerical results (source commit `ed2c28a`). See its [README](pretrain_then_convert_paper/README.md).
- [`NOTE.md`](NOTE.md): scope of the original transfer archive and intentionally excluded material.
- [`MANIFEST.sha256`](MANIFEST.sha256): original SHA-256 checksums for all 965 code, analysis, and result files.

The two source directories retain their original sibling layout. Model checkpoints, sample-image PNG grids, the `paper/` LaTeX package, and rebuildable `two_component.json` intermediate curves were excluded by the source archive.

## Integrity

All transferred files were restored and verified against the original manifest. The source archive SHA-256 is `2927ae95e0b2fd529070770bbb9256d4268c73c9b467688c27ea087f1fa4dac8`.

```sh
shasum -a 256 -c MANIFEST.sha256
```

## License

See [`pretrain_then_convert/LICENSE`](pretrain_then_convert/LICENSE) and [`pretrain_then_convert/NOTICE.md`](pretrain_then_convert/NOTICE.md). The training code includes a separately licensed DMD2 adaptation, as documented there.
