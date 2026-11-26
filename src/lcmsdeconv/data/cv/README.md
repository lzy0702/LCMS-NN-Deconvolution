# Saved controlled vocabularies

lcmsdeconv reads mzML and mzMLb, and writes mzML, with these copies of the PSI-MS controlled
vocabulary and the Units of Measurement Ontology (UO), so it never fetches them from
purl.obolibrary.org, as pyteomics (version 5 and later) otherwise does for every file it reads and
psims for the first file a process writes (`lcmsdeconv/io/vocabularies.py`). Reading needs
PSI-MS, which carries the units it uses (UO is looked up only for a unit PSI-MS lacks); writing
needs both. Each file is the vocabulary exactly as downloaded, gzipped with no time stamp and no
file name in the gzip header; the checksums below are of the uncompressed file
(`gzip -cd psi-ms.obo.gz | sha256sum`).

| File | Vocabulary, and the purl mzML files name it by | Downloaded on 2026-11-26 from | data-version | date | sha256 (uncompressed) |
| --- | --- | --- | --- | --- | --- |
| `psi-ms.obo.gz` | PSI-MS, `http://purl.obolibrary.org/obo/ms/psi-ms.obo` | `https://github.com/HUPO-PSI/psi-ms-CV/releases/download/v4.2.18/psi-ms.obo` (where the purl led that day; the same bytes as `psi-ms.obo` on the repository's master branch) | 4.2.18 | 19:11:2026 10:10 | `dc44e4de316d35f1ad62d018b921647c08ba9c36b5aae8cb970d733cd174a221` |
| `uo.obo.gz` | UO, `http://purl.obolibrary.org/obo/uo.obo` | `https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/uo.obo` (where the purl leads) | releases/2026-10-12 | none (owl:versionInfo 2026-10-12) | `ade7be5a515551d3d653bda8ad256563df322e83c96b41da4f848e8a470c2846` |

Where each purl leads is set by the OBO Foundry's PURL configuration
(`config/ms.yml` and `config/uo.yml` of https://github.com/OBOFoundry/purl.obolibrary.org).
The files lcmsdeconv writes name these versions in their `<cvList>`.

## Licences and attribution

- **PSI-MS** (https://github.com/HUPO-PSI/psi-ms-CV), as its header states: "This work is
  licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license"
  (https://creativecommons.org/licenses/by/4.0/). Publisher: HUPO Proteomics Standards Initiative
  Mass Spectrometry Standards Working Group and HUPO Proteomics Standards Initiative Proteomics
  Informatics Working Group. The repository has no licence file, so this header is the licence
  statement followed here; the OBO Foundry registry entry for the vocabulary (`ms`) names CC BY
  3.0 instead. Its maintainers ask users to cite: Mayer G, Montecchi-Palazzi L, Ovelleiro D,
  Jones AR, Binz PA, Deutsch EW, Chambers M, Kallhardt M, Levander F, Shofstahl J, Orchard S,
  Vizcaíno JA, Hermjakob H, Stephan C, Meyer HE, Eisenacher M; HUPO-PSI Group. The HUPO
  proteomics standards initiative- mass spectrometry controlled vocabulary. Database (Oxford).
  2013;2013:bat009. doi:10.1093/database/bat009.
- **UO**, "Units of measurement ontology"
  (https://github.com/bio-ontology-research-group/unit-ontology), as its header states:
  `dcterms:license https://creativecommons.org/licenses/by/4.0/` (CC BY 4.0).

Both are redistributed unchanged apart from the gzip compression.

## Refreshing

```bash
python scripts/refresh_vocabularies.py
```

downloads both from where their purls lead into a temporary directory, checks that psims parses
each into a vocabulary holding the terms lcmsdeconv writes mzML with, and only then writes both
here, gzipped the same way; it prints each one's data-version, date, licence lines and sha256.
`--from DIR` takes `psi-ms.obo` and `uo.obo` downloaded into `DIR` instead. After a refresh, put
the new versions, dates and checksums in the table above, in the top-level README.md and in
`tests/test_vocabularies.py`, which checks them. Nothing reads the vocabularies from the network
at run time.
