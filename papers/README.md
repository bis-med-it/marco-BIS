# papers/: local staging area for source PDFs

This folder holds the source-paper PDFs used to draft and verify model
dictionary entries (Mode B staging and entry verification). **PDFs are
not distributed with the repository**. Published papers are typically
under copyright, so `.gitignore` excludes `papers/**/*.pdf` and each
user stages their own copies.

The `verified.pdf_location` field in each dictionary entry
(`skills/model-dictionary/entries/*.yaml`) records the local filename
the entry was verified against at verification time. To re-verify an
entry against its source, obtain the paper (publisher, working-paper
series, or your library) and save it here under that filename.

**Naming convention.** Each entry's PDF is named after its dictionary
entry id, so `verified.pdf_location` is `papers/<id>.pdf` (e.g.
`kilian_2009_oil_svar.pdf`, `uhlig_2005_sign_restrictions.pdf`). A few
entries also list supporting PDFs.
Supporting papers that are not themselves dictionary entries use a
descriptive `author_year_topic.pdf` name (e.g.
`blanchard_perotti_2002_fiscal_svar.pdf`, `lewis_2021_hetid_robust_inference.pdf`,
`mertens_ravn_2019_reply.pdf`).

Public-domain papers can travel with your own copy (with `git add -f`
if you keep it under version control).
