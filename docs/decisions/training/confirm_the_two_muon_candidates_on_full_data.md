# Confirm the two Muon candidates on full data

Date: 2026-08-13. Status: recorded in the decision log.

- Replace the planned 50%-data confirmation with a 100%-training-data
  confirmation at the user's direction. Train rank-64 Muon at the two rates
  advanced by the internal screen (`5e-5` and `1e-4`) for seeds 0, 1, and 2.
- Reuse the three recipe-identical full-data AdamW `5e-5` rank-64 cells rather
  than retraining them. Their source commit, metrics checksum, seeds, row count,
  and training recipe must pass the preparation contract.
- Since full-data training consumes the screen's internal-development pool, use
  the frozen 822-row competition validation artifact as a confirmatory surface.
  Freeze the two rates and the promotion gate before launch, prohibit further
  optimizer-rate tuning on these results, and leave the final test untouched.
