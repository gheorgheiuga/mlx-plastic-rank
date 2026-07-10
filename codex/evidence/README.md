# Evidence Snapshots

This directory stores compact review artifacts for PopRank experiments. These
files are small summaries that let reviewers see the measured claim without
committing generated datasets, model checkpoints, adapter packs, logs, or raw
`out/` directories.

Evidence snapshots are not release-grade datasets. Check each snapshot's
source dataset and license metadata before using a result for product or
commercial claims.

`fault_codes_paired_control_screen_seed42.json` is the current strongest local
artifact. It records a single-seed paired falsification screen with same-budget
random, shuffled, target-constant, and cross-domain controls. Its promotion gate
passed, but its evidence status remains diagnostic until repeated across seeds,
datasets, and base checkpoints.

`text_to_sql_fullscale_summary.json` records the completed 10,000-row
Text-to-SQL replication. It passed the original tradeoff gate; the random and
shuffled control candidates subsequently added to its spec remain unrun.
