# Filovirus clinical question review

**Status: draft; independent clinician review pending.** The 15 cases in `cases.json` are source-anchored screening questions prepared from the local PDFs. Their expected concepts and prohibited claims are proposed checks, not approved clinical guidance. Automated claim checks and case screening cannot establish clinical correctness.

## Review workflow

1. Confirm that every source anchor points to the stated PDF page:

   ```bash
   python scripts/clinical_review.py validate
   ```

2. Run the bot against the draft set and inspect `evaluation_results/clinical_screen.json`:

   ```bash
   python scripts/clinical_review.py evaluate
   ```

3. Give `cases.json`, the source PDFs, and a copy of `reviewer_template.csv` to **two independent clinicians**. Each clinician should read the cited page in the PDF and review the question, expected concepts, prohibited claims, and the bot's answer. In the CSV, enter their own identity, `reviewer_role=clinician`, `yes` or `no` for each check, `decision=approve` or `reject`, an ISO date/time, and notes. A reviewer should mark `reject` when a case is ambiguous, outdated, unsafe, or unsupported by its source. Save the completed file as `clinical_review/reviews.csv`.

4. After correcting rejected cases and obtaining two distinct approvals per case, run:

   ```bash
   python scripts/clinical_review.py gate --reviews clinical_review/reviews.csv
   ```

The gate requires all checks and two named clinician approvals for every case. It rejects reviews tied to an older case set or source PDF revision. A failed gate is the expected result until human reviews are complete. If case expectations change, generate a fresh template with `python scripts/clinical_review.py prepare --reviews clinical_review/new_template.csv`; do not overwrite completed reviews.

## Reading results

The automatic screen checks response status, key concepts, prohibited phrases, and whether the anchored source was retrieved. It also records the claim-level model check. A passing automatic screen means only that these machine checks passed. Reviewers must inspect whether each medical claim follows from its cited text, including population, timing, uncertainty, and qualifiers.
