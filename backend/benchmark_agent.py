# benchmark_agent.py
#
# Run this from inside your Sales_Outreach_Agent/backend folder
# (same place as app.py), with your .env (GOOGLE_API_KEY, GROQ_API_KEY) set up.
#
#   python benchmark_agent.py
#
# It runs your real agent.graph over every lead in a CSV, times each run,
# and writes two files:
#   - benchmark_log.csv     -> one row per lead, with timing + scores
#   - benchmark_summary.txt -> the aggregate numbers you can quote on your resume
#
# Every number below is computed from what actually happened when you ran
# this on your machine, so you can explain exactly how you got it.

import time
import csv
import statistics
from app import agent, _initial_state  # reuses your real graph, unmodified

CSV_PATH = "../synthetic_leads.csv"   # swap in a bigger CSV if you want more data points
SENDER_COMPANY = "BrightPath Academy"
SENDER_ROLE = "Admissions Advisor"

LOG_ROWS = []


def run_one_lead(lead: dict) -> dict:
    start = time.perf_counter()
    result = agent.graph.invoke(_initial_state(lead, SENDER_COMPANY, SENDER_ROLE))
    elapsed = time.perf_counter() - start

    checks = result["checks"]
    avg_check_score = statistics.mean(c["score"] for c in checks) if checks else 0
    first_pass = result["revision_count"] == 0 and result["status"] == "approved"

    return {
        "contact_name": lead.get("contact_name", ""),
        "status": result["status"],
        "revisions": result["revision_count"],
        "first_pass_approval": first_pass,
        "avg_check_score": round(avg_check_score, 1),
        "spam_score": next((c["score"] for c in checks if c["check"] == "spam_check"), None),
        "personalization_score": next((c["score"] for c in checks if c["check"] == "personalization_check"), None),
        "tone_score": next((c["score"] for c in checks if c["check"] == "tone_check"), None),
        "time_seconds": round(elapsed, 2),
    }


def main():
    import pandas as pd
    leads = pd.read_csv(CSV_PATH).fillna("").to_dict(orient="records")

    print(f"Running agent on {len(leads)} leads...\n")
    for i, lead in enumerate(leads, 1):
        row = run_one_lead(lead)
        LOG_ROWS.append(row)
        print(f"[{i}/{len(leads)}] {row['contact_name']}: {row['status']} "
              f"(revisions={row['revisions']}, {row['time_seconds']}s)")

    # ---- write per-lead CSV log ----
    with open("benchmark_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_ROWS[0].keys())
        writer.writeheader()
        writer.writerows(LOG_ROWS)

    # ---- compute aggregate stats ----
    total = len(LOG_ROWS)
    approved = sum(1 for r in LOG_ROWS if r["status"] == "approved")
    flagged = sum(1 for r in LOG_ROWS if r["status"] == "flagged")
    first_pass = sum(1 for r in LOG_ROWS if r["first_pass_approval"])
    avg_revisions = statistics.mean(r["revisions"] for r in LOG_ROWS)
    avg_time = statistics.mean(r["time_seconds"] for r in LOG_ROWS)
    total_time = sum(r["time_seconds"] for r in LOG_ROWS)
    avg_check_score = statistics.mean(r["avg_check_score"] for r in LOG_ROWS)

    summary = f"""
BENCHMARK SUMMARY  (n = {total} leads, from {CSV_PATH})
=========================================================
Approval rate:              {approved}/{total}  ({approved/total*100:.1f}%)
Flagged rate:                {flagged}/{total}  ({flagged/total*100:.1f}%)
First-pass approval rate:   {first_pass}/{total}  ({first_pass/total*100:.1f}%)
Avg. revision cycles/lead:  {avg_revisions:.2f}
Avg. quality-check score:   {avg_check_score:.1f}/100
Avg. time per lead:         {avg_time:.2f}s
Total time for batch:       {total_time:.2f}s ({total_time/60:.2f} min)
=========================================================

How to cite these on your resume (example):
"Achieved a {first_pass/total*100:.0f}% first-pass approval rate across a
{total}-lead test batch, with an average of {avg_revisions:.1f} revision
cycles per email before approval."
"""
    print(summary)
    with open("benchmark_summary.txt", "w") as f:
        f.write(summary)

    print("Wrote benchmark_log.csv and benchmark_summary.txt")


if __name__ == "__main__":
    main()