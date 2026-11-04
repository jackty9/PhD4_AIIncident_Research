"""
AIID full extraction via GraphQL (https://incidentdatabase.ai/api/graphql)

- Authenticates with your AIID API token (Bearer header), read from the
  AIID_TOKEN environment variable only.
- Pages by incident_id (sorted ASC, filter incident_id > last seen) instead of
  skip/limit, so no incidents are skipped when the database changes mid-pull
  (the cause of the 34 missing incidents in the 14 Aug 2026 pull).
- Saves the raw JSON plus two flat CSVs (incidents, reports).
- Runs completeness checks, optionally against an independent ID list
  (e.g. the AIID Excel export).

Usage:
  python scripts/aiid_extract.py [OUT_DIR] [--excel PATH]

  OUT_DIR defaults to data/aiid/. Requires the AIID_TOKEN environment
  variable to be set; the script aborts without making any request if it
  is not.
"""

import argparse
import json, os, sys, time, datetime as dt
import requests
import pandas as pd

# ---------------------------------------------------------------- settings
PAGE = 100                                       # incidents per request

TOKEN = os.environ.get("AIID_TOKEN")
if not TOKEN:
    print("ERROR: AIID_TOKEN environment variable is not set. Aborting "
          "without making any request. Set it and re-run.", file=sys.stderr)
    sys.exit(1)

URL = "https://incidentdatabase.ai/api/graphql"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "User-Agent": "aiid-research-extraction",
}

QUERY = """
query Page($after: Int!, $limit: Int!) {
  incidents(
    filter: { incident_id: { GT: $after } }
    sort: { incident_id: ASC }
    pagination: { limit: $limit }
  ) {
    incident_id
    title
    description
    date
    date_modified
    AllegedDeployerOfAISystem { name }
    AllegedDeveloperOfAISystem { name }
    AllegedHarmedOrNearlyHarmedParties { name }
    implicated_systems { name }
    reports {
      report_number
      title
      date_published
      url
      source_domain
      language
    }
    classifications {
      namespace
      publish
      attributes { short_name value_json }
    }
  }
}
"""


def post(variables, retries=5):
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers=HEADERS,
                              json={"query": QUERY, "variables": variables},
                              timeout=180)
            j = r.json()
            if "errors" in j:
                raise RuntimeError(json.dumps(j["errors"])[:500])
            return j["data"]["incidents"]
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  attempt {attempt + 1} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("giving up after repeated failures")


# ---------------------------------------------------------------- pull
def pull_all():
    out, after, limit = [], 0, PAGE
    while True:
        try:
            batch = post({"after": after, "limit": limit})
        except RuntimeError:
            if limit > 10:            # heavy CSET pages can time out: shrink
                limit = max(10, limit // 2)
                print(f"  reducing page size to {limit}")
                continue
            raise
        if not batch:
            break
        out.extend(batch)
        after = batch[-1]["incident_id"]
        print(f"fetched {len(out):>5}  (up to incident_id {after})")
        limit = PAGE
    return out


# ---------------------------------------------------------------- flatten
def names(lst):
    return "; ".join(x["name"] for x in (lst or []) if x.get("name"))


def decode(v):
    try:
        v = json.loads(v)
    except Exception:
        return v
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return v


def flatten(raw):
    inc_rows, rep_rows = [], []
    for i in raw:
        row = {
            "incident_id": i["incident_id"],
            "title": i["title"],
            "description": i["description"],
            "date": i["date"],
            "date_modified": i["date_modified"],
            "deployer": names(i["AllegedDeployerOfAISystem"]),
            "developer": names(i["AllegedDeveloperOfAISystem"]),
            "harmed": names(i["AllegedHarmedOrNearlyHarmedParties"]),
            "implicated_systems": names(i["implicated_systems"]),
            "report_count": len(i["reports"] or []),
            "namespaces": "; ".join(sorted({c["namespace"] for c in i["classifications"] or []})),
        }
        # one column per namespace__attribute (MIT, CSETv1, GMF, ...);
        # per-annotator CSET namespaces are kept only as a count
        for c in i["classifications"] or []:
            ns = c["namespace"]
            if ns.startswith("CSETv1_Annotator"):
                continue
            for a in c["attributes"] or []:
                row[f"{ns}__{a['short_name']}"] = decode(a["value_json"])
        row["n_cset_annotators"] = sum(
            c["namespace"].startswith("CSETv1_Annotator") for c in i["classifications"] or [])
        inc_rows.append(row)

        for r in i["reports"] or []:
            rep_rows.append({"incident_id": i["incident_id"], **r})

    inc = pd.DataFrame(inc_rows).sort_values("incident_id")
    rep = pd.DataFrame(rep_rows)
    return inc, rep


# ---------------------------------------------------------------- checks
def checks(inc, excel_export=None):
    """Runs completeness checks. Returns True iff every check passes (no
    real gaps, no Excel-export IDs missing from the pull)."""
    ok = True
    ids = inc["incident_id"]
    print("\n=== completeness checks ===")
    print("incidents:", len(inc), "| unique:", ids.nunique(), "| max id:", ids.max())
    gaps = sorted(set(range(1, ids.max() + 1)) - set(ids))
    print("ID gaps:", len(gaps), gaps[:60])

    # independent confirmation that each gap really is absent
    back = []
    if gaps:
        q = """query($ids:[Int]){ incidents(filter:{incident_id:{IN:$ids}}){ incident_id } }"""
        r = requests.post(URL, headers=HEADERS,
                          json={"query": q, "variables": {"ids": gaps}}, timeout=120).json()
        back = [x["incident_id"] for x in r["data"]["incidents"]]
    print("gap IDs that DO exist (should be none):", back)
    if back:
        ok = False

    excel_missing = []
    if excel_export:
        x = pd.read_excel(excel_export, header=2)
        xid = set(pd.to_numeric(x["Incident ID"], errors="coerce").dropna().astype(int))
        excel_missing = sorted(xid - set(ids))
        print("in Excel export but not in this pull (should be none):", excel_missing)
        if excel_missing:
            ok = False

    inc["year"] = pd.to_datetime(inc["date"], errors="coerce").dt.year
    mit = inc["namespaces"].str.contains(r"\bMIT\b", na=False)
    cov = inc.assign(mit=mit).groupby("year")["mit"].agg(["size", "sum", "mean"])
    print("\nMIT coverage by year (2018+):")
    print(cov.loc[cov.index >= 2018].round(3).to_string())

    return ok


# ---------------------------------------------------------------- run
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", default="data/aiid/",
                         help="Output directory for the raw JSON and CSVs (default: data/aiid/)")
    parser.add_argument("--excel", default=None,
                         help="Path to an AIID Excel export (header row 3, 'Incident ID' column) "
                              "to cross-check pulled IDs against")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.date.today().isoformat()

    raw = pull_all()
    with open(f"{out_dir}/aiid_raw_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"extracted": stamp, "incidents": raw}, f, ensure_ascii=False)

    inc, rep = flatten(raw)
    inc.to_csv(f"{out_dir}/aiid_incidents_{stamp}.csv", index=False)
    rep.to_csv(f"{out_dir}/aiid_reports_{stamp}.csv", index=False)
    print(f"\nsaved {len(inc)} incidents and {len(rep)} report links to {out_dir}")

    passed = checks(inc, excel_export=args.excel)
    if not passed:
        print("\nCOMPLETENESS CHECK FAILED -- see output above.", file=sys.stderr)
        sys.exit(1)
