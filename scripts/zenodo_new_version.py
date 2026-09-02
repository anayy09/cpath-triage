"""
scripts/zenodo_new_version.py

Publish a new version of the Zenodo record, replacing the code-only archive with
the full deposit that includes the prediction artifacts (revision item C-E2).

Why this exists. The v2.0 record was created through the GitHub integration,
which archives only what git tracks. `results/` is gitignored, so that record
holds a 177 kB source zip and none of the parquet files every table is computed
from. Zenodo does not allow adding files to a published version, so the fix is a
new version under the same concept DOI. Doing it through a script rather than
the web UI means the exact payload is reviewable before anything is published,
and the step is repeatable if it has to be done again.

Safety. This does nothing irreversible unless `--publish` is passed. Without it
the script creates the draft, uploads the file and sets the metadata, then stops
and prints the draft URL for inspection. A draft can be deleted; a published
version cannot.

Authentication. Reads ZENODO_TOKEN from the environment or from .env. The token
needs the `deposit:write` and `deposit:actions` scopes. Create one at
https://zenodo.org/account/settings/applications/tokens/new/ . The token is
never printed, and it is never passed on the command line where it would land in
shell history.

Usage:
    # Inspect what the current record holds, change nothing
    python scripts/zenodo_new_version.py --inspect

    # Create the draft, upload, set metadata, and stop before publishing
    python scripts/zenodo_new_version.py --version v2.1

    # The same, then publish
    python scripts/zenodo_new_version.py --version v2.1 --publish
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE = "https://zenodo.org/api"
RECORD_ID = "22245990"          # the published v2.0 version
CONCEPT_ID = "22245989"         # concept record, stable across versions
DEFAULT_ZIP = PROJECT_ROOT / "build" / "zenodo_deposit.zip"
TIMEOUT = httpx.Timeout(60.0, read=600.0, write=600.0)


def load_token() -> str:
    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        env = PROJECT_ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip().startswith("ZENODO_TOKEN"):
                    _, _, val = line.partition("=")
                    token = val.strip().strip('"').strip("'")
                    break
    if not token:
        raise SystemExit(
            "ZENODO_TOKEN is not set.\n"
            "Create a token with the deposit:write and deposit:actions scopes at\n"
            "  https://zenodo.org/account/settings/applications/tokens/new/\n"
            "then either add a ZENODO_TOKEN line to .env or set it for the session:\n"
            '  $env:ZENODO_TOKEN = "..."'
        )
    return token


def show(resp: httpx.Response, what: str) -> dict:
    if resp.status_code >= 400:
        raise SystemExit(f"{what} failed with HTTP {resp.status_code}:\n{resp.text[:1200]}")
    return resp.json() if resp.content else {}


def summarize(dep: dict, label: str) -> None:
    files = dep.get("files", [])
    print(f"\n{label}")
    print(f"  id        : {dep.get('id')}")
    print(f"  state     : {dep.get('state')}  submitted={dep.get('submitted')}")
    print(f"  doi       : {dep.get('doi') or dep.get('metadata', {}).get('prereserve_doi', {}).get('doi', '(reserved on publish)')}")
    print(f"  version   : {dep.get('metadata', {}).get('version')}")
    print(f"  title     : {dep.get('metadata', {}).get('title', '')[:70]}")
    print(f"  files     : {len(files)}")
    for f in files:
        size = f.get("filesize") or f.get("size") or 0
        print(f"     {size:>12,}  {f.get('filename') or f.get('key')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect", action="store_true",
                    help="Print the current record and exit. Changes nothing.")
    ap.add_argument("--version", default=None,
                    help="Version label for the new release, e.g. v2.1.")
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--upload-name", default=None,
                    help="Filename to store on Zenodo. Defaults to cpath-triage-{version}-code-and-artifacts.zip")
    ap.add_argument("--keep-existing-files", action="store_true",
                    help="Keep the inherited code-only zip. Off by default, since the new "
                         "archive already contains the code.")
    ap.add_argument("--publish", action="store_true",
                    help="Publish. Irreversible: a published version cannot be deleted.")
    args = ap.parse_args()

    token = load_token()
    auth = {"Authorization": f"Bearer {token}"}

    with httpx.Client(timeout=TIMEOUT, headers=auth, follow_redirects=True) as c:
        current = show(c.get(f"{BASE}/deposit/depositions/{RECORD_ID}"), "Reading the record")
        summarize(current, f"Current record {RECORD_ID} (concept {CONCEPT_ID})")
        if args.inspect:
            return 0

        if not args.version:
            raise SystemExit("\n--version is required unless --inspect is given.")
        if not args.zip.exists():
            raise SystemExit(f"\nArchive not found: {args.zip}\n"
                             "Build it with: python scripts/make_deposit.py --zip")

        upload_name = args.upload_name or f"cpath-triage-{args.version}-code-and-artifacts.zip"
        size = args.zip.stat().st_size
        print(f"\nWill upload {args.zip.name} as {upload_name} ({size:,} bytes)")

        new = show(c.post(f"{BASE}/deposit/depositions/{RECORD_ID}/actions/newversion"),
                   "Creating the new version")
        draft_url = new["links"]["latest_draft"]
        draft = show(c.get(draft_url), "Fetching the draft")
        draft_id = draft["id"]
        print(f"Draft created: {draft_id}")

        if not args.keep_existing_files:
            for f in draft.get("files", []):
                fid = f.get("id")
                r = c.delete(f"{BASE}/deposit/depositions/{draft_id}/files/{fid}")
                if r.status_code >= 400:
                    raise SystemExit(f"Removing inherited file failed: {r.status_code} {r.text[:400]}")
                print(f"Removed inherited file: {f.get('filename')}")

        bucket = draft["links"]["bucket"]
        print(f"Uploading {size:,} bytes ...")
        with args.zip.open("rb") as fh:
            show(c.put(f"{bucket}/{upload_name}", content=fh), "Uploading the archive")
        print("Upload complete.")

        meta = dict(draft["metadata"])
        meta["version"] = args.version
        meta.pop("doi", None)          # let Zenodo mint the new version DOI
        meta.pop("prereserve_doi", None)
        show(c.put(f"{BASE}/deposit/depositions/{draft_id}",
                   json={"metadata": meta}), "Setting metadata")

        draft = show(c.get(f"{BASE}/deposit/depositions/{draft_id}"), "Re-reading the draft")
        summarize(draft, "Draft ready")

        if not args.publish:
            print("\nNot published. Review the draft, then rerun with --publish:")
            print(f"  https://zenodo.org/uploads/{draft_id}")
            return 0

        published = show(c.post(f"{BASE}/deposit/depositions/{draft_id}/actions/publish"),
                         "Publishing")
        summarize(published, "Published")
        print(f"\nConcept DOI (cite this): 10.5281/zenodo.{CONCEPT_ID}")
        print(f"Version DOI            : {published.get('doi')}")
        print("\nNow update: main.tex Code availability, CITATION.cff version/date/doi,")
        print("docs/RESPONSE_EVIDENCE_SR_R1.md, and E2 of the response letter.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
