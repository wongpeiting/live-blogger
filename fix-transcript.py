#!/usr/bin/env python3
"""
fix-transcript.py — Real-time transcript cleanup.

Watches transcript.jsonl and fixes names, terms, abbreviations as new lines
appear. Run alongside live-pipeline.py.

Corrections live in corrections.json. Edit that file anytime — the watcher
reloads it automatically so you can add fixes mid-session.

Usage:
    python3 fix-transcript.py --watch             # watch mode (run alongside pipeline)
    python3 fix-transcript.py                     # one-shot fix
    python3 fix-transcript.py --dry-run           # show what would change
    python3 fix-transcript.py --add-name "Wirtz Chia=Edward Chia"
    python3 fix-transcript.py --add-term "grit stability=grid stability"
    python3 fix-transcript.py --add-abbrev "C P F=CPF"
"""

import argparse
import json
import os
import re
import signal
import sys
import time

DEFAULT_CORRECTIONS_FILE = "corrections.json"
DEFAULT_TRANSCRIPT_FILE = "transcript.jsonl"

DEFAULT_CORRECTIONS = {
    "_comment": "Edit this file to add corrections. Watcher reloads automatically.",
    "names": {
        "Wirtz Chia": "Edward Chia",
        "Wirt Chia": "Edward Chia",
    },
    "terms": {
        "grit stability": "grid stability",
        "Hormos": "Hormuz",
        "De Carbonization": "decarbonisation",
        "de carbonization": "decarbonisation",
        "Electrocity": "electricity",
        "electrocity": "electricity",
        "bio diversity": "biodiversity",
        "cyber security": "cybersecurity",
        "block chain": "blockchain",
        "Beps": "BEPS",
    },
    "abbreviations": {
        "L N G": "LNG",
        "L.N.G.": "LNG",
        "M T I": "MTI",
        "M.T.I.": "MTI",
        "A I": "AI",
        "A.I.": "AI",
        "E V": "EV",
        "E.V.": "EV",
        "C P F": "CPF",
        "H D B": "HDB",
        "G S T": "GST",
        "N S": "NS",
        "M O E": "MOE",
        "M O H": "MOH",
        "M O F": "MOF",
        "M N D": "MND",
        "M H A": "MHA",
        "M O T": "MOT",
        "L T A": "LTA",
        "E D B": "EDB",
        "J T C": "JTC",
        "N E A": "NEA",
        "P U B": "PUB",
        "S P F": "SPF",
        "S A F": "SAF",
        "M O M": "MOM",
        "N D P": "NDP",
        "S M E": "SME",
        "S M Es": "SMEs",
        "G R C": "GRC",
        "S M C": "SMC",
        "N C M P": "NCMP",
        "N M P": "NMP",
        "B T O": "BTO",
        "C O E": "COE",
        "P S L E": "PSLE",
        "I S D": "ISD",
        "M A S": "MAS",
        "P A P": "PAP",
        "W P": "WP",
        "P S P": "PSP",
        "S M E P O": "SME PO",
    },
}

shutdown = False


def signal_handler(sig, frame):
    global shutdown
    print("\n[fix] Stopping watcher...")
    shutdown = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_corrections(path: str) -> dict:
    """Load corrections from JSON file, creating default if missing."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(DEFAULT_CORRECTIONS, f, indent=2, ensure_ascii=False)
        print(f"[fix] Created {path} with default corrections.")
    with open(path) as f:
        return json.load(f)


def apply_corrections(text: str, corrections: dict) -> str:
    """Apply all corrections to a text string."""
    # 1. Name corrections (case-sensitive, exact match, longest first)
    for wrong, right in sorted(
        corrections.get("names", {}).items(), key=lambda x: -len(x[0])
    ):
        text = text.replace(wrong, right)

    # 2. Term corrections (case-insensitive, longest first)
    for wrong, right in sorted(
        corrections.get("terms", {}).items(), key=lambda x: -len(x[0])
    ):
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)

    # 3. Abbreviation normalisation (word-boundary, longest first)
    for wrong, right in sorted(
        corrections.get("abbreviations", {}).items(), key=lambda x: -len(x[0])
    ):
        text = re.sub(r"\b" + re.escape(wrong) + r"\b", right, text)

    return text


def count_lines(path: str) -> int:
    """Count lines in a file without reading them all into memory."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def read_lines(path: str) -> list[str]:
    """Read all lines from file."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: str, lines: list[str]):
    """Write lines back to file atomically (write tmp then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for line in lines:
            f.write(line + "\n")
    os.replace(tmp, path)


def fix_all(transcript_path: str, corrections: dict, dry_run: bool = False) -> int:
    """One-shot fix of entire transcript. Returns number of changes."""
    lines = read_lines(transcript_path)
    if not lines:
        return 0

    changes = 0
    fixed_lines = []

    for i, raw_line in enumerate(lines):
        if not raw_line.strip():
            fixed_lines.append(raw_line)
            continue
        entry = json.loads(raw_line)
        original = entry.get("text", "")
        fixed = apply_corrections(original, corrections)

        if fixed != original:
            changes += 1
            if dry_run:
                print(f"  [line {i+1}] {original[:100]}")
                print(f"         → {fixed[:100]}")
            entry["text"] = fixed

        fixed_lines.append(json.dumps(entry, ensure_ascii=False))

    if not dry_run and changes > 0:
        write_lines(transcript_path, fixed_lines)

    return changes


def watch_and_fix(transcript_path: str, corrections_path: str):
    """Watch transcript.jsonl and fix new lines as they appear.

    Also reloads corrections.json when it changes, so you can add
    new fixes mid-session without restarting.
    """
    global shutdown

    corrections = load_corrections(corrections_path)
    corrections_mtime = os.path.getmtime(corrections_path) if os.path.exists(corrections_path) else 0
    last_line_count = 0

    # First pass: fix everything that's already there
    if os.path.exists(transcript_path):
        changes = fix_all(transcript_path, corrections)
        last_line_count = count_lines(transcript_path)
        if changes:
            print(f"[fix] Initial pass: fixed {changes} line(s)")
        else:
            print(f"[fix] Initial pass: {last_line_count} line(s), no changes needed")

    print(f"[fix] Watching {transcript_path} (poll every 2s)")
    print(f"[fix] Corrections: {corrections_path} (auto-reloads on change)")
    print(f"[fix] Ctrl+C to stop\n")

    while not shutdown:
        time.sleep(2)

        # Check if corrections.json changed — reload if so
        if os.path.exists(corrections_path):
            mtime = os.path.getmtime(corrections_path)
            if mtime != corrections_mtime:
                corrections_mtime = mtime
                corrections = load_corrections(corrections_path)
                print(f"[fix] Reloaded corrections.json — re-fixing entire transcript")
                changes = fix_all(transcript_path, corrections)
                last_line_count = count_lines(transcript_path)
                if changes:
                    print(f"[fix] Re-fix: {changes} line(s) changed")
                continue

        # Check for new lines
        if not os.path.exists(transcript_path):
            continue

        current_count = count_lines(transcript_path)
        if current_count <= last_line_count:
            continue

        # New lines appeared — fix them
        lines = read_lines(transcript_path)
        new_start = last_line_count
        changes = 0
        modified = False

        for i in range(new_start, len(lines)):
            raw = lines[i].strip()
            if not raw:
                continue
            entry = json.loads(raw)
            original = entry.get("text", "")
            fixed = apply_corrections(original, corrections)
            if fixed != original:
                changes += 1
                entry["text"] = fixed
                lines[i] = json.dumps(entry, ensure_ascii=False)
                modified = True
                print(f"[fix] Line {i+1}: {original[:80]}")
                print(f"            → {fixed[:80]}")

        if modified:
            write_lines(transcript_path, lines)
            print(f"[fix] Fixed {changes} new line(s)")

        last_line_count = len(lines)

    print("[fix] Watcher stopped")


def add_correction(corrections_path: str, category: str, pair: str):
    """Add a correction to corrections.json. Format: 'wrong=right'."""
    if "=" not in pair:
        print(f"[fix] Invalid format. Use: wrong=right")
        sys.exit(1)
    wrong, right = pair.split("=", 1)
    corrections = load_corrections(corrections_path)
    corrections[category][wrong] = right
    with open(corrections_path, "w") as f:
        json.dump(corrections, f, indent=2, ensure_ascii=False)
    print(f"[fix] Added {category}: '{wrong}' → '{right}'")
    print(f"[fix] Watcher will auto-reload and re-fix transcript.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time transcript cleanup — run alongside live-pipeline.py"
    )
    parser.add_argument(
        "-f", "--file", default=DEFAULT_TRANSCRIPT_FILE,
        help=f"Transcript JSONL file. Default: {DEFAULT_TRANSCRIPT_FILE}",
    )
    parser.add_argument(
        "-c", "--corrections", default=DEFAULT_CORRECTIONS_FILE,
        help=f"Corrections JSON file. Default: {DEFAULT_CORRECTIONS_FILE}",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch mode: continuously fix new lines as they appear",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without modifying the file",
    )
    parser.add_argument(
        "--add-name", metavar="WRONG=RIGHT",
        help="Add a name correction. E.g. 'Wirtz Chia=Edward Chia'",
    )
    parser.add_argument(
        "--add-term", metavar="WRONG=RIGHT",
        help="Add a term correction. E.g. 'grit stability=grid stability'",
    )
    parser.add_argument(
        "--add-abbrev", metavar="WRONG=RIGHT",
        help="Add an abbreviation correction. E.g. 'C P F=CPF'",
    )

    args = parser.parse_args()

    if args.add_name:
        add_correction(args.corrections, "names", args.add_name)
        return
    if args.add_term:
        add_correction(args.corrections, "terms", args.add_term)
        return
    if args.add_abbrev:
        add_correction(args.corrections, "abbreviations", args.add_abbrev)
        return

    if args.watch:
        watch_and_fix(args.file, args.corrections)
    else:
        corrections = load_corrections(args.corrections)
        if args.dry_run:
            changes = fix_all(args.file, corrections, dry_run=True)
            print(f"\n[fix] Dry run: {changes} line(s) would change")
        else:
            changes = fix_all(args.file, corrections)
            if changes:
                print(f"[fix] Fixed {changes} line(s)")
            else:
                print(f"[fix] No changes needed")


if __name__ == "__main__":
    main()
