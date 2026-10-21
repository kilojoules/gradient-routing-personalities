#!/usr/bin/env python3
"""Rebuild every table and figure in this folder from scores/ and check (or update) the shipped copies.

    python analysis/reproduce.py            # rebuild in a temp dir; verify results/ (csv/json numerically, text up to last-digit rounding)
    python analysis/reproduce.py --update   # rebuild and overwrite results/ and figures/

Checks, in order: the exact t critical values against an independent computation; the SHA-256 and row counts of the three
score files against scores/MANIFEST.json; that the folder holds only expected file types; every shipped table against its
rebuild (CSV to 1e-9 relative, JSON exactly, text files token by token: decimals may differ by one unit in their last printed
digit, integers and everything else must match exactly); that results/ holds no file that this script neither rebuilds nor
lists as provenance; and the internal consistency of the provenance files (response_diversity.csv, the two pool summaries and
fnu_copy_counts.csv), which are counts-only summaries of local raw files and cannot be rebuilt here: their counts must add up,
response_diversity.csv must agree with the opener flags in scores/, the pool summary's repeated-text fingerprint must be one that
appears in response_diversity.csv, and every run that emits that text must be counted as exact copies in fnu_copy_counts.csv. This
does not verify them against the raw data. Figures are regenerated and must exist; their bytes are not
compared (matplotlib output can differ across versions), the numbers they plot are what is verified.

Needs only numpy and matplotlib (see requirements.txt). Nothing here touches a cluster or any raw text.
"""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
sys.path.insert(0, str(HERE))
import stats  # noqa: E402
from materialize_sidecars import materialize  # noqa: E402

TABLES = {  # produced file (relative to the build dir) -> shipped location (relative to the folder)
    "corner_fnu/figure_seed_values.csv": "results/corner_fnu/figure_seed_values.csv",
    "corner_fnu/figure_arm_summary.csv": "results/corner_fnu/figure_arm_summary.csv",
    "corner_fnu/figure_paired_contrasts.csv": "results/corner_fnu/figure_paired_contrasts.csv",
    "corner_fnu/figure_manifest.json": "results/corner_fnu/figure_manifest.json",
    "corner_fnu/figure_notes.txt": "results/corner_fnu/figure_notes.txt",
    "corner_fnu/judge_label_noise.txt": "results/corner_fnu/judge_label_noise.txt",
    "corner_fnu/fnu_overlap_sensitivity.txt": "results/corner_fnu/fnu_overlap_sensitivity.txt",
    "grid/grid_figure_seed_values.csv": "results/grid/grid_figure_seed_values.csv",
    "grid/grid_figure_contrasts.csv": "results/grid/grid_figure_contrasts.csv",
    "grid/grid_figure_manifest.json": "results/grid/grid_figure_manifest.json",
    "grid/grid_figure_notes.txt": "results/grid/grid_figure_notes.txt",
    "grid/grid_leak_split.csv": "results/grid/grid_leak_split.csv",
    "grid/grid_classifier_seed_values.csv": "results/grid/grid_classifier_seed_values.csv",
    "grid/readout_classifier.txt": "results/grid/readout_classifier.txt",
    "grid/readout_persona.txt": "results/grid/readout_persona.txt",
    "grid/opener_vs_toxicity.txt": "results/grid/opener_vs_toxicity.txt",
}
PROVENANCE = {  # shipped under results/ but built from raw files that are not in the repo, so not rebuilt here
    "results/response_diversity.csv", "results/corner_fnu/fnu_pool_summary.json", "results/training_pools_summary.json",
    "results/corner_fnu/fnu_copy_counts.csv",
}
FIGURES = {
    "grid/fig_grid_toxicity.png": "figures/fig_grid_toxicity.png",
    "grid/fig_grid_leak_split.png": "figures/fig_grid_leak_split.png",
    "grid/fig_grid_persona.png": "figures/fig_grid_persona.png",
    "grid/fig_grid_robustness.png": "figures/fig_grid_robustness.png",
    "grid/grid_results_figures.pdf": "figures/grid_results_figures.pdf",
    "corner_fnu/fig_fnu_deployment.png": "figures/fig_fnu_deployment.png",
    "corner_fnu/fig_mechanism.png": "figures/fig_mechanism.png",
    "corner_fnu/fig_absorption_contrasts.png": "figures/fig_absorption_contrasts.png",
    "corner_fnu/toxicity_absorption_figures.pdf": "figures/toxicity_absorption_figures.pdf",
}
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd, stdout_path=None):
    res = subprocess.run([sys.executable] + [str(c) for c in cmd], capture_output=True, text=True, cwd=str(FOLDER))
    if res.returncode:
        raise SystemExit("FAILED: %s\n%s\n%s" % (" ".join(map(str, cmd)), res.stdout[-1500:], res.stderr[-1500:]))
    if stdout_path:
        stats.write_text_lf(stdout_path, res.stdout)


def same_number(a, b, tol=1e-9):
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    return abs(x - y) <= tol * max(1.0, abs(x), abs(y))


def same_text(built, shipped):
    """Equal up to one unit in the last printed digit of every number (guards against rounding ties across Python versions)."""
    a, b = NUMBER.split(built), NUMBER.split(shipped)
    na, nb = NUMBER.findall(built), NUMBER.findall(shipped)
    if a != b or len(na) != len(nb):
        return False
    for x, y in zip(na, nb):
        if "e" in x.lower() or "e" in y.lower() or "." not in x or "." not in y:
            if x != y:  # counts, seeds and exponents must match exactly
                return False
        elif abs(float(x) - float(y)) > 1.01 * 10 ** (-len(y.split(".")[1])):
            return False
    return True


def compare(built, shipped):
    if not shipped.exists():
        return ["missing shipped file %s" % shipped.relative_to(FOLDER)]
    if built.suffix == ".csv":
        a = list(csv.reader(open(built, newline="")))
        b = list(csv.reader(open(shipped, newline="")))
        if len(a) != len(b):
            return ["%s: %d rows built vs %d shipped" % (shipped.name, len(a), len(b))]
        return ["%s row %d differs" % (shipped.name, i) for i, (r, s) in enumerate(zip(a, b))
                if len(r) != len(s) or not all(same_number(x, y) for x, y in zip(r, s))][:5]
    if built.suffix == ".json":
        return [] if json.load(open(built)) == json.load(open(shipped)) else ["%s differs" % shipped.name]
    return [] if same_text(built.read_text(), shipped.read_text()) else ["%s differs (beyond last-digit rounding)" % shipped.name]


def check_manifest():
    problems = []
    manifest = json.load(open(FOLDER / "scores" / "MANIFEST.json"))
    intact = True
    if set(manifest["files"]) != SCORES_DATA_FILES:
        return ["MANIFEST.json must list exactly %s, not %s" % (sorted(SCORES_DATA_FILES), sorted(manifest["files"]))]
    for name, digest in manifest["files"].items():
        if not (FOLDER / "scores" / name).exists():
            problems.append("scores/%s is missing" % name)
            intact = False
        elif sha256(FOLDER / "scores" / name) != digest:
            problems.append("scores/%s does not match its SHA-256 in MANIFEST.json" % name)
            intact = False
    if intact:
        try:
            n_clf = sum(1 for _ in gzip.open(FOLDER / "scores" / "classifier_scores.csv.gz", "rt")) - 1
            n_jdg = sum(1 for _ in gzip.open(FOLDER / "scores" / "judge_scores.jsonl.gz", "rt"))
            n_flg = sum(1 for _ in gzip.open(FOLDER / "scores" / "opener_flags.csv.gz", "rt")) - 1
        except (OSError, EOFError, UnicodeDecodeError) as err:
            return problems + ["scores/*.gz unreadable: %s" % err]
        if (n_clf, n_jdg, n_flg) != (manifest["rows"]["classifier_rows"], manifest["rows"]["judge_rows"], manifest["rows"]["opener_rows"]):
            problems.append("row counts %d/%d/%d differ from MANIFEST.json %s" % (n_clf, n_jdg, n_flg, manifest["rows"]))
    return problems


def short_strings_only(obj, where):
    """The counts-only JSON summaries must not hold prose: every string in them is a short identifier or key."""
    if isinstance(obj, dict):
        long_keys = ["%s holds a key of %d characters" % (where, len(k)) for k in obj if len(str(k)) > 64]
        return long_keys + [p for v in obj.values() for p in short_strings_only(v, where)]
    if isinstance(obj, list):
        return [p for v in obj for p in short_strings_only(v, where)]
    if isinstance(obj, str) and len(obj) > 48:
        return ["%s holds a string of %d characters (counts-only files must not hold prose)" % (where, len(obj))]
    return []


def check_pool_summaries(diversity_rows):
    """Internal consistency of the two counts-only pool summaries (counts must add up; the repeated-text fingerprint must be one
    that response_diversity.csv also shows). They cannot be rebuilt here, so this is all that can be checked."""
    problems = []
    fnu = json.load(open(FOLDER / "results" / "corner_fnu" / "fnu_pool_summary.json"))
    grid = json.load(open(FOLDER / "results" / "training_pools_summary.json"))

    def sized(label, d, n_key="n", shares="completion_starts_with"):
        n = d[n_key]
        if shares in d and sum(d[shares].values()) != n:
            problems.append("%s: opener counts do not add up to n" % label)
        for k in ("distinct_completions", "distinct_opening_keys", "with_evaluation_target", "with_evaluation_prompt_text", "classifier_B_at_least_0.5"):
            if k in d and not 0 <= d[k] <= n:
                problems.append("%s: %s out of range" % (label, k))
        if "judge_toxicity_counts" in d and sum(d["judge_toxicity_counts"].values()) != n:
            problems.append("%s: judge counts do not add up to n" % label)

    for k in ("caught_32", "slip_through_128_grid", "caught_64"):
        sized("training_pools_summary.%s" % k, grid[k])
    for k in ("corpus_first", "corpus_v34"):
        sized("training_pools_summary.%s" % k, grid[k])
    for k in ("slip_through_pool_all_candidates", "slip_through_used_128", "caught_64"):
        sized("fnu_pool_summary.%s" % k, fnu[k])
        dup = fnu[k].get("duplication")
        if dup and not (1 <= dup["largest_identical_group_size"] <= fnu[k]["n"] and 1 <= dup["distinct_completions"] <= fnu[k]["n"]
                        and dup["distinct_completions"] <= fnu[k]["n"] - dup["largest_identical_group_size"] + 1
                        and 1 <= dup["distinct_opening_keys"] <= fnu[k]["n"]):
            problems.append("fnu_pool_summary.%s: duplication counts out of range" % k)
    used = fnu["slip_through_used_128"]["duplication"]
    if sum(used["largest_identical_group_judge_toxicity_counts"].values()) != used["largest_identical_group_size"]:
        problems.append("fnu_pool_summary: judge counts of the repeated text do not add up to its group size")
    if used["largest_identical_group_rows_with_evaluation_prompt_text"] > fnu["evaluation_overlap"]["slip_through_used_128_with_evaluation_prompt_text"]:
        problems.append("fnu_pool_summary: more repeated-text rows overlap the evaluation prompts than rows overall")
    if used["largest_identical_group_size"] > fnu["slip_through_pool_all_candidates"]["duplication"]["largest_identical_group_size"]:
        problems.append("fnu_pool_summary: the 128 hold more copies of the repeated text than the whole pool")
    if (grid["caught_64"]["n"], grid["caught_64"]["selector_A"], grid["caught_64"]["completion_starts_with"]) != (
            fnu["caught_64"]["n"], fnu["caught_64"]["selector_A"], fnu["caught_64"]["completion_starts_with"]):
        problems.append("the 64 caught examples differ between the two pool summaries")
    fixed = {r["constant_fingerprint"] for r in diversity_rows if r["experiment"] == "fnu" and r["constant_fingerprint"]}
    if used["largest_identical_group_fingerprint"] not in fixed:
        problems.append("the repeated fn-U training text's fingerprint is not the fixed reply of any collapsed fn-U run in response_diversity.csv")
    return problems


def check_copy_counts(diversity_rows):
    """fnu_copy_counts.csv must cover exactly the fn-U runs, with counts that fit the number of replies, and every run that emits the
    pool's repeated text (response_diversity.csv: a single repeated reply with that fingerprint) must be all exact copies."""
    problems = []
    fnu = json.load(open(FOLDER / "results" / "corner_fnu" / "fnu_pool_summary.json"))
    fingerprint = fnu["slip_through_used_128"]["duplication"]["largest_identical_group_fingerprint"]
    runs = {(r["cell"], int(r["seed"]), r["battery"]): r for r in diversity_rows if r["experiment"] == "fnu"}
    copies = list(csv.DictReader(open(FOLDER / "results" / "corner_fnu" / "fnu_copy_counts.csv", newline="")))
    if {(c["cell"], int(c["seed"]), c["battery"]) for c in copies} != set(runs) or len(copies) != len(runs):
        return ["fnu_copy_counts.csv does not cover exactly the fn-U runs in response_diversity.csv"]
    for c in copies:
        key = (c["cell"], int(c["seed"]), c["battery"])
        n = int(c["n_replies"])
        counts = [int(c[k]) for k in ("n_exact_copy", "n_shorter_version", "n_longer_version", "n_contain_continuation_start")]
        if n != int(runs[key]["n_responses"]) or any(not 0 <= x <= n for x in counts) or sum(counts[:3]) > n:
            problems.append("fnu_copy_counts.csv: counts inconsistent for %s" % (key,))
        if runs[key]["constant_fingerprint"] == fingerprint and counts[0] != n:
            problems.append("fnu_copy_counts.csv: %s emits the repeated text in every reply but is not counted as all exact copies" % (key,))
    return problems


def check_provenance():
    """Self-consistency of the files built from raw data that is not in the repo (they cannot be rebuilt here)."""
    problems = []
    keys = {}
    text = gzip.open(FOLDER / "scores" / "classifier_scores.csv.gz", "rt", newline="").read()
    for r in csv.DictReader(text.splitlines()):
        if r["scoring"] == "dual":
            k = (r["experiment"], r["cell"], int(r["seed"]), r["battery"])
            keys[k] = keys.get(k, 0) + 1
    rows = list(csv.DictReader(open(FOLDER / "results" / "response_diversity.csv", newline="")))
    if {(r["experiment"], r["cell"], int(r["seed"]), r["battery"]) for r in rows} != set(keys) or len(rows) != len(keys):
        problems.append("response_diversity.csv does not cover exactly the runs in scores/")
    for r in rows:
        k = (r["experiment"], r["cell"], int(r["seed"]), r["battery"])
        if int(r["n_responses"]) != keys.get(k) or r["ids_match_scores"] != "True":
            problems.append("response_diversity.csv: n_responses or ids_match_scores inconsistent for %s" % (k,))
        single = int(r["n_distinct"]) == 1
        if single != bool(r["constant_fingerprint"]) or (single and not re.fullmatch(r"[0-9a-f]{8}", r["constant_fingerprint"])):
            problems.append("response_diversity.csv: fingerprint present iff n_distinct == 1 violated for %s" % (k,))
        if not 1 <= int(r["n_distinct"]) <= int(r["n_responses"]):
            problems.append("response_diversity.csv: n_distinct out of range for %s" % (k,))
    flags = {}
    ftext = gzip.open(FOLDER / "scores" / "opener_flags.csv.gz", "rt", newline="").read()
    for r in csv.DictReader(ftext.splitlines()):
        flags.setdefault((r["experiment"], r["cell"], int(r["seed"])), []).append(
            (int(r["opener_full"]), int(r["opener_start"]), int(r["opener_match_len"])))
    for r in rows:
        if r["battery"] != "tx":
            continue
        f = flags.get((r["experiment"], r["cell"], int(r["seed"])), [])
        n = max(len(f), 1)
        if (len(f) != int(r["n_responses"]) or abs(sum(x[0] for x in f) / n - float(r["frac_start_with_opener"])) > 5.1e-5
                or abs(sum(x[1] for x in f) / n - float(r["frac_start_with_opener_40"])) > 5.1e-5
                or any(x[0] > x[1] or x[1] != (x[2] >= 40) or (x[0] and x[2] < 45) or not 0 <= x[2] <= 55 for x in f)):
            problems.append("response_diversity.csv: opener fractions disagree with scores/opener_flags.csv.gz for %s" % (r["cell"] + " s" + r["seed"],))
    for name in ("results/corner_fnu/fnu_pool_summary.json", "results/training_pools_summary.json"):
        problems += short_strings_only(json.load(open(FOLDER / name)), name)
    problems += check_pool_summaries(rows)
    problems += check_copy_counts(rows)
    return problems


ALLOWED_TOP = {"README.md", "requirements.txt", ".gitignore", "analysis", "docs", "figures", "pipeline", "results", "scores"}
ALLOWED_SUFFIX = {  # per top-level directory; scores/ is an exact list of names instead
    "analysis": {".py"}, "pipeline": {".py", ".sh", ".md"}, "docs": {".md"}, "figures": {".png", ".pdf"},
    "results": {".csv", ".json", ".txt"},
}
SCORES_DATA_FILES = {"classifier_scores.csv.gz", "judge_scores.jsonl.gz", "opener_flags.csv.gz"}
SCORES_FILES = SCORES_DATA_FILES | {"MANIFEST.json", "README.md"}


def check_folder_contents(update=False):
    """No stray file types, top-level entries or large files (a raw generation file would fail here); figures/ must hold exactly
    the figures this script regenerates (a missing one is only an error in check mode, since --update recreates it)."""
    problems = []
    expected = {Path(v).name for v in FIGURES.values()}
    present = {f.name for f in (FOLDER / "figures").iterdir()} if (FOLDER / "figures").is_dir() else set()
    problems += ["unexpected file in figures/: %s" % n for n in sorted(present - expected)]
    if not update:
        problems += ["missing figure: figures/%s" % n for n in sorted(expected - present)]
    for entry in FOLDER.iterdir():
        if entry.name not in ALLOWED_TOP and not (entry.name in {"build", "__pycache__"} and entry.is_dir()):
            problems.append("unexpected top-level entry: %s" % entry.name)
    for f in FOLDER.rglob("*"):
        rel = f.relative_to(FOLDER)
        if f.is_dir() or {"build", "__pycache__"} & set(rel.parts[:-1]):
            continue
        top = rel.parts[0]
        if len(rel.parts) == 1:
            pass  # README.md, requirements.txt, .gitignore: named in ALLOWED_TOP
        elif top == "scores":
            if len(rel.parts) != 2 or rel.name not in SCORES_FILES:
                problems.append("unexpected file in scores/: %s" % rel)
        elif f.suffix not in ALLOWED_SUFFIX.get(top, set()):
            problems.append("unexpected file type: %s" % rel)
        if f.stat().st_size > 3_000_000:
            problems.append("file larger than 3 MB: %s" % rel)
    return problems


def check_no_stale_results():
    known = set(TABLES.values()) | PROVENANCE
    present = {str(p.relative_to(FOLDER)) for p in (FOLDER / "results").rglob("*") if p.is_file()}
    return ["results/ contains a file this script neither rebuilds nor lists as provenance: %s" % f
            for f in sorted(present - known)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="overwrite results/ and figures/ with the rebuilt files")
    ap.add_argument("--keep", help="keep the rebuild directory at this path instead of a temp dir")
    args = ap.parse_args()
    import matplotlib
    import numpy
    print("python %s, numpy %s, matplotlib %s" % (platform.python_version(), numpy.__version__, matplotlib.__version__))
    stats.check_constants()
    problems = check_manifest() + check_folder_contents(args.update)
    if problems:  # the inputs themselves are wrong: report cleanly instead of failing inside the rebuild
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.keep) if args.keep else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        sidecars = work / "sidecars"
        print("rebuilt %d sidecar files from scores/" % materialize(FOLDER / "scores", sidecars))
        run([HERE / "plot_absorption.py", "--data-dir", sidecars, "--out-dir", work / "corner_fnu"])
        run([HERE / "plot_grid_results.py", "--data", sidecars, "--out", work / "grid"])
        run([HERE / "analyze_grid_classifier.py", sidecars, work / "grid" / "grid_classifier_seed_values.csv"],
            work / "grid" / "readout_classifier.txt")
        run([HERE / "analyze_grid_persona.py", sidecars], work / "grid" / "readout_persona.txt")
        run([HERE / "analyze_fnu_judge.py", sidecars, "--diversity", FOLDER / "results" / "response_diversity.csv"],
            work / "corner_fnu" / "judge_label_noise.txt")
        run([HERE / "analyze_fnu_overlap.py", sidecars, "--pool-summary", FOLDER / "results" / "corner_fnu" / "fnu_pool_summary.json"],
            work / "corner_fnu" / "fnu_overlap_sensitivity.txt")
        run([HERE / "analyze_opener_vs_toxicity.py", sidecars], work / "grid" / "opener_vs_toxicity.txt")
        for built, shipped in TABLES.items():
            if args.update:
                dest = FOLDER / shipped
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(work / built, dest)
            else:
                problems += compare(work / built, FOLDER / shipped)
        for built, shipped in FIGURES.items():
            if not (work / built).exists():
                problems.append("figure not produced: %s" % built)
            elif args.update:
                dest = FOLDER / shipped
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(work / built, dest)
    problems += check_no_stale_results()
    if not problems:
        problems += check_provenance()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("updated results/ and figures/" if args.update else
          "OK: manifest and folder contents verified; all %d tables reproduce from scores/; provenance files self-consistent; figures regenerated" % len(TABLES))


if __name__ == "__main__":
    main()
