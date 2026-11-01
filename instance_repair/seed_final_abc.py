"""Terminal seed handoff: remove exact non-window ABC anchors, never insert seeds."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math

POLICY = "final-abc-remove-a-only-keep-windows-20261101"
WINDOW_CLASSES = frozenset(("windowpane", "blind"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),
                         encoding="utf-8")
    temporary.replace(path)


def passes(record):
    """Recompute the agreed test from measured counts; never trust a cached decision."""
    ids = [record[k] for k in ("A", "B", "C")]
    require(len(set(ids)) == 3, "ABC requires three distinct mask identities")
    sizes = record["metric"]["areas"]
    a, b, c, union, covered, overlap = [
        sizes[k] for k in ("A", "B", "C", "union_BC", "covered_A", "intersection_BC")]
    require(all(type(n) is int and n >= 0 for n in (a, b, c, union, covered, overlap)),
            "Invalid projected pixel count")
    require(overlap <= min(b, c) and union == b + c - overlap
            and covered <= min(a, union), "Inconsistent projected counts")
    fractions = record["view_choice"]["visible_fractions"]
    require(set(fractions) == set(ids), "Incomplete source visibility")
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
                for v in fractions.values()), "Invalid visibility")
    require(record["view_choice"]["frame"] == record["target_frame"],
            "Target frame mismatch")
    return (min(a, b, c) > 0 and a >= max(b, c)
            and min(fractions.values()) >= .8
            and 10 * covered > 7 * a and 10 * covered > 7 * union
            and 10 * overlap < min(b, c))


def filter_final(manifest, search, semantics, seeds):
    """Pure, ordered subset operation, after all original selection and preflight."""
    require(manifest.get("status") in ("complete", "automatic_v2_complete"),
            "Finalize original semantic/v2 selection and preflight first")
    require("final_abc" not in manifest, "ABC was already applied; use the original final list")
    require(search.get("status") == "PASS", "Incomplete ABC evidence")
    groups = manifest["groups"]
    ids = [g["primary"]["id"] for g in groups]
    require(len(ids) == len(set(ids)), "Duplicate final mask ID")
    require(len({g["id"] for g in groups}) == len(groups), "Duplicate final group ID")
    for group in groups:
        primary = group["primary"]
        mid = primary["id"]
        require(mid in seeds and mid in semantics, "Missing mask or per-mask semantic evidence")
        require(isinstance(semantics[mid].get("top_class"), str)
                and bool(semantics[mid]["top_class"]), "Missing per-mask class")
        for key in ("id", "frame", "mask", "mask_sha256", "digest", "support_sha256"):
            require(primary[key] == seeds[mid][key], "Final mask identity changed: " + mid + "/" + key)
    anchors = {}
    for anchor in search["anchors"]:
        a = anchor["A"]
        require(a in seeds and a not in anchors, "Unknown or repeated ABC anchor")
        require(anchor.get("exhaustive") is True
                and anchor["tested_pairs"] == anchor["possible_pairs"]
                == len(anchor["evaluations"]), "Incomplete pair search")
        pairs = set()
        for r in anchor["evaluations"]:
            require(r["A"] == a, "Wrong evidence anchor")
            pair = tuple(sorted((r["B"], r["C"])))
            require(pair not in pairs, "Repeated BC evidence")
            pairs.add(pair)
            for mid in (a, r["B"], r["C"]):
                require(mid in seeds and mid in semantics, "Unknown evidence mask")
                require(semantics[mid]["exclude"] is False, "Semantically ineligible evidence")
                require(r["mask_hashes"][mid] == seeds[mid]["mask_sha256"],
                        "Stale ABC evidence for " + mid)
            passes(r)  # Validate counts even for an unsuccessful triple.
        anchors[a] = anchor
    removed = []
    protected = []
    witnesses = {}
    for mid in ids:
        if semantics[mid]["top_class"] in WINDOW_CLASSES:
            protected.append(mid)
            continue
        # Exact A only. Duplicate-class aliases and evidence B/C are never promoted.
        hits = [r for r in anchors.get(mid, {}).get("evaluations", []) if passes(r)]
        if hits:
            removed.append(mid)
            witnesses[mid] = hits
    remove_set = set(removed)
    output = copy.deepcopy(manifest)
    output["groups"] = [copy.deepcopy(g) for g in groups if g["primary"]["id"] not in remove_set]
    if "objects" in manifest:
        require([o["mask_id"] for o in manifest["objects"]] == ids,
                "Object map and final groups disagree")
        output["objects"] = [copy.deepcopy(o) for o in manifest["objects"]
                             if o["mask_id"] not in remove_set]
    audit = dict(policy=POLICY, input_count=len(ids), output_count=len(output["groups"]),
                 removed_A=removed, window_unchanged=protected, added=[], restored=[],
                 rerun_downstream=False, witnesses=witnesses)
    output["final_abc"] = {k: v for k, v in audit.items() if k != "witnesses"}
    return output, audit


def load_context(path):
    path = Path(path).resolve()
    context = read(path)
    for key in ("grouping", "supports", "pairs", "semantics", "geometry",
                "mapping_config", "mask_root", "visible_root"):
        p = Path(context[key])
        context[key] = str((path.parent / p).resolve() if not p.is_absolute() else p)
    return context


def finalize(manifest_path, context_path, out):
    """Registered current pipeline entry. Boundary/selection precede this final gate."""
    from .seed_boundary import validate_selected as validate_boundary
    from .seed_abc_evidence import Search
    out = Path(out)
    require(not out.exists(), "Use a new output directory")
    context = load_context(context_path)
    validate_boundary(manifest_path, context["grouping"], context["mask_root"])
    source = read(manifest_path)
    source_hash = sha(manifest_path)
    out.mkdir(parents=True)
    builder = Search(context, out / "evidence")
    search = builder.run()
    result, audit = filter_final(source, search, builder.semantics, builder.seeds)
    require(sha(manifest_path) == source_hash, "Original final manifest changed")
    builder.check_sources()
    save(out / "original_final_manifest.json", source)
    save(out / "seed_manifest.json", result)
    save(out / "audit.json", audit)
    receipt = dict(policy=POLICY, context=context, source_manifest=str(Path(manifest_path).resolve()),
                   source_manifest_sha256=source_hash,
                   files={name: sha(out / name) for name in
                          ("original_final_manifest.json", "seed_manifest.json", "audit.json",
                           "evidence/search.json", "evidence/source_lock.json")},
                   implementation={str(p): sha(p) for p in implementation_files()})
    save(out / "receipt.json", receipt)
    return validate_selected(out / "seed_manifest.json", out / "receipt.json")


def implementation_files():
    root = Path(__file__).resolve().parent
    return [root / name for name in
            ("seed_final_abc.py", "seed_abc_evidence.py", "seed_abc_projection.py")]


def validate_selected(manifest_path, receipt_path):
    """Current tracking handoff must consume the checked post-ABC manifest."""
    from .seed_boundary import validate_selected as validate_boundary
    receipt_path = Path(receipt_path)
    root = receipt_path.parent
    receipt = read(receipt_path)
    require(receipt["policy"] == POLICY, "Wrong final seed policy")
    require(Path(manifest_path).resolve() == (root / "seed_manifest.json").resolve(),
            "Tracking must consume the post-ABC manifest")
    for name, expected in receipt["files"].items():
        require(sha(root / name) == expected, "Final artifact changed: " + name)
    expected_code = {p.name: sha(p) for p in implementation_files()}
    require({Path(p).name: h for p, h in receipt["implementation"].items()} == expected_code,
            "Final seed implementation differs from receipt")
    for path, expected in read(root / "evidence/source_lock.json").items():
        require(sha(path) == expected, "ABC source changed: " + path)
    require(sha(receipt["source_manifest"]) == receipt["source_manifest_sha256"],
            "Original final manifest changed")
    context = receipt["context"]
    grouping = read(context["grouping"])
    semantics = read(context["semantics"])["records"]
    result, audit = filter_final(read(root / "original_final_manifest.json"),
                                read(root / "evidence/search.json"), semantics,
                                {s["id"]: s for s in grouping["seeds"]})
    require(result == read(manifest_path) and audit == read(root / "audit.json"),
            "Final selection is not the exact approved subset")
    validate_boundary(manifest_path, context["grouping"], context["mask_root"])
    return dict(status="PASS", policy=POLICY, tracking_manifest=str(Path(manifest_path).resolve()),
                input_count=audit["input_count"], output_count=audit["output_count"],
                removed_A=audit["removed_A"], added=[], window_unchanged=audit["window_unchanged"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("finalize")
    for key in ("manifest", "context", "out"):
        q.add_argument("--" + key, required=True, type=Path)
    q = sub.add_parser("validate-selected")
    for key in ("manifest", "receipt"):
        q.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args(argv)
    result = (finalize(args.manifest, args.context, args.out) if args.command == "finalize"
              else validate_selected(args.manifest, args.receipt))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
