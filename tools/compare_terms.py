import json
import pathlib
import re
import sys

FIXTURE = pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "upstream_terms.json"
SOURCE = pathlib.Path(__file__).parent.parent / "microdux"

WEIGHTS = {
    "Velocity": ("rewards.py", "Weights"),
    "StandUp": ("rewards.py", "StandUpWeights"),
    "SitStand": ("rewards.py", "SitStandWeights"),
    "Spin": ("spin.py", "Weights"),
    "Swizzle": ("swizzle.py", "Weights"),
    "Rollers": ("rewards.py", "RollerWeights"),
    "RollerStandUp": ("rewards.py", "RollerStandUpWeights"),
    "Roulade": ("rewards.py", "RouladeWeights"),
    "BallKick": ("rewards.py", "BallKickWeights"),
    "GroundPick": ("rewards.py", "GroundPickWeights"),
}


def ours(module, cls):
    text = (SOURCE / module).read_text()
    found = re.search(rf"class {cls}[^\n]*:\n((?:\s+\w+: float[^\n]*\n)+)", text)
    if not found:
        return {}
    return {n: float(v) for n, v in re.findall(r"(\w+): float = ([-\d.e+]+)", found.group(1))}


def show(task, frozen):
    module, cls = WEIGHTS[task]
    mine = ours(module, cls)
    theirs = frozen.get(task, {})
    print(f"\n=== {task}: ours {len(mine)} terms, upstream {len(theirs)}")
    for term, entry in sorted(theirs.items(), key=lambda kv: -(kv[1]["weight"] or 0)):
        weight = entry["weight"]
        mark = "curriculum" if entry.get("curriculum") else ""
        print(f"  {term:34s} {str(entry['func'])[:24]:26s} {weight:9.4g}  {mark}")
    print("  ours:")
    for name, weight in sorted(mine.items(), key=lambda kv: -kv[1]):
        print(f"    {name:32s} {weight:9.4g}")


def main():
    frozen = json.loads(FIXTURE.read_text())
    tasks = sys.argv[1:] or sorted(WEIGHTS)
    for task in tasks:
        show(task, frozen)


if __name__ == "__main__":
    main()
