"""Read the known floor-workspace export defect without modifying dataset files.

The upstream l4i_repair_bddl.py describes the same nonexistent main_table
target. Only temporary placement-region names are repaired. Objects, fixtures,
goals, coordinates, language and the recorded simulation state stay unchanged.
"""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path


def typed_declarations(tokens):
    """Coalesce repeated PDDL type groups without dropping any declared object."""
    result = {}
    names = []
    seen = set()
    index = 0
    while index < len(tokens):
        value = tokens[index]
        index += 1
        if value == "-":
            if not names or index == len(tokens):
                raise ValueError("Malformed typed declaration")
            category = tokens[index]
            index += 1
            result.setdefault(category, []).extend(names)
            names = []
        else:
            if not isinstance(value, str) or value in seen:
                raise ValueError("Repeated/invalid object name")
            seen.add(value)
            names.append(value)
    if names:
        raise ValueError("Expected explicit object/fixture types")
    return result


def complete_export(parsed, tokens, canonical):
    result = deepcopy(parsed)
    changes = {}
    for name in ["objects", "fixtures"]:
        group = next(g for g in tokens if isinstance(g, list) and g[0] == ":" + name)
        declared = typed_declarations(group[1:])
        if declared != canonical[name]:
            raise ValueError("Scene declarations differ from its original LIBERO task")
        if list(declared) != list(canonical[name]):
            raise ValueError("Object/fixture category order differs")
        if result[name] != declared:
            changes[name] = dict(before=result[name], after=declared)
        result[name] = declared
    if (
        parsed["problem_name"] != canonical["problem_name"]
        or parsed["goal_state"] != canonical["goal_state"]
    ):
        raise ValueError("Scene task/goal differs from its original LIBERO task")
    refs = {
        arg
        for pred in parsed["goal_state"] + parsed["initial_state"]
        for arg in pred[1:]
    }
    added = []
    for name in sorted(refs):
        if name not in result["regions"] and name in canonical["regions"]:
            result["regions"][name] = deepcopy(canonical["regions"][name])
            added.append(name)
    if added:
        changes["restored_task_regions"] = added
    return result, changes


def repair_floor_workspace(parsed):
    if parsed["problem_name"] != "libero_floor_manipulation":
        return parsed, None
    wrong = {
        name: region
        for name, region in parsed["regions"].items()
        if region["target"] == "main_table"
    }
    if not wrong:
        return parsed, None
    if parsed["fixtures"] != {"floor": ["floor"]}:
        raise ValueError("Unexpected fixtures in malformed floor scene")
    mapping = {name: "floor_" + name.removeprefix("main_table_") for name in wrong}
    if any(not name.startswith("main_table_") for name in wrong):
        raise ValueError("Unexpected malformed floor region name")
    if set(mapping.values()) & parsed["regions"].keys():
        raise ValueError("Workspace repair would duplicate regions")
    if any(
        arg in mapping or arg == "main_table"
        for pred in parsed["goal_state"]
        for arg in pred
    ):
        raise ValueError("Workspace repair must not modify task goals")
    result = deepcopy(parsed)
    result["regions"] = {
        mapping.get(name, name): {
            **region,
            "target": "floor" if name in mapping else region["target"],
        }
        for name, region in result["regions"].items()
    }
    result["initial_state"] = [
        [mapping.get(arg, arg) for arg in pred] for pred in result["initial_state"]
    ]
    return result, dict(
        adapter="floor_workspace_export_v1",
        region_name_map=mapping,
        changed_fields=[
            "regions.names",
            "regions.target",
            "initial_state.region_reference",
        ],
        dataset_files_modified=False,
        goals_modified=False,
        coordinates_modified=False,
    )


@contextmanager
def snapshot_scene_adapter(scene_root, repo_root):
    """Serial construction per process, with the original parser always restored."""
    from libero.libero.envs import bddl_utils

    original = bddl_utils.robosuite_parse_problem
    audits = []

    def parse(path):
        from echo.io import sha256 as sha

        DATA, REPO = Path(scene_root).resolve(), Path(repo_root).resolve()
        from bddl.parsing import scan_tokens

        relative = Path(path).resolve().relative_to(DATA)
        if relative.parts[0] != "scenes":
            raise ValueError("Expected a scene inside the pinned data root")
        suite, task = relative.parts[1:3]
        reference = REPO / "LIBERO/libero/libero/bddl_files" / suite / (task + ".bddl")
        canonical = original(reference)
        parsed, changes = complete_export(
            original(path), scan_tokens(filename=str(path)), canonical
        )
        if changes:
            audits.append(
                dict(
                    adapter="snapshot_bddl_export_v1",
                    changes=changes,
                    reference_task=str(reference.relative_to(REPO)),
                    reference_sha256=sha(reference),
                    dataset_files_modified=False,
                    goals_modified=False,
                )
            )
        parsed, audit = repair_floor_workspace(parsed)
        if audit is not None:
            audits.append(audit)
        return parsed

    bddl_utils.robosuite_parse_problem = parse
    try:
        yield audits
    finally:
        bddl_utils.robosuite_parse_problem = original
