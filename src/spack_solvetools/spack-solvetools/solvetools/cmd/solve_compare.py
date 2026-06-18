# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import argparse
import inspect
import json
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import spack.cmd
import spack.config
import spack.hash_types as ht
import spack.llnl.util.tty as tty
import spack.llnl.util.tty.color as color
import spack.package_base
import spack.solver.asp as asp
import spack.spec
import spack.util.parallel
from spack.cmd.common.arguments import add_concretizer_args
from spack.solver.asp import (
    ErrorHandler,
    PyclingoDriver,
    Result,
    SpecBuilder,
    build_criteria_names,
)
from spack.solver.core import extract_args

# Try to import SolverPriorityConstants (available in level-opt branch)
try:
    from spack.solver.asp import SolverPriorityConstants

    HAS_PRIO_CONSTANTS = True
except ImportError:
    HAS_PRIO_CONSTANTS = False


level = "long"
section = "developer"
description = "capture and compare solve optimization criteria and DAG output"


def setup_parser(subparser: argparse.ArgumentParser):
    sp = subparser.add_subparsers(metavar="SUBCOMMAND", dest="solve_compare_command")

    # Run subcommand
    run_parser = sp.add_parser(
        "run", help="run solve for specs and save criteria and DAG output"
    )
    run_parser.add_argument(
        "specfile",
        help="text file with one spec per line to solve",
    )
    run_parser.add_argument(
        "-o",
        "--output-dir",
        help="directory to save output files",
        required=True,
    )
    run_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="number of parallel threads (default: number of CPUs)",
        default=os.cpu_count(),
    )
    run_parser.add_argument(
        "--label",
        help="label for this run (default: timestamp)",
        default=None,
    )
    add_concretizer_args(run_parser)

    # Diff subcommand
    diff_parser = sp.add_parser(
        "diff", help="compare DAGs between two runs and show which specs differ"
    )
    diff_parser.add_argument(
        "before",
        help="directory from first run",
    )
    diff_parser.add_argument(
        "after",
        help="directory from second run",
    )
    diff_parser.add_argument(
        "-o",
        "--output",
        help="output file for diff (default: stdout)",
        default=None,
    )

    # Show subcommand
    show_parser = sp.add_parser("show", help="show results from a previous run")
    show_parser.add_argument(
        "run_dir",
        help="directory from a previous run",
    )
    show_parser.add_argument(
        "--spec",
        help="show only this spec",
        default=None,
    )


def _capture_solve_with_criteria(
    inputs: Tuple[str, bool]
) -> Tuple[str, Optional[Dict], Optional[str], Optional[str]]:
    """
    Solve a single spec and capture optimization criteria and DAG.
    Returns (spec_str, criteria_data, dag_output, error_message)
    """
    spec_str, use_fresh = inputs
    try:
        specs = spack.cmd.parse_specs(spec_str)
        if len(specs) != 1:
            return (spec_str, None, None, "Expected exactly one spec")

        solver = asp.Solver()
        setup = asp.SpackSolverSetup()
        reuse = [] if use_fresh else solver.selector.reusable_specs(specs)

        result, timer, _ = solver.driver.solve(setup, specs, reuse=reuse)

        if not result.satisfiable:
            return (spec_str, None, None, "Unsatisfiable spec")

        # Extract optimization criteria
        criteria_data = {
            "spec": spec_str,
            "satisfiable": result.satisfiable,
            "nmodels": result.nmodels,
            "criteria": [],
        }

        if result.criteria:
            for criterion in result.criteria:
                criteria_data["criteria"].append(
                    {
                        "name": criterion.name,
                        "priority": criterion.priority,
                        "value": criterion.value,
                        "kind": str(criterion.kind),
                    }
                )

        # Add priority constants if available
        if hasattr(result, "prio_constants") and result.prio_constants is not None:
            pc = result.prio_constants
            criteria_data["prio_constants"] = {
                "max_depth": pc.max_depth,
                "level_opt": pc.level_opt,
                "indep_opt": pc.indep_opt,
                "low_offset": pc.low_offset,
                "concr_offset": pc.concr_offset,
                "hinge_offset": pc.hinge_offset,
                "built_offset": pc.built_offset,
                "high_offset": pc.high_offset,
                "error_offset": pc.error_offset,
                "fixed_offset": pc.fixed_offset,
            }

        # Generate DAG output as JSON
        dag_json = {}
        for spec in result.specs:
            dag_json[spec.name] = json.loads(spec.to_json(hash=ht.dag_hash))

        return (spec_str, criteria_data, dag_json, None)

    except Exception as e:
        return (spec_str, None, None, str(e))


def run(args):
    """Run solve for each spec and save optimization criteria and DAG output"""
    input_file = pathlib.Path(args.specfile)
    if not input_file.exists():
        tty.die(f"Spec file not found: {args.specfile}")

    try:
        spec_strs = [
            line.strip() for line in input_file.read_text().split("\n") if line.strip()
        ]
    except OSError as e:
        tty.die(f"Could not read the input spec file: {e}")

    if not spec_strs:
        tty.die("No specs found in input file")

    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    if args.label:
        label = args.label
    else:
        label = time.strftime("%Y%m%d-%H%M%S")

    run_dir = output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    tty.msg(f"Output directory: {run_dir}")

    # Warmup: bootstrap clingo in the main thread before parallel execution
    tty.msg("Warming up (bootstrapping clingo)...")
    try:
        warmup_specs = spack.cmd.parse_specs("zlib")
        solver = asp.Solver()
        solver.driver.solve(
            asp.SpackSolverSetup(),
            warmup_specs,
            reuse=solver.selector.reusable_specs(warmup_specs),
        )
    except Exception as e:
        tty.warn(f"Warmup failed: {e}")

    tty.msg(f"Processing {len(spec_strs)} specs with {args.jobs} threads...")

    use_fresh = hasattr(args, "concretizer_reuse") and args.concretizer_reuse is False

    results = []
    errors = []

    # Prepare inputs for parallel execution
    inputs = [(spec_str, use_fresh) for spec_str in spec_strs]

    # Use spack's parallel execution with maxtaskperchild=1 to avoid hanging
    if args.jobs > 1:
        record_iterator = spack.util.parallel.imap_unordered(
            _capture_solve_with_criteria,
            inputs,
            processes=args.jobs,
            debug=tty.is_debug(),
            maxtaskperchild=1,
        )
    else:
        record_iterator = map(_capture_solve_with_criteria, inputs)

    # Process results as they complete
    idx = 0
    for output in record_iterator:
        idx += 1
        spec_str, criteria_data, dag_output, error = output

        if error:
            errors.append({"spec": spec_str, "error": error})
            tty.warn(f"[{idx}/{len(spec_strs)}] Failed: {spec_str} - {error}")
        else:
            results.append(criteria_data)

            # Save individual spec results
            safe_name = spec_str.replace("/", "_").replace(" ", "_").replace("@", "-")
            spec_dir = run_dir / safe_name
            spec_dir.mkdir(exist_ok=True)

            # Save criteria as JSON
            criteria_file = spec_dir / "criteria.json"
            with open(criteria_file, "w") as f:
                json.dump(criteria_data, f, indent=2)

            # Save DAG output as JSON
            if dag_output:
                dag_file = spec_dir / "dag.json"
                with open(dag_file, "w") as f:
                    json.dump(dag_output, f, indent=2)

            tty.msg(f"[{idx}/{len(spec_strs)}] {spec_str}")

    # Save summary
    summary = {
        "label": label,
        "num_specs": len(spec_strs),
        "num_successful": len(results),
        "num_failed": len(errors),
        "specs": spec_strs,
        "use_fresh": use_fresh,
    }

    summary_file = run_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Save errors if any
    if errors:
        errors_file = run_dir / "errors.json"
        with open(errors_file, "w") as f:
            json.dump(errors, f, indent=2)

    tty.msg(f"\nSuccessful: {len(results)}, Failed: {len(errors)}")
    tty.msg(f"Results saved to: {run_dir}")


def _display_criteria(criteria_data):
    """Display optimization criteria in the same format as spack solve"""
    criteria_list = criteria_data.get("criteria", [])
    if not criteria_list:
        return

    # Reconstruct criteria objects from JSON
    class Criterion:
        def __init__(self, data):
            self.name = data["name"]
            self.priority = data["priority"]
            self.value = data["value"]
            # Parse kind back to enum
            kind_str = data["kind"]
            if "BUILD" in kind_str or kind_str == "0":
                self.kind = asp.OptimizationKind.BUILD
            elif "CONCRETE" in kind_str or kind_str == "1":
                self.kind = asp.OptimizationKind.CONCRETE
            else:
                self.kind = asp.OptimizationKind.BUILD  # default

    criteria = [Criterion(c) for c in criteria_list]
    maxlen = max(len(c.name) for c in criteria) if criteria else 0

    # Check if we have prio_constants (level-opt branch)
    prio_data = criteria_data.get("prio_constants")
    if prio_data:
        # Reconstruct priority constants
        class PrioConstants:
            def __init__(self, data):
                self.max_depth = data["max_depth"]
                self.level_opt = data["level_opt"]
                self.indep_opt = data["indep_opt"]
                self.low_offset = data["low_offset"]
                self.concr_offset = data["concr_offset"]
                self.hinge_offset = data["hinge_offset"]
                self.built_offset = data["built_offset"]
                self.high_offset = data["high_offset"]
                self.error_offset = data["error_offset"]
                self.fixed_offset = data["fixed_offset"]

        pc = PrioConstants(prio_data)

        def extract_level(priority, kind):
            level_upper_bound_built = pc.built_offset + pc.max_depth * pc.level_opt
            level_upper_bound_concr = pc.concr_offset + pc.max_depth * pc.level_opt

            if kind == asp.OptimizationKind.BUILD:
                if pc.built_offset <= priority < level_upper_bound_built:
                    offset_from_level_start = priority - pc.built_offset
                    return offset_from_level_start // pc.level_opt
            elif kind == asp.OptimizationKind.CONCRETE:
                if pc.concr_offset <= priority < level_upper_bound_concr:
                    offset_from_level_start = priority - pc.concr_offset
                    return offset_from_level_start // pc.level_opt
            return None

        sections = [
            ("High Priority", lambda c: c.priority >= pc.high_offset),
            ("Build Priority", lambda c: pc.built_offset <= c.priority < pc.high_offset),
            ("Hinge Priority", lambda c: pc.hinge_offset <= c.priority < pc.built_offset),
            ("Concrete/Reuse Priority", lambda c: pc.concr_offset <= c.priority < pc.hinge_offset),
            ("Low Priority", lambda c: pc.low_offset <= c.priority < pc.concr_offset),
        ]

        for section_name, section_filter in sections:
            section_criteria = [c for c in criteria if section_filter(c)]
            if not section_criteria:
                continue

            color.cprint(f"\n@*{{{section_name}:}}")

            # Group criteria by (name, kind)
            criteria_groups = {}
            for criterion in section_criteria:
                key = (criterion.name, criterion.kind)
                if key not in criteria_groups:
                    criteria_groups[key] = []
                criteria_groups[key].append(criterion)

            sorted_groups = sorted(
                criteria_groups.items(),
                key=lambda x: max(c.priority for c in x[1]),
                reverse=True
            )

            rows = []
            for key, group in sorted_groups:
                name, kind = key
                group_sorted = sorted(group, key=lambda c: c.priority, reverse=True)
                highest_priority = group_sorted[0].priority

                prev_value = 0
                values_by_level = {}

                for criterion in group_sorted:
                    if len(group) > 1:  # Level-expanded
                        internal_level = extract_level(criterion.priority, criterion.kind)
                        if internal_level is not None:
                            display_value = criterion.value - prev_value
                            prev_value = criterion.value
                            display_level = pc.max_depth - 1 - internal_level
                            values_by_level[display_level] = display_value
                    else:  # Fixed criterion
                        values_by_level[0] = criterion.value

                rows.append((highest_priority, values_by_level, name, kind))

            # Print header
            header_cols = ["Priority"] + [f"L{i}" for i in range(pc.max_depth)] + ["Criterion"]
            col_widths = [8] + [6] * pc.max_depth + [maxlen]

            header_parts = [f"{col:>{w}}" for col, w in zip(header_cols[:-1], col_widths[:-1])]
            header_parts.append(f"{header_cols[-1]:<{col_widths[-1]}}")
            header = "  " + "  ".join(header_parts)
            color.cprint("@*{" + header + "}")

            # Print rows
            for priority, values_by_level, name, kind in rows:
                value_cols = []
                for level in range(pc.max_depth):
                    if level in values_by_level:
                        value_cols.append((values_by_level[level], kind))
                    else:
                        value_cols.append((None, kind))

                row_parts = [f"  @K{{{priority:>8}}}"]

                for value, k in value_cols:
                    if value is None:
                        row_parts.append(f"  @K{{{'-':>6}}}")
                    elif value > 0:
                        if k == asp.OptimizationKind.CONCRETE:
                            row_parts.append(f"  @b{{{value:>6}}}")
                        elif k == asp.OptimizationKind.BUILD:
                            row_parts.append(f"  @g{{{value:>6}}}")
                        else:
                            row_parts.append(f"  @y{{{value:>6}}}")
                    else:
                        row_parts.append(f"  @K{{{value:>6}}}")

                lc = "@K"
                if any(v > 0 for v in values_by_level.values() if v is not None):
                    if kind == asp.OptimizationKind.CONCRETE:
                        lc = "@b"
                    elif kind == asp.OptimizationKind.BUILD:
                        lc = "@g"
                    else:
                        lc = "@y"

                row_parts.append(f"  {lc}{{{name:<{maxlen}}}}")
                color.cprint("".join(row_parts))

        print()
        print()
        color.cprint("  @*{Legend:}")
        color.cprint("    @g{Specs to be built}")
        color.cprint("    @b{Reused specs}")
        color.cprint("    @y{Other criteria}")
        print()

    else:
        # Develop branch: simple format
        color.cprint("@*{  Priority  Value  Criterion}")
        for c in criteria:
            value = f"@K{{{c.value:>5}}}"
            grey_out = True
            if c.value > 0:
                value = f"@*{{{c.value:>5}}}"
                grey_out = False

            if grey_out:
                lc = "@K"
            elif c.kind == asp.OptimizationKind.CONCRETE:
                lc = "@b"
            elif c.kind == asp.OptimizationKind.BUILD:
                lc = "@g"
            else:
                lc = "@y"

            color.cprint(f"  @K{{{c.priority:8}}}  {value}  {lc}{{{c.name:<{maxlen}}}}")
        print()
        print()
        color.cprint("  @*{Legend:}")
        color.cprint("    @g{Specs to be built}")
        color.cprint("    @b{Reused specs}")
        color.cprint("    @y{Other criteria}")
        print()


def show(args):
    """Show results from a previous run"""
    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.exists():
        tty.die(f"Run directory not found: {args.run_dir}")

    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        tty.die(f"Summary file not found: {summary_file}")

    with open(summary_file) as f:
        summary = json.load(f)

    color.cprint(f"\n@*{{Run: {summary['label']}}}")
    color.cprint(
        f"@*{{Results:}} {summary['num_successful']}/{summary['num_specs']} successful"
    )
    print()

    if args.spec:
        # Show specific spec
        safe_name = args.spec.replace("/", "_").replace(" ", "_").replace("@", "-")
        spec_dir = run_dir / safe_name
        criteria_file = spec_dir / "criteria.json"

        if not criteria_file.exists():
            tty.die(f"Spec not found in run: {args.spec}")

        with open(criteria_file) as f:
            criteria_data = json.load(f)

        if not criteria_data:
            tty.die("Criteria file is empty or invalid")

        color.cprint(f"@*{{Spec: {args.spec}}}")
        if "nmodels" in criteria_data:
            color.cprint(f"@*{{Models considered:}} {criteria_data['nmodels']}")
        print()

        # Show optimization criteria using the same format as spack solve
        _display_criteria(criteria_data)

        # Show DAG as tree
        color.cprint("@*{DAG:}")
        dag_file = spec_dir / "dag.json"
        if dag_file.exists():
            with open(dag_file) as f:
                dag_data = json.load(f)

            # Reconstruct specs from JSON and display as tree
            specs = []
            for spec_name, spec_dict in dag_data.items():
                spec = spack.spec.Spec.from_dict(spec_dict)
                specs.append(spec)

            # Display as tree with color and non-defaults highlighted
            tree_output = spack.spec.tree(
                specs,
                color=True,
                format=spack.spec.DISPLAY_FORMAT,
                hashlen=7,
                hashes=False,
                status_fn=None,
                show_types=False,
                highlight_version_fn=spack.package_base.non_preferred_version,
                highlight_variant_fn=spack.package_base.non_default_variant,
            )
            print(tree_output)
        else:
            print("(DAG file not found)")

    else:
        # Show all specs
        for spec_str in summary["specs"]:
            safe_name = spec_str.replace("/", "_").replace(" ", "_").replace("@", "-")
            spec_dir = run_dir / safe_name
            criteria_file = spec_dir / "criteria.json"

            if criteria_file.exists():
                print(f"  \033[32mOK\033[0m      {spec_str}")
            else:
                print(f"  \033[31mFAILED\033[0m  {spec_str}")


def diff(args):
    """Compare DAGs between two runs and show which specs differ"""
    before_dir = pathlib.Path(args.before)
    after_dir = pathlib.Path(args.after)

    if not before_dir.exists():
        tty.die(f"Before directory not found: {args.before}")
    if not after_dir.exists():
        tty.die(f"After directory not found: {args.after}")

    # Load summaries
    before_summary_file = before_dir / "summary.json"
    after_summary_file = after_dir / "summary.json"

    if not before_summary_file.exists():
        tty.die(f"Summary file not found: {before_summary_file}")
    if not after_summary_file.exists():
        tty.die(f"Summary file not found: {after_summary_file}")

    with open(before_summary_file) as f:
        before_summary = json.load(f)
    with open(after_summary_file) as f:
        after_summary = json.load(f)

    # Setup output
    if args.output:
        output_file = open(args.output, "w")
        out = output_file
    else:
        out = sys.stdout

    def write(msg):
        out.write(msg + "\n")

    write(f"# DAG Comparison: {before_summary['label']} vs {after_summary['label']}")
    write("")

    # Find common specs
    before_specs = set(before_summary["specs"])
    after_specs = set(after_summary["specs"])
    common_specs = before_specs & after_specs

    if not common_specs:
        write("No common specs found between the two runs")
        return

    write(f"Comparing {len(common_specs)} common specs")
    write("")

    # Compare DAGs
    identical = []
    different = []
    missing_data = []

    for spec_str in sorted(common_specs):
        safe_name = spec_str.replace("/", "_").replace(" ", "_").replace("@", "-")

        before_dag_file = before_dir / safe_name / "dag.json"
        after_dag_file = after_dir / safe_name / "dag.json"

        if not before_dag_file.exists() or not after_dag_file.exists():
            missing_data.append(spec_str)
            continue

        try:
            with open(before_dag_file) as f:
                before_dag = json.load(f)
            with open(after_dag_file) as f:
                after_dag = json.load(f)
        except (json.JSONDecodeError, ValueError):
            missing_data.append(spec_str)
            continue

        # Compare the JSON structures
        if before_dag == after_dag:
            identical.append(spec_str)
        else:
            different.append(spec_str)

    # Report results
    write("## Summary")
    write("")
    write(f"Total specs compared: {len(common_specs)}")
    write(f"Identical DAGs:       {len(identical)}")
    write(f"Different DAGs:       {len(different)}")
    write(f"Missing data:         {len(missing_data)}")
    write("")

    if different:
        write("## Specs with Different DAGs")
        write("")
        for spec_str in different:
            write(f"  - {spec_str}")
        write("")

    if missing_data:
        write("## Specs with Missing Data")
        write("")
        for spec_str in missing_data:
            write(f"  - {spec_str}")
        write("")

    if args.output:
        output_file.close()
        tty.msg(f"Diff saved to: {args.output}")
    else:
        # Colorized output for terminal
        print()
        color.cprint(f"@*{{# DAG Comparison: {before_summary['label']} vs {after_summary['label']}}}")
        print()
        color.cprint(f"@*{{Comparing {len(common_specs)} common specs}}")
        print()

        # Summary with colors
        color.cprint("@*{Summary:}")
        print(f"  Total specs:     {len(common_specs)}")
        if identical:
            color.cprint(f"  @g{{Identical DAGs:  {len(identical)}}}")
        if different:
            color.cprint(f"  @y{{Different DAGs:  {len(different)}}}")
        if missing_data:
            color.cprint(f"  @r{{Missing data:    {len(missing_data)}}}")
        print()

        if different:
            color.cprint("@*{Specs with Different DAGs:}")
            for spec_str in different:
                color.cprint(f"  @y{{-}} {spec_str}")
            print()

        if missing_data:
            color.cprint("@*{Specs with Missing Data:}")
            for spec_str in missing_data:
                color.cprint(f"  @r{{-}} {spec_str}")
            print()


def solve_compare(parser: argparse.ArgumentParser, args):
    if not args.solve_compare_command:
        parser.print_help()
        return

    if args.solve_compare_command == "run":
        run(args)
    elif args.solve_compare_command == "show":
        show(args)
    elif args.solve_compare_command == "diff":
        diff(args)
