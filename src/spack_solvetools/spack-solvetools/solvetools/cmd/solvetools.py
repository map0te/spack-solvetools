# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import argparse
import pprint
import re
import sys
import time
import warnings
from typing import Dict, List, Optional

import spack.cmd
import spack.config
import spack.llnl.util.lang
import spack.solver.asp as asp
import spack.spec
import spack.binary_distribution
import spack.cmd.spec
import spack.environment
import spack.hash_types as ht
import spack.llnl.util.tty as tty
import spack.llnl.util.tty.color as color
import spack.package_base
from spack.cmd.common.arguments import add_concretizer_args
from spack.solver.asp import ErrorHandler, PyclingoDriver, Result, SpecBuilder, UnsatisfiableSpecError, build_criteria_names
from spack.solver.core import extract_args


level = "long"
section = "developer"
description = "solve visualization and profiling tools"


def setup_parser(subparser: argparse.ArgumentParser):
    sp = subparser.add_subparsers(metavar='SUBCOMMAND', dest='solvetools_command')

    # List-models subcommand
    list_models_parser = sp.add_parser(
        'list-models',
        help='list intermediate models during spec solving'
    )
    list_models_parser.add_argument(
        "spec",
        help="spec to solve and list intermediate models",
    )
    list_models_parser.add_argument(
        "-o",
        "--output",
        help="write output to file instead of stdout",
        default=None,
    )
    add_concretizer_args(list_models_parser)

    # Profile subcommand
    profile_parser = sp.add_parser(
        'profile',
        help='profile the solve phase and print statistics'
    )
    profile_parser.add_argument(
        "--show",
        action="store",
        default="opt,solutions",
        help="select outputs\n\ncomma-separated list of:\n"
        "  asp          asp program text\n"
        "  opt          optimization criteria for best model\n"
        "  output       raw clingo output\n"
        "  solutions    models found by asp program\n"
        "  all          all of the above",
    )
    profile_parser.add_argument(
        "--timers",
        action="store_true",
        default=False,
        help="print out timers for different solve phases",
    )
    profile_parser.add_argument(
        "--stats",
        action="store_true",
        default=False,
        help="print out statistics from clingo"
    )
    spack.cmd.spec.setup_parser(profile_parser)


# List-models implementation
models = []

def capturing_run_clingo(self, specs_arg, setup_arg, problem_str, control_file_paths, timer):
    with timer.measure("load"):
        self.control.add("base", [], problem_str)
        for path in control_file_paths:
            self.control.load(path)

    with timer.measure("ground"):
        self.control.ground([("base", [])])

    def on_model(model):
        models.append((model.cost, model.symbols(shown=True, terms=True), model.number))

    timer.start("solve")
    time_limit = spack.config.CONFIG.get("concretizer:timeout", 0)
    timeout_end = time.monotonic() + time_limit if time_limit > 0 else float("inf")
    error_on_timeout = spack.config.CONFIG.get("concretizer:error_on_timeout", True)

    with self.control.solve(on_model=on_model, async_=True) as handle:
        finished = False
        while not finished and time.monotonic() < timeout_end:
            finished = handle.wait(1.0)

        if not finished:
            specs_str = ", ".join(spack.llnl.util.lang.elide_list([str(s) for s in specs_arg], 4))
            header = f"Spack is taking more than {time_limit} seconds to solve for {specs_str}"
            if error_on_timeout:
                raise UnsatisfiableSpecError(f"{header}, stopping concretization")
            warnings.warn(f"{header}, using the best configuration found so far")
            handle.cancel()

        solve_result = handle.get()
    timer.stop("solve")

    result = Result(specs_arg)
    result.satisfiable = solve_result.satisfiable
    if not result.satisfiable:
        return result

    timer.start("construct_specs")
    builder = SpecBuilder(specs_arg, hash_lookup=setup_arg.reusable_and_possible)
    min_cost, best_model, _ = min(models)

    error_handler = ErrorHandler(best_model, specs_arg)
    error_handler.raise_if_errors()

    spec_attrs = [(name, tuple(rest)) for name, *rest in extract_args(best_model, "attr")]
    spec_dict = builder.build_specs(spec_attrs)

    result.answers.append((list(min_cost), 0, spec_dict))
    criteria_args = extract_args(best_model, "opt_criterion")
    result.criteria = build_criteria_names(min_cost, criteria_args)
    result.nmodels = len(models)
    result.possible_dependencies = setup_arg.pkgs
    timer.stop("construct_specs")
    timer.stop()

    return result


def process_model_to_specs(symbols, original_specs, reusable_and_possible):
    builder = SpecBuilder(original_specs, hash_lookup=reusable_and_possible)
    spec_attrs = [(name, tuple(rest)) for name, *rest in extract_args(symbols, "attr")]
    spec_dict = builder.build_specs(spec_attrs)
    root_names = {s.name for s in original_specs}
    return [spec for key, spec in spec_dict.items() if key.id == "0" and spec.name in root_names]


def format_model_output(
    model_num: int,
    cost: tuple,
    specs: Optional[List[spack.spec.Spec]],
    use_color: bool = True,
) -> str:
    header = "=" * 77
    separator = "-" * 77
    cost_str = "[" + ", ".join(str(c) for c in cost) + "]"
    title = f"Model {model_num} - Cost: {cost_str}"

    output = f"\n{header}\n{title}\n{header}\n"

    if specs is None:
        output += "[Invalid or incomplete model]\n"
    else:
        tree_output = spack.spec.tree(
            specs,
            color=use_color,
            format=spack.spec.DISPLAY_FORMAT,
            hashlen=7,
            hashes=False,
            status_fn=None,
            show_types=False,
        )
        output += tree_output

    output += f"\n{separator}\n"
    return output


def list_models(args):
    specs = spack.cmd.parse_specs(args.spec)
    if len(specs) != 1:
        tty.die("solvetools list-models requires exactly one spec")

    models.clear()
    solver = asp.Solver()
    setup = asp.SpackSolverSetup()
    use_fresh = hasattr(args, 'concretizer_reuse') and args.concretizer_reuse is False
    reuse = [] if use_fresh else solver.selector.reusable_specs(specs)

    original_run_clingo = PyclingoDriver._run_clingo
    try:
        PyclingoDriver._run_clingo = capturing_run_clingo
        result, timer, _ = solver.driver.solve(setup, specs, reuse=reuse)
    except Exception as e:
        tty.die(f"Solve failed: {e}")
    finally:
        PyclingoDriver._run_clingo = original_run_clingo

    if not result.satisfiable:
        tty.warn("Solve was unsatisfiable.")

    use_color = sys.stdout.isatty() and args.output is None

    output_lines = []
    for i, (cost, symbols, num) in enumerate(models):
        specs_list = None
        if result.satisfiable:
            specs_list = process_model_to_specs(
                symbols, specs, setup.reusable_and_possible if hasattr(setup, "reusable_and_possible") else {}
            )

        output_lines.append(
            format_model_output(i + 1, cost, specs_list, use_color)
        )

    output_text = "".join(output_lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
    else:
        print(output_text, end="")


# Profile implementation
def _process_result(result, show, required_format, kwargs):
    opt, _, _ = min(result.answers)
    if ("opt" in show) and (not required_format):
        tty.msg("Best of %d considered solutions." % result.nmodels)

        print()
        maxlen = max(len(s.name) for s in result.criteria)
        color.cprint("@*{  Priority  Value  Criterion}")

        for i, criterion in enumerate(result.criteria, 1):
            value = f"@K{{{criterion.value:>5}}}"
            grey_out = True
            if criterion.value > 0:
                value = f"@*{{{criterion.value:>5}}}"
                grey_out = False

            if grey_out:
                lc = "@K"
            elif criterion.kind == asp.OptimizationKind.CONCRETE:
                lc = "@b"
            elif criterion.kind == asp.OptimizationKind.BUILD:
                lc = "@g"
            else:
                lc = "@y"

            color.cprint(f"  @K{{{i:8}}}  {value}  {lc}{{{criterion.name:<{maxlen}}}}")
        print()
        print()
        color.cprint("  @*{Legend:}")
        color.cprint("    @g{Specs to be built}")
        color.cprint("    @b{Reused specs}")
        color.cprint("    @y{Other criteria}")
        print()

    if "solutions" in show:
        if required_format:
            for spec in result.specs:
                if required_format == "yaml":
                    sys.stdout.write(spec.to_yaml(hash=ht.dag_hash))
                elif required_format == "json":
                    sys.stdout.write(spec.to_json(hash=ht.dag_hash))
        else:
            tree_str = spack.spec.tree(result.specs, color=sys.stdout.isatty(), **kwargs)
            sys.stdout.write(tree_str)
        print()

    if result.unsolved_specs and "solutions" in show:
        tty.msg(asp.Result.format_unsolved(result.unsolved_specs))


def profile(args):
    fmt = spack.spec.DISPLAY_FORMAT
    if args.namespaces:
        fmt = "{namespace}." + fmt

    show_status = args.install_status
    if show_status:
        spack.binary_distribution.load_buildcache_index()
        status_fn = spack.cmd.buildcache_status_fn(spack.binary_distribution.BINARY_INDEX)
    else:
        status_fn = None

    kwargs = {
        "cover": args.cover,
        "format": fmt,
        "hashlen": None if args.very_long else 7,
        "show_types": args.types,
        "status_fn": status_fn,
        "hashes": args.long or args.very_long,
        "highlight_version_fn": (
            spack.package_base.non_preferred_version if args.non_defaults else None
        ),
        "highlight_variant_fn": (
            spack.package_base.non_default_variant if args.non_defaults else None
        ),
    }

    show = re.split(r"\s*,\s*", args.show)
    show_options = ("asp", "opt", "output", "solutions")
    if "all" in show:
        show = show_options
    for d in show:
        if d not in show_options:
            raise ValueError(
                "Invalid option for '--show': '%s'\nchoose from: (%s)"
                % (d, ", ".join(show_options + ("all",)))
            )

    required_format = args.format

    env = spack.environment.active_environment()
    if args.specs:
        specs = spack.cmd.parse_specs(args.specs)
    elif env:
        specs = list(env.user_specs)
    else:
        tty.die("requires at least one spec or an active environment")

    # Import the bundled profiler module
    from ..profiler import ProfilePropagator

    solver = asp.Solver()
    output_config = asp.OutputConfiguration(
        out=sys.stdout if "asp" in show else None,
        timers=args.timers,
        stats=args.stats,
        setup_only=set(show) == {"asp"}
    )

    unify = spack.config.get("concretizer:unify")
    allow_deprecated = spack.config.get("config:deprecated", False)

    # Patch the driver to enable profiling
    original_solve = solver.driver.solve

    def profile_solve(*args, **kwargs):
        control = kwargs.get('control') or asp.default_clingo_control()
        propagator = ProfilePropagator()
        control.register_propagator(propagator)
        kwargs['control'] = control
        result = original_solve(*args, **kwargs)

        if output_config.timers:
            tty.msg("Timers:")
            if len(args) > 3:
                args[3].write_tty()
            print()

        if output_config.stats:
            tty.msg("Statistics:")
            # Stats would be in the result
            print()

        tty.msg("Profile")
        propagator.print_profile(40)

        return result

    solver.driver.solve = profile_solve

    if unify == "when_possible":
        for idx, result in enumerate(
            solver.solve_in_rounds(
                specs,
                out=output_config.out,
                timers=args.timers,
                stats=args.stats,
                allow_deprecated=allow_deprecated,
            )
        ):
            if "solutions" in show:
                tty.msg("ROUND {0}".format(idx))
                tty.msg("")
            else:
                print("% END ROUND {0}\n".format(idx))
            if not output_config.setup_only:
                _process_result(result, show, required_format, kwargs)
    elif unify:
        result = solver.solve(
            specs,
            out=output_config.out,
            timers=args.timers,
            stats=args.stats,
            setup_only=output_config.setup_only,
            allow_deprecated=allow_deprecated,
        )
        if not output_config.setup_only:
            _process_result(result, show, required_format, kwargs)
    else:
        for spec in specs:
            tty.msg("SOLVING SPEC:", spec)
            result = solver.solve(
                [spec],
                out=output_config.out,
                timers=args.timers,
                stats=args.stats,
                setup_only=output_config.setup_only,
                allow_deprecated=allow_deprecated,
            )
            if not output_config.setup_only:
                _process_result(result, show, required_format, kwargs)


def solvetools(parser: argparse.ArgumentParser, args):
    if not args.solvetools_command:
        parser.print_help()
        return

    if args.solvetools_command == "list-models":
        list_models(args)
    elif args.solvetools_command == "profile":
        profile(args)
