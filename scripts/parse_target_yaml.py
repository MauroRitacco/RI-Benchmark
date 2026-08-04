#!/usr/bin/env python3
"""Parse <target>.yaml for a given target and method.

Usage examples:
  python parse_target_yaml.py --target disk_1111 --method casa
  python parse_target_yaml.py --target disk_1111 --method casa --key niter
  python parse_target_yaml.py --target disk_1111 --key nval
"""
import argparse
import json
import os
import sys

try:
    import yaml
except ImportError:
    # Fallback for environments where yaml is not available
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Parse <target>.yaml for a given target and method.")
    parser.add_argument("--target", required=True, help="Target name (e.g. disk_1111)")
    parser.add_argument(
        "--method",
        default=None,
        choices=["r2d2", "casa", "wsclean", "deepinv"],
        help="Method section to extract from the YAML",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Extract a specific key from the method section or top level.",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    yaml_path = None
    for data_type in ["simulated", "archival"]:
        candidate = os.path.join(base_dir, "data", data_type, args.target, f"{args.target}.yaml")
        if os.path.exists(candidate):
            yaml_path = candidate
            break

    if yaml_path is None:
        print(f"{args.target}.yaml not found for target {args.target}", file=sys.stderr)
        sys.exit(1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Top-level key lookup (no method required)
    if args.key and not args.method:
        value = data.get(args.key)
        if value is None:
            sys.exit(3)
        if isinstance(value, bool):
            print("true" if value else "false")
        elif isinstance(value, (list, dict)):
            print(json.dumps(value))
        else:
            print(value)
        return

    methods = data.get("methods", {})
    method_config = methods.get(args.method)
    if method_config is None:
        print(f"Method {args.method} not found in {yaml_path}", file=sys.stderr)
        sys.exit(2)

    if args.key:
        value = method_config.get(args.key)
        if value is None:
            sys.exit(3)
        if isinstance(value, bool):
            print("true" if value else "false")
        elif isinstance(value, (list, dict)):
            print(json.dumps(value))
        else:
            print(value)
    else:
        print(json.dumps(method_config))


if __name__ == "__main__":
    main()
