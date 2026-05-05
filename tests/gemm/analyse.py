#!/usr/bin/env python3
"""
NSight Systems (.nsys-rep) file analyzer for GEMM performance profiling.
Processes nsys-rep files, extracts CUDA kernel events, and calculates TFLOPS for nvjet_qqtst operations.
"""

import os
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


def parse_filename_params(filename: str) -> Dict[str, int]:
    """
    Parse GEMM parameters from filename.
    Expected format: mbs{mbs}.seq{seq}.hds{hds}.out{out}*
    """
    params = {
        'mbs': 1,
        'seq_len': 4096,
        'hidden_size': 7168,
        'output_size': 1536
    }

    mbs_match = re.search(r'mbs(\d+)', filename)
    if mbs_match:
        params['mbs'] = int(mbs_match.group(1))

    seq_match = re.search(r'seq(\d+)', filename)
    if seq_match:
        params['seq_len'] = int(seq_match.group(1))

    hds_match = re.search(r'hds(\d+)', filename)
    if hds_match:
        params['hidden_size'] = int(hds_match.group(1))

    out_match = re.search(r'out(\d+)', filename)
    if out_match:
        params['output_size'] = int(out_match.group(1))

    return params


def convert_nsys_to_json(nsys_path: str) -> Optional[str]:
    """
    Convert nsys-rep file to JSON format using nsys CLI.
    Returns path to JSON file or None if conversion fails.
    """
    json_path = str(Path(nsys_path).with_suffix('.json'))

    if os.path.exists(json_path):
        return json_path

    try:
        cmd = ['nsys', 'export', '-t', 'json', nsys_path, '-o', json_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if os.path.exists(json_path):
            return json_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Failed to convert {nsys_path}: {e}")
        return None

    return None


def build_string_mapping(data: str) -> Dict[str, str]:
    """
    Parse JSON string data and build id to value mapping from String type entries.
    """
    string_mapping = {}

    for match in re.finditer(r'"type":"String","id":"([^"]+)","value":"([^"]*)"', data):
        entry_id = match.group(1)
        value = match.group(2)
        string_mapping[entry_id] = value

    return string_mapping


def extract_cuda_events(data: str) -> List[Dict[str, Any]]:
    """
    Extract Type 79 cuda events from JSON data.
    Returns list of parsed event dictionaries.
    """
    events = []

    pattern = r'\{"Type":79,"CudaEvent":\{[^}]*?"startNs":"(\d+)"[^}]*?"endNs":"(\d+)"[^}]*?"kernel":\{([^}]+?)\}\}\}'

    for match in re.finditer(pattern, data, re.DOTALL):
        start_ns = int(match.group(1))
        end_ns = int(match.group(2))
        duration_ns = end_ns - start_ns

        kernel_part = match.group(3)

        fields = {}
        for field_name in ['demangledName', 'mangledName', 'shortName', 'eventCategory']:
            field_match = re.search(rf'{field_name}":"([^"]+)"', kernel_part)
            if field_match:
                fields[field_name] = field_match.group(1)

        event = {
            'startNs': start_ns,
            'endNs': end_ns,
            'durationNs': duration_ns,
            'durationUs': duration_ns / 1000.0,
            'durationMs': duration_ns / 1000000.0,
            'kernel': fields
        }
        events.append(event)

    return events


def resolve_string_ids(events: List[Dict[str, Any]], string_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Resolve string IDs in events using the string mapping.
    """
    for event in events:
        for field in ['demangledName', 'mangledName', 'shortName', 'eventCategory']:
            if field in event['kernel']:
                id_value = event['kernel'][field]
                if id_value in string_mapping:
                    event['kernel'][field] = string_mapping[id_value]

    return events


def calculate_gemm_tflops(params: Dict[str, int], duration_ms: float) -> float:
    """
    Calculate TFLOPS for GEMM operation.
    Formula: TFLOPS = (2 * M * N * K) / (time_seconds * 1e12)
    where M = seq_len * mbs, N = output_size, K = hidden_size
    """
    M = params['seq_len'] * params['mbs']
    N = params['output_size']
    K = params['hidden_size']

    flops = 2 * M * N * K
    time_seconds = duration_ms / 1000.0

    tflops = flops / (time_seconds * 1e12)
    return tflops


def analyze_json_data(json_data: str, filename: str) -> Dict[str, Any]:
    """
    Analyze JSON data directly.
    Returns analysis results including parameters, events, and GEMM statistics.
    """
    params = parse_filename_params(filename)

    string_mapping = build_string_mapping(json_data)
    events = extract_cuda_events(json_data)
    events = resolve_string_ids(events, string_mapping)

    gemm_events = []
    for event in events:
        short_name = event['kernel'].get('shortName', '')
        if 'nvjet_qqtst' in short_name:
            tflops = calculate_gemm_tflops(params, event['durationMs'])
            gemm_event = event.copy()
            gemm_event['tflops'] = tflops
            gemm_events.append(gemm_event)

    total_gemm_time_ms = sum(e['durationMs'] for e in gemm_events)
    avg_gemm_tflops = sum(e['tflops'] for e in gemm_events) / len(gemm_events) if gemm_events else 0
    max_gemm_tflops = max(e['tflops'] for e in gemm_events) if gemm_events else 0

    return {
        'filename': filename,
        'parameters': params,
        'total_events': len(events),
        'gemm_events': {
            'count': len(gemm_events),
            'total_time_ms': total_gemm_time_ms,
            'avg_tflops': avg_gemm_tflops,
            'max_tflops': max_gemm_tflops,
            'events': gemm_events[:10]  # First 10 for summary
        },
        'string_mapping_size': len(string_mapping)
    }


def analyze_nsys_file(nsys_path: str) -> Dict[str, Any]:
    """
    Analyze a single nsys-rep file.
    Returns analysis results including parameters, events, and GEMM statistics.
    """
    filename = os.path.basename(nsys_path)
    params = parse_filename_params(filename)

    json_path = convert_nsys_to_json(nsys_path)
    if json_path is None:
        return {
            'filename': filename,
            'error': 'Failed to convert to JSON'
        }

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = f.read()
    except Exception as e:
        return {
            'filename': filename,
            'error': f'Failed to read JSON: {e}'
        }

    return analyze_json_data(json_data, filename)


def analyze_json_file(json_path: str) -> Dict[str, Any]:
    """
    Analyze a single JSON file directly.
    Returns analysis results including parameters, events, and GEMM statistics.
    """
    filename = os.path.basename(json_path)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = f.read()
    except Exception as e:
        return {
            'filename': filename,
            'error': f'Failed to read JSON: {e}'
        }

    return analyze_json_data(json_data, filename)


def analyze_directory(directory: str = None) -> List[Dict[str, Any]]:
    """
    Analyze all nsys-rep files in a directory.
    """
    if directory is None:
        script_dir = Path(__file__).parent
        directory = str(script_dir)
    else:
        directory = Path(directory)

    results = []

    nsys_files = list(directory.glob('*.nsys-rep'))
    if not nsys_files:
        print(f"No .nsys-rep files found in {directory}")
        return results

    for nsys_path in sorted(nsys_files):
        print(f"Analyzing: {nsys_path.name}")
        result = analyze_nsys_file(str(nsys_path))
        results.append(result)

    return results


def analyze_json_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Analyze all JSON files in a directory.
    """
    directory = Path(directory)

    results = []

    json_files = list(directory.glob('*.json'))
    if not json_files:
        print(f"No .json files found in {directory}")
        return results

    for json_path in sorted(json_files):
        print(f"Analyzing: {json_path.name}")
        result = analyze_json_file(str(json_path))
        results.append(result)

    return results


def print_summary(results: List[Dict[str, Any]]):
    """
    Print a summary of analysis results.
    """
    print("\n" + "=" * 80)
    print("NSYS-REP Analysis Summary")
    print("=" * 80)

    for result in results:
        if 'error' in result:
            print(f"\n{result['filename']}: ERROR - {result['error']}")
            continue

        params = result['parameters']
        gemm = result['gemm_events']

        print(f"\n{result['filename']}")
        print(f"  Parameters: MBS={params['mbs']}, SEQ={params['seq_len']}, "
              f"HDS={params['hidden_size']}, OUT={params['output_size']}")
        print(f"  Total events: {result['total_events']}")
        print(f"  GEMM events: {gemm['count']}")
        print(f"  Total GEMM time: {gemm['total_time_ms']:.4f} ms")
        print(f"  Avg GEMM TFLOPS: {gemm['avg_tflops']:.2f}")
        print(f"  Max GEMM TFLOPS: {gemm['max_tflops']:.2f}")

    print("\n" + "=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze nsys-rep files for GEMM performance profiling'
    )
    parser.add_argument(
        '--path',
        help='Path to nsys-rep file or directory containing nsys-rep files'
    )
    parser.add_argument(
        '--json-path',
        help='Path to JSON file or directory containing JSON files (already converted)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Path for detailed results in JSON/EXCEL format'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary to stdout'
    )

    args = parser.parse_args()

    if args.path and args.json_path:
        print("Error: Cannot specify both --path and --json-path")
        return 1

    if not args.path and not args.json_path:
        print("Error: Must specify either --path or --json-path")
        return 1

    results = []

    if args.path:
        path = Path(args.path)
        if path.is_file() and path.suffix == '.nsys-rep':
            results = [analyze_nsys_file(str(path))]
        elif path.is_dir():
            results = analyze_directory(str(path))
        else:
            print(f"Error: {args.path} is not a .nsys-rep file or directory")
            return 1

    elif args.json_path:
        path = Path(args.json_path)
        if path.is_file() and path.suffix == '.json':
            results = [analyze_json_file(str(path))]
        elif path.is_dir():
            results = analyze_json_directory(str(path))
        else:
            print(f"Error: {args.json_path} is not a .json file or directory")
            return 1

    if args.summary:
        print_summary(results)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "result.json"), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())