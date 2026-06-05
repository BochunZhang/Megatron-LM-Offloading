#!/usr/bin/env python3
"""
Extract throughput per GPU from train.log files and export to Excel.

Usage:
    python extract_throughput.py <output_xlsx_path> <train_log_path>...

Example:
    python extract_throughput.py throughput_summary.xlsx /path/to/train.log /path2/to/train.log
"""

import argparse
import os
import re
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


def extract_throughput_from_log(log_path: str) -> dict:
    """
    Extract throughput data from a single train.log file.

    Returns:
        dict with keys: file_name, pcie_bus, throughput_value, unit, found
    """
    result = {
        'file_name': os.path.basename(log_path),
        'file_path': log_path,
        'pcie_bus': '',
        'throughput_value': None,
        'unit': 'TFLOPS/s/GPU',
        'found': False
    }

    # Try to extract PCIe bus from file path (e.g., 0009_01_00.0 or 0009:01:00.0)
    path_str = str(log_path)
    pcie_pattern = r'([0-9a-fA-F]{4}[_:][0-9a-fA-F]{2}[_:][0-9a-fA-F]{2}\.[0-9a-fA-F])'
    pcie_match = re.search(pcie_pattern, path_str)
    if pcie_match:
        result['pcie_bus'] = pcie_match.group(1).replace('_', ':')

    # Read and parse log file
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"  Warning: Failed to read {log_path}: {e}")
        return result

    # Look for throughput pattern: "throughput per GPU (TFLOPS/s/GPU): xxxx"
    # Support various formats like:
    #   throughput per GPU (TFLOPS/s/GPU): 123.45
    #   throughput per GPU (TFLOPS/s/GPU):  123.45
    pattern = r'throughput per GPU \(TFLOPS/s/GPU\):\s*([\d.]+)'
    match = re.search(pattern, content, re.IGNORECASE)

    if match:
        result['throughput_value'] = float(match.group(1))
        result['found'] = True

    return result


def generate_excel(output_path: str, results: list):
    """Generate Excel file with throughput data."""
    if not OPENPYXL_AVAILABLE:
        print("Error: openpyxl not available. Install with: pip install openpyxl")
        return False

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Throughput Summary"

    # Header style
    header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
    header_font = Font(bold=True, color='FFFFFF')
    center_align = Alignment(horizontal='center', vertical='center')

    # Headers
    headers = ['File Name', 'PCIe Bus', 'Throughput per GPU (TFLOPS/s/GPU)']
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align

    # Data rows
    row = 2
    for result in results:
        if result['found']:
            ws.cell(row=row, column=1, value=result['file_name'])
            ws.cell(row=row, column=2, value=result['pcie_bus'])
            ws.cell(row=row, column=3, value=result['throughput_value'])

            # Center alignment for all cells
            for col in range(1, 4):
                ws.cell(row=row, column=col).alignment = center_align

            row += 1

    # Add summary row if we have data
    if row > 2:
        # Empty row
        row += 1

        # Summary statistics
        values = [r['throughput_value'] for r in results if r['found']]
        if values:
            avg_value = sum(values) / len(values)
            max_value = max(values)
            min_value = min(values)

            summary_fill = PatternFill(start_color='E7E6E6', end_color='E7E6E6', fill_type='solid')
            summary_font = Font(bold=True)

            ws.cell(row=row, column=1, value="Statistics")
            ws.cell(row=row, column=1).font = summary_font
            ws.cell(row=row, column=1).fill = summary_fill
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
            ws.cell(row=row, column=3, value=f"Count: {len(values)}")
            ws.cell(row=row, column=3).alignment = center_align
            row += 1

            ws.cell(row=row, column=1, value="Average")
            ws.cell(row=row, column=1).font = summary_font
            ws.cell(row=row, column=1).fill = summary_fill
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
            ws.cell(row=row, column=3, value=round(avg_value, 4))
            ws.cell(row=row, column=3).alignment = center_align
            row += 1

            ws.cell(row=row, column=1, value="Max")
            ws.cell(row=row, column=1).font = summary_font
            ws.cell(row=row, column=1).fill = summary_fill
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
            ws.cell(row=row, column=3, value=max_value)
            ws.cell(row=row, column=3).alignment = center_align
            row += 1

            ws.cell(row=row, column=1, value="Min")
            ws.cell(row=row, column=1).font = summary_font
            ws.cell(row=row, column=1).fill = summary_fill
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
            ws.cell(row=row, column=3, value=min_value)
            ws.cell(row=row, column=3).alignment = center_align

    # Adjust column widths
    ws.column_dimensions['A'].width = 40
    ws.column_dimensions['B'].width = 25
    ws.column_dimensions['C'].width = 35

    # Save workbook
    wb.save(output_path)
    print(f"  Throughput summary saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract throughput per GPU from train.log files and export to Excel'
    )
    parser.add_argument('output_xlsx', help='Output Excel file path')
    parser.add_argument('train_logs', nargs='+', help='Train log file(s) to process')

    args = parser.parse_args()

    if not OPENPYXL_AVAILABLE:
        print("Error: openpyxl not available. Install with: pip install openpyxl")
        return 1

    print("=" * 60)
    print("Extracting throughput data from train.log files")
    print("=" * 60)

    results = []
    processed = 0
    found_count = 0

    for log_path in args.train_logs:
        log_path = os.path.abspath(log_path)
        if not os.path.exists(log_path):
            print(f"  Warning: File not found: {log_path}")
            continue

        print(f"  Processing: {log_path}")
        result = extract_throughput_from_log(log_path)
        results.append(result)
        processed += 1

        if result['found']:
            print(f"    ✓ Found throughput: {result['throughput_value']} {result['unit']}")
            if result['pcie_bus']:
                print(f"      PCIe Bus: {result['pcie_bus']}")
            found_count += 1
        else:
            print(f"    ✗ No throughput data found")

    print(f"\nProcessed {processed} file(s), found throughput data in {found_count} file(s)")

    if results:
        generate_excel(args.output_xlsx, results)

    print("=" * 60)
    return 0


if __name__ == '__main__':
    exit(main())
