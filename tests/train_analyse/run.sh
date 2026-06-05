#!/bin/bash
# NSYS Profile Analysis Pipeline
# Scans for .nsys-rep files and generates xlsx reports
#
# Usage:
#   ./run.sh [--iteration N] [--detail] <SCAN_PATH>
#
# Examples:
#   ./run.sh ./logs/                          # Scan directory recursively
#   ./run.sh --detail ./logs/                 # Enable detail mode (output JSON)
#   ./run.sh --iteration 20 ./profile.nsys-rep # Analyze specific file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
ITERATION=16
SCAN_DIR=""
SPECIFIC_FILE=""
DETAIL_MODE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iteration|-i)
            ITERATION="$2"
            shift 2
            ;;
        --detail|-d)
            DETAIL_MODE="--detail"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] <SCAN_PATH>"
            echo ""
            echo "Options:"
            echo "  --iteration N    Iteration number to analyze (default: 16)"
            echo "  --detail         Enable detail mode to export JSON files"
            echo "  --help          Show this help message"
            echo ""
            echo "Arguments:"
            echo "  SCAN_PATH       Directory to scan or specific .nsys-rep file"
            echo ""
            echo "Output:"
            echo "  Results saved to <SCAN_PATH>-result/ subdirectory"
            exit 0
            ;;
        -*)
            echo "Error: Unknown option: $1"
            exit 1
            ;;
        *)
            if [[ -f "$1" && "$1" == *.nsys-rep ]]; then
                SPECIFIC_FILE="$1"
            elif [[ -d "$1" ]]; then
                SCAN_DIR="$1"
            else
                echo "Error: Path not found: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate input
if [[ -z "$SCAN_DIR" && -z "$SPECIFIC_FILE" ]]; then
    echo "Error: Please specify a directory or .nsys-rep file"
    echo "Usage: $0 [OPTIONS] <SCAN_PATH>"
    exit 1
fi

# Determine output base directory
if [[ -n "$SPECIFIC_FILE" ]]; then
    # For specific file, use its parent directory
    INPUT_BASE="$(dirname "$SPECIFIC_FILE")"
    OUTPUT_BASE="${INPUT_BASE}-result"
else
    # For directory scan
    INPUT_BASE="$SCAN_DIR"
    OUTPUT_BASE="${SCAN_DIR}-result"
fi

echo "=========================================="
echo "NSYS Profile Analysis Pipeline"
echo "=========================================="
echo "Input: ${SPECIFIC_FILE:-$SCAN_DIR}"
echo "Output: ${OUTPUT_BASE}"
echo "Iteration: ${ITERATION}"
echo "Detail mode: ${DETAIL_MODE:-disabled}"
echo ""

# Check nsys CLI availability
if ! command -v nsys &> /dev/null; then
    echo "Warning: nsys command not found in PATH"
    echo "Make sure NSight Systems is installed and nsys is available"
fi

# Function to process a single .nsys-rep file
process_nsys_rep() {
    local nsys_file="$1"
    # Calculate relative path from INPUT_BASE
    local abs_nsys_file="$(cd "$(dirname "$nsys_file")" && pwd)/$(basename "$nsys_file")"
    local abs_input_base="$(cd "$INPUT_BASE" && pwd)"
    local rel_path="${abs_nsys_file#$abs_input_base/}"
    local base_name=$(basename "${nsys_file}" .nsys-rep)
    local rel_dir=$(dirname "$rel_path")

    # Output directory: xxx-result/aaa/
    local output_dir
    if [[ "$rel_dir" == "." ]]; then
        output_dir="${OUTPUT_BASE}"
    else
        output_dir="${OUTPUT_BASE}/${rel_dir}"
    fi
    local json_file="${output_dir}/${base_name}.json"
    local sqlite_file="${output_dir}/${base_name}.sqlite"
    local xlsx_file="${output_dir}/${base_name}.xlsx"

    echo "Processing: ${nsys_file}"
    echo "  Relative path: ${rel_path}"
    echo "  Output dir: ${output_dir}"
    echo "  Output file: ${xlsx_file}"

    # Create output directory
    mkdir -p "${output_dir}"

    # Export to JSON if not exists or older than source
    if [[ ! -f "${json_file}" ]] || [[ "${nsys_file}" -nt "${json_file}" ]]; then
        echo "  Exporting to JSON..."
        if command -v nsys &> /dev/null; then
            nsys export -t json "${nsys_file}" -o "${json_file}" || {
                echo "  Warning: JSON export failed"
                return 1
            }
        else
            echo "  Warning: nsys not available, skipping JSON export"
            return 1
        fi
    else
        echo "  JSON already exists: ${json_file}"
    fi

    # Create SQLite copy if not exists
    if [[ ! -f "${sqlite_file}" ]] || [[ "${nsys_file}" -nt "${sqlite_file}" ]]; then
        echo "  Creating SQLite copy..."
        if command -v nsys &> /dev/null; then
            nsys export -t sqlite "${nsys_file}" -o "${sqlite_file}" 2>/dev/null || {
                echo "  Using original .nsys-rep as SQLite source"
                ln -sf "$(realpath "${nsys_file}")" "${sqlite_file}" 2>/dev/null || \
                    cp "${nsys_file}" "${sqlite_file}" 2>/dev/null || true
            }
        fi
    else
        echo "  SQLite already exists: ${sqlite_file}"
    fi

    # Run Python analyzer
    if [[ -f "${json_file}" ]]; then
        echo "  Running Python analyzer..."
        PYENV_VERSION=megatron-py3.13.9 pyenv exec python3 "${SCRIPT_DIR}/nsys_profile_analyzer.py" \
            --sqlite "${sqlite_file}" \
            --json "${json_file}" \
            --output "${output_dir}" \
            --iteration "${ITERATION}" \
            ${DETAIL_MODE} || {
            echo "  Warning: Python analysis failed"
            return 1
        }

        # Check if xlsx was generated
        if [[ -f "${xlsx_file}" ]]; then
            echo "  ✓ Generated: ${xlsx_file}"
        else
            echo "  Warning: Expected output file not found"
        fi
    fi

    echo ""
}

# Main processing
if [[ -n "$SPECIFIC_FILE" ]]; then
    # Process specific file
    process_nsys_rep "${SPECIFIC_FILE}"
else
    # Scan directory recursively for .nsys-rep files
    echo "Scanning recursively: ${SCAN_DIR}"
    echo ""

    found=0
    while IFS= read -r -d '' file; do
        process_nsys_rep "${file}"
        found=1
    done < <(find "${SCAN_DIR}" -name "*.nsys-rep" -type f -print0 2>/dev/null)

    if [[ $found -eq 0 ]]; then
        echo "No .nsys-rep files found in ${SCAN_DIR}"
        echo ""
        exit 1
    fi
fi

echo "=========================================="
echo "Analysis complete!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "=========================================="
