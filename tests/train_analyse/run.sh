#!/bin/bash
# NSYS Profile Analysis Pipeline
# Scans for .nsys-rep files, exports to JSON/SQLite, and runs Python analysis
#
# Usage:
#   ./run.sh [--iteration N] [--detail] [PATH]
#
# Examples:
#   ./run.sh                                    # Scan current directory recursively
#   ./run.sh /path/to/logs/                     # Scan specific directory recursively
#   ./run.sh --detail /path/to/logs/            # Enable detail mode (output JSON)
#   ./run.sh --iteration 20 /path/to/file.nsys-rep  # Analyze specific file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
ITERATION=16
SCAN_DIR="."
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
            echo "Usage: $0 [OPTIONS] [PATH]"
            echo ""
            echo "Options:"
            echo "  --iteration N    Iteration number to analyze (default: 16)"
            echo "  --detail         Enable detail mode to export JSON files"
            echo "  --help          Show this help message"
            echo ""
            echo "Arguments:"
            echo "  PATH            Directory to scan or specific .nsys-rep file"
            exit 0
            ;;
        *)
            if [[ -f "$1" && "$1" == *.nsys-rep ]]; then
                SPECIFIC_FILE="$1"
            elif [[ -d "$1" ]]; then
                SCAN_DIR="$1"
            else
                echo "Error: Unknown option or invalid path: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

echo "=========================================="
echo "NSYS Profile Analysis Pipeline"
echo "=========================================="
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
    local file_dir=$(dirname "${nsys_file}")
    local base_name=$(basename "${nsys_file}" .nsys-rep)
    local analyse_dir="${file_dir}/analyse"
    local json_file="${analyse_dir}/${base_name}.json"
    local sqlite_file="${analyse_dir}/${base_name}.sqlite"

    echo "Processing: ${nsys_file}"
    echo "  Base name: ${base_name}"
    echo "  Analyse dir: ${analyse_dir}"

    # Create analyse directory
    mkdir -p "${analyse_dir}"

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
            --output "${analyse_dir}" \
            --iteration "${ITERATION}" \
            ${DETAIL_MODE} || {
            echo "  Warning: Python analysis failed"
            return 1
        }
    fi

    echo "  Done: ${base_name}"
    echo "  Results saved to: ${analyse_dir}"
    echo ""
}

# Main processing
if [[ -n "${SPECIFIC_FILE}" ]]; then
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
        echo "Searched recursively in: ${SCAN_DIR}"
        echo ""
        echo "You can specify a file directly:"
        echo "  $0 /path/to/profile.nsys-rep"
        exit 1
    fi
fi

echo "=========================================="
echo "Analysis complete!"
echo "Results saved in analyse/ subdirectories"
echo "=========================================="
