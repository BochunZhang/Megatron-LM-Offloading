#!/bin/bash
# NSYS Profile Analysis Pipeline
# Scans for .nsys-rep files, exports to JSON/SQLite, and runs Python analysis
#
# Usage:
#   ./run.sh [--iteration N] [--output-dir DIR] [NSYS_REP_PATH]
#
# Examples:
#   ./run.sh                                    # Scan current directory
#   ./run.sh logs/nsys-profile/                 # Scan specific directory
#   ./run.sh --iteration 20 /path/to/file.nsys-rep  # Analyze specific file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
ITERATION=16
OUTPUT_DIR="${PROJECT_ROOT}/documents/train_analyse"
SCAN_DIR="."
SPECIFIC_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iteration|-i)
            ITERATION="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [PATH]"
            echo ""
            echo "Options:"
            echo "  --iteration N    Iteration number to analyze (default: 16)"
            echo "  --output-dir DIR Output directory (default: documents/train_analyse)"
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

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "NSYS Profile Analysis Pipeline"
echo "=========================================="
echo "Iteration: ${ITERATION}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Check nsys CLI availability
if ! command -v nsys &> /dev/null; then
    echo "Warning: nsys command not found in PATH"
    echo "Make sure NSight Systems is installed and nsys is available"
fi

# Function to process a single .nsys-rep file
process_nsys_rep() {
    local nsys_file="$1"
    local base_name=$(basename "${nsys_file}" .nsys-rep)
    local json_file="${OUTPUT_DIR}/${base_name}.json"
    local sqlite_file="${OUTPUT_DIR}/${base_name}.sqlite"

    echo "Processing: ${nsys_file}"
    echo "  Base name: ${base_name}"

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
            # Export to sqlite
            cp "${nsys_file}" "${sqlite_file}.tmp" 2>/dev/null || true
            # Actually we need to export to sqlite format
            # Try nsys export sqlite
            nsys export -t sqlite "${nsys_file}" -o "${sqlite_file}" 2>/dev/null || {
                # Fallback: create a symbolic link with .sqlite extension
                # and let Python use the original .nsys-rep file
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
        python3 "${SCRIPT_DIR}/nsys_profile_analyzer.py" \
            --sqlite "${sqlite_file}" \
            --json "${json_file}" \
            --output "${OUTPUT_DIR}/${base_name}_analysis" \
            --iteration "${ITERATION}" || {
            echo "  Warning: Python analysis failed"
            return 1
        }
    fi

    echo "  Done: ${base_name}"
    echo ""
}

# Main processing
if [[ -n "${SPECIFIC_FILE}" ]]; then
    # Process specific file
    process_nsys_rep "${SPECIFIC_FILE}"
else
    # Scan directory for .nsys-rep files
    echo "Scanning: ${SCAN_DIR}"
    echo ""

    found=0
    while IFS= read -r -d '' file; do
        process_nsys_rep "${file}"
        found=1
    done < <(find "${SCAN_DIR}" -maxdepth 2 -name "*.nsys-rep" -type f -print0 2>/dev/null)

    if [[ $found -eq 0 ]]; then
        echo "No .nsys-rep files found in ${SCAN_DIR}"
        echo ""
        echo "Searched paths:"
        echo "  - ${SCAN_DIR}"
        echo "  - ${SCAN_DIR}/*/"
        echo ""
        echo "You can specify a file directly:"
        echo "  $0 /path/to/profile.nsys-rep"
        exit 1
    fi
fi

echo "=========================================="
echo "Analysis complete!"
echo "Results in: ${OUTPUT_DIR}"
echo "=========================================="
