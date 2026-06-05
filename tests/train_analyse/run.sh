#!/bin/bash
# NSYS Profile Analysis Pipeline
# Scans for .nsys-rep files and generates xlsx reports
#
# Usage:
#   ./run.sh [--iteration N] [--rank "0,1,2,3"] <SCAN_PATH>
#
# Examples:
#   ./run.sh ./logs/                          # Scan directory recursively
#   ./run.sh --rank "0,1" ./logs/             # Analyze ranks 0 and 1
#   ./run.sh --iteration 20 ./profile.nsys-rep  # Analyze specific file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
ITERATION=16
RANKS="0"
SCAN_DIR=""
SPECIFIC_FILE=""

# Array to track failed items
declare -a FAILED_ITEMS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iteration|-i)
            ITERATION="$2"
            shift 2
            ;;
        --rank|-r)
            RANKS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] <SCAN_PATH>"
            echo ""
            echo "Options:"
            echo "  --iteration N    Iteration number to analyze (default: 16)"
            echo "  --rank RANKS     Comma-separated ranks to analyze (default: 0)"
            echo "  --help          Show this help message"
            echo ""
            echo "Arguments:"
            echo "  SCAN_PATH       Directory to scan or specific .nsys-rep file"
            echo ""
            echo "Output:"
            echo "  JSON/SQLite: Same directory as .nsys-rep file"
            echo "  Excel: <nsys-rep-path>-result/ subdirectory"
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

# Convert comma-separated ranks to space-separated for python
RANKS_ARG="${RANKS//,/ }"

echo "=========================================="
echo "NSYS Profile Analysis Pipeline"
echo "=========================================="
echo "Input: ${SPECIFIC_FILE:-$SCAN_DIR}"
echo "Ranks: ${RANKS}"
echo "Iteration: ${ITERATION}"
echo ""

# Check nsys CLI availability
if ! command -v nsys &> /dev/null; then
    echo "Warning: nsys command not found in PATH"
fi

# Function to process train.log for throughput extraction
process_train_log() {
    local nsys_file="$1"
    local output_dir="$2"
    local file_dir=$(dirname "$nsys_file")
    local train_log="${file_dir}/train.log"

    # Only process if train.log exists
    if [[ ! -f "${train_log}" ]]; then
        return 0
    fi

    echo "  Found train.log, extracting throughput data..."

    # Output path for throughput Excel (same directory as nsys-rep output)
    local throughput_xlsx="${output_dir}/throughput_summary.xlsx"

    # Run Python script to extract throughput
    python3 "${SCRIPT_DIR}/extract_throughput.py" \
        "${throughput_xlsx}" \
        "${train_log}"

    local python_exit_code=$?
    if [[ $python_exit_code -ne 0 ]]; then
        echo "  Warning: Throughput extraction failed with exit code $python_exit_code"
    elif [[ -f "${throughput_xlsx}" ]]; then
        echo "  ✓ Throughput data extracted to: ${throughput_xlsx}"
    fi

    return 0
}

# Function to process a single .nsys-rep file
process_nsys_rep() {
    local nsys_file="$1"
    local file_dir=$(dirname "$nsys_file")
    local base_name=$(basename "${nsys_file}" .nsys-rep)

    # Input files (same directory as nsys-rep)
    local json_file="${file_dir}/${base_name}.json"
    local sqlite_file="${file_dir}/${base_name}.sqlite"

    # Output directory: xxx-result/aaa/
    # For xxx/aaa/bbb.nsys-rep -> xxx-result/aaa/
    local abs_file_dir="$(cd "$file_dir" && pwd)"
    local parent_dir="$(dirname "$abs_file_dir")"
    local dir_name="$(basename "$abs_file_dir")"
    local output_dir="${parent_dir}-result/${dir_name}"

    echo "=========================================="
    echo "Processing: ${nsys_file}"
    echo "  JSON/SQLite dir: ${file_dir}"
    echo "  Excel output dir: ${output_dir}"
    echo "=========================================="

    # Check if JSON and SQLite already exist
    local need_export=false
    if [[ ! -f "${json_file}" ]] || [[ ! -f "${sqlite_file}" ]]; then
        need_export=true
        echo "  JSON/SQLite not found, need to export"
    elif [[ "${nsys_file}" -nt "${json_file}" ]] || [[ "${nsys_file}" -nt "${sqlite_file}" ]]; then
        need_export=true
        echo "  JSON/SQLite outdated, need to re-export"
    else
        echo "  Using existing JSON: ${json_file}"
        echo "  Using existing SQLite: ${sqlite_file}"
    fi

    # Export to JSON/SQLite if needed
    if [[ "$need_export" == true ]]; then
        if command -v nsys &> /dev/null; then
            echo "  Exporting to JSON..."
            nsys export -t json "${nsys_file}" -o "${json_file}" || {
                echo "  Error: JSON export failed"
                FAILED_ITEMS+=("${nsys_file}: JSON export failed")
                return 1
            }

            echo "  Exporting to SQLite..."
            nsys export -t sqlite "${nsys_file}" -o "${sqlite_file}" || {
                echo "  Warning: SQLite export failed, using .nsys-rep as fallback"
                ln -sf "$(realpath "${nsys_file}")" "${sqlite_file}" 2>/dev/null || \
                    cp "${nsys_file}" "${sqlite_file}" 2>/dev/null || true
            }
        else
            echo "  Error: nsys not available and JSON/SQLite not found"
            FAILED_ITEMS+=("${nsys_file}: nsys not available and JSON/SQLite not found")
            return 1
        fi
    fi

    # Create Excel output directory
    mkdir -p "${output_dir}"

    # Check if xlsx files already exist and are newer than source
    local xlsx_exists=false
    local xlsx_count=$(find "${output_dir}" -name "*.xlsx" -type f 2>/dev/null | wc -l)
    if [[ $xlsx_count -gt 0 ]]; then
        # Check if any xlsx is newer than json file
        local newest_xlsx=$(find "${output_dir}" -name "*.xlsx" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [[ -n "$newest_xlsx" && "$newest_xlsx" -nt "$json_file" ]]; then
            echo "  Excel files already exist and are up-to-date: ${output_dir}"
            echo "  ✓ Found $xlsx_count Excel file(s), skipping analysis"
            # Still process train.log for throughput extraction
            process_train_log "${nsys_file}" "${output_dir}"
            echo ""
            return 0
        fi
    fi

    # Run Python analyzer
    if [[ -f "${json_file}" ]]; then
        echo "  Running Python analyzer..."
        python3 "${SCRIPT_DIR}/nsys_profile_analyzer.py" \
            --sqlite "${sqlite_file}" \
            --json "${json_file}" \
            --output "${output_dir}" \
            --iteration "${ITERATION}" \
            --rank ${RANKS_ARG}

        local python_exit_code=$?
        if [[ $python_exit_code -ne 0 ]]; then
            echo "  Warning: Python analysis failed with exit code $python_exit_code"
            FAILED_ITEMS+=("${nsys_file}: Python analysis failed (exit $python_exit_code)")
            return 1
        fi

        # Check if xlsx files were generated
        local xlsx_count=$(find "${output_dir}" -name "*.xlsx" 2>/dev/null | wc -l)
        if [[ $xlsx_count -gt 0 ]]; then
            echo "  ✓ Generated $xlsx_count Excel file(s) in: ${output_dir}"
            # Process train.log for throughput extraction
            process_train_log "${nsys_file}" "${output_dir}"
        else
            echo "  Warning: No Excel files found in output directory"
            FAILED_ITEMS+=("${nsys_file}: No Excel output generated")
            return 1
        fi
    else
        echo "  Error: JSON file not available: ${json_file}"
        FAILED_ITEMS+=("${nsys_file}: JSON file not available")
        return 1
    fi

    echo ""
    return 0
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
echo "=========================================="

# Report failed items
if [[ ${#FAILED_ITEMS[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: The following items failed:"
    echo "------------------------------------------"
    for item in "${FAILED_ITEMS[@]}"; do
        echo "  ✗ ${item}"
    done
    echo "------------------------------------------"
    echo "Total failed: ${#FAILED_ITEMS[@]}"
    exit 1
else
    echo "All items processed successfully!"
    exit 0
fi
