#!/bin/bash

echo $pwd

# Directory containing the .pgf files
PGF_DIR="results"

# Get the search string from the first argument
SEARCH_STRING="$1"

# Find .pgf files matching the search string
if [ -z "$SEARCH_STRING" ]; then
    # No search string provided, process all files
    FILES=("$PGF_DIR"/*.pgf)
else
    # Search string provided, process only matching files
    FILES=("$PGF_DIR/"*"$SEARCH_STRING"*.pgf)
fi

# Check if there are any .pgf files
if [ ${#FILES[@]} -eq 0 ] || [ ! -e "${FILES[0]}" ]; then
    echo "No .pgf files matching '$SEARCH_STRING' found in $PGF_DIR."
    exit 1
fi

TOTAL_FILES=${#FILES[@]}

# Counter for progress
COUNT=0

echo "Compiling PGF plots to PDF..."

# For each .pgf file
for PGF_FILE in "${FILES[@]}"; do
    # Increment the counter
    COUNT=$((COUNT + 1))

    # Get the base filename without extension
    BASENAME=$(basename "$PGF_FILE" .pgf)

    # Create the .tex wrapper file
    TEX_FILE="$PGF_DIR/$BASENAME.tex"

    cat > "$TEX_FILE" <<EOF
\documentclass{standalone}
\def\mathdefault#1{#1}
\usepackage{times}
\usepackage{pgf}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\input{$BASENAME.pgf}
\end{document}
EOF

    # Compile the .tex file to produce .pdf
    export TEXINPUTS=.:../../:
    pdflatex -output-directory "$PGF_DIR"  -interaction=nonstopmode "$TEX_FILE" >/dev/null 2>&1
    # luatex -output-directory "$PGF_DIR" "$TEX_FILE" 

    # Remove temporary files
    rm "$TEX_FILE"
    rm "$PGF_DIR/$BASENAME.aux" "$PGF_DIR/$BASENAME.log"

    # Show progress
    PROGRESS=$((COUNT * 100 / TOTAL_FILES))
    echo -ne "[$COUNT/$TOTAL_FILES] Compiled $BASENAME.pgf to PDF. Progress: $PROGRESS% \r"

    # break
done

echo -e "\nAll PGF plots have been compiled to PDF."