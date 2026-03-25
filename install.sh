# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Install in editable mode directly from source
pip install -e .

# Open users' original directory
cd "$original_dir"
