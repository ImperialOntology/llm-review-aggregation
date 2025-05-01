#!/bin/bash

echo ${PBS_O_WORKDIR}
# Read and export environment variables from config file
if [ -f "${PBS_O_WORKDIR}/config" ]; then
    echo "Loading environment variables from config file..."
    while IFS= read -r line; do
        # Skip empty lines and comments (lines starting with #)
        if [ -n "$line" ] && ! echo "$line" | grep -q '^#'; then
            # Export the variable
            export "$line"
            echo "Exported: $line"
        fi
    done < "${PBS_O_WORKDIR}/config"
else
    echo "config file not found. Exiting."
    exit 1
fi

# Set up conda
eval "$(~/miniforge3/bin/conda shell.bash hook)"

# Check if SETUP_ENV is true
if [ "$SETUP_ENV" = "true" ]; then
    echo "SETUP_ENV is true. Setting up the environment..."

    # Create and activate the conda environment
    conda create -n llm-env python=3.12 -y
    source activate "$ENV_NAME"

    # Install Python dependencies
    pip install -r $PBS_O_WORKDIR/requirements.txt
else
    echo "SETUP_ENV is not true. Skipping environment setup."
    # Activate the existing conda environment
    source activate "$ENV_NAME"
fi

cd $PBS_O_WORKDIR

python scripts/python_scripts/setup.py
if [ $? -ne 0 ]; then
    echo "Error: setup.py failed. Exiting."
    exit 1
fi
