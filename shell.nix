{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python310
    uv
  ];

  shellHook = ''
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
      echo "Creating new virtual environment..."
      uv venv --prompt "medAlpaca"
    fi

    # Set up some helpful environment variables
    export PYTHONPATH="$PWD:$PYTHONPATH"
    export PIP_PREFIX="$(pwd)/.venv"
    export PYTHON="python3.10"

    # Activate virtual environment
    source .venv/bin/activate
  '';
}