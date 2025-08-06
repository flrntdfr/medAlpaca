{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python310
    uv
  ];

  env = {
    PYTHONPATH = "$PWD:$PYTHONPATH";
    PIP_PREFIX = "$(pwd)/.venv";
    PYTHON = "python3.10";
  };

  shellHook = ''
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
      echo "Creating new virtual environment..."
      uv venv --prompt "medAlpaca"
    fi

    # Activate virtual environment
    source .venv/bin/activate
  '';
}