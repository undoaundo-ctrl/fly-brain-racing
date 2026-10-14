{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  buildInputs = with pkgs; [ python311 swig gcc pkg-config SDL2 ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:$LD_LIBRARY_PATH
    [ -d .venv ] || python -m venv .venv
    source .venv/bin/activate
    pip install -q -r requirements.txt
  '';
}
