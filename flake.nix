{
  description = "python_flake";

  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }: let
    pkgs = nixpkgs.legacyPackages."x86_64-linux";
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      packages = with pkgs; [
        (pkgs.python313.withPackages(pypkgs: with pypkgs; [
          pint
        ]))
        ruff
        pyright
        mypy
      ];
    };
  };
}
