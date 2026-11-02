"""Tiny synthetic dataset (EMT energies, forces, stresses) for smoke tests of the training loop."""

import argparse

import ase.db
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT


def main(path, n=64, seed=0):
    rng = np.random.default_rng(seed)
    with ase.db.connect(path, append=False) as db:
        for i in range(n):
            el = ["Cu", "Al"][i % 2]
            atoms = bulk(el, "fcc", a=3.6 if el == "Cu" else 4.05, cubic=True).repeat((1, 1, 2))
            if i % 3 == 0:  # mixed composition
                atoms.symbols[rng.integers(len(atoms))] = "Al" if el == "Cu" else "Cu"
            atoms.rattle(0.05, seed=int(rng.integers(1 << 30)))
            atoms.calc = EMT()
            atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
            db.write(atoms)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--n", type=int, default=64)
    a = p.parse_args()
    main(a.path, a.n)
