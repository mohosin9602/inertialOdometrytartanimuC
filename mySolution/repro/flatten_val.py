#!/usr/bin/env python3
"""Present the val split as a flat directory of .npz files.

The starter script resolves trajectories as <test_root>/<traj_id>.npz, but the
val split ships nested by platform (val/car/car_val_0000.npz). Symlinks give the
script the flat layout it expects without copying 1 GB or touching the data.
"""
import os

from paths import DATA, OUT

PLATFORMS = ("car", "dog", "drone", "human")


def main():
    dest = os.path.join(OUT, "val_flat")
    os.makedirs(dest, exist_ok=True)

    n = 0
    for p in PLATFORMS:
        src_dir = os.path.join(DATA, "val", p)
        m = n
        for name in sorted(os.listdir(src_dir)):
            if not name.endswith(".npz"):
                continue
            link = os.path.join(dest, name)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.join(src_dir, name), link) # symlink is a special file system object that acts as a shortcut or pointer to another file or directory. When interacting with a symlink, operating system transparently redirects the operation to the actual target file.
            n += 1
        print(f"Linked {n-m} {PLATFORMS} trajectories into {dest}")
    print(f"linked {n} trajectories into {dest} in total. (expected 80)")


if __name__ == "__main__":
    main()
