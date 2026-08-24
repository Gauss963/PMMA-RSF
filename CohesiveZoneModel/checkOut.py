import time
import subprocess
import glob
import os

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

cmd = ["squeue", "-u", "gauss112"]
out_dir = "/home/gauss112/outs"

while True:
    print("\n\n           -------------------------------------------------------------------------")
    subprocess.run(cmd)
    print("           -------------------------------------------------------------------------\n\n")

    files = sorted(
        glob.glob(os.path.join(out_dir, "Akantu-*.out")),
        key=os.path.getmtime
    )

    if files:
        subprocess.run(["tail", "-n", "4", files[-1]])
    else:
        print("No Akantu output files found.")

    time.sleep(5)
    clear_terminal()