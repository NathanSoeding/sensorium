#!/usr/bin/env python
import subprocess
import sys
import os
import argparse

def create_run(base_path, name, extra_args):
    # Setup directories
    path = os.path.join(base_path, name)
    os.makedirs(path, exist_ok=True)

    
    # Build command
    cmd = [
        'nohup', 'python', 'train_base.py',
        '--output_dir', path,
        '--wandb_run_name', name,
        *extra_args,
    ]
    
    # Redirect output
    with open(f"{path}/nohup.out", "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    
    # Save PID
    with open(f"{path}/pid.txt", "w") as f:
        f.write(str(process.pid))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    
    args, unknown = parser.parse_known_args()
    create_run(args.path, args.name, unknown)