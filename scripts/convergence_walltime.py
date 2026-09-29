"""Set only this pilot's dependent convergence job walltime from measured cost."""

import getpass
import json
import math
from pathlib import Path
import re
import subprocess


def recommended_hours(predicted_hours):
    if not math.isfinite(predicted_hours) or predicted_hours <= 0:
        raise ValueError('Invalid measured runtime estimate')
    hours = max(16, 2 * math.ceil((1.25 * predicted_hours + .5) / 2))
    if hours > 24:
        raise ValueError(f'Pilot requires {hours} h including margin, above the partition 24 h cap; replan before production')
    return hours


def configure_dependent_job(report_path, pilot_id):
    path = Path(report_path)
    report = json.loads(path.read_text())
    hours = recommended_hours(report['predicted_finest_hours'])
    report['recommended_wall_hours'] = hours
    if pilot_id:
        listing = subprocess.check_output([
            'squeue', '--noheader', '--user=' + getpass.getuser(),
            '--name=PMMA-DT-0368-0383', '--format=%i|%T|%E',
        ], text=True)
        matches = []
        for row in listing.splitlines():
            job, state, dependency = row.strip().split('|', 2)
            if re.search(r'afterok:' + re.escape(str(pilot_id)) + r'(?!\d)', dependency):
                if state != 'PENDING' or not job.isdecimal():
                    raise RuntimeError('Dependent job is no longer pending; refusing to alter it')
                matches.append(job)
        if len(matches) > 1:
            raise RuntimeError('Duplicate convergence submissions; refusing automatic changes')
        for job in matches:
            subprocess.run(['scontrol', 'update', 'JobId=' + job,
                            f'TimeLimit={hours}:00:00'], check=True)
            report['production_job_id'] = job
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Convergence wall limit from measured pilot: {hours} h', flush=True)


def slurm_budget_seconds(job_id):
    limit = subprocess.check_output(['squeue', '--noheader', '--jobs=' + job_id,
                                     '--format=%l'], text=True).strip()
    days, clock = limit.split('-', 1) if '-' in limit else ('0', limit)
    parts = [int(x) for x in clock.split(':')]
    if len(parts) == 2:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    budget = int(days) * 86400 + hours * 3600 + minutes * 60 + seconds - 3600
    if budget < 3600:
        raise ValueError('Insufficient Slurm time for production and checkpoint margin')
    return budget


if __name__ == '__main__':
    import os
    print(slurm_budget_seconds(os.environ['SLURM_JOB_ID']))
