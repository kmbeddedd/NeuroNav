import subprocess
import json

def run_cmd(cmd):
    return subprocess.check_output(cmd, shell=True, text=True, errors='replace')

# 1. Inspect kkkk commits (from origin)
origin_refs = ['origin/main', 'origin/sumit', 'origin/Kunal']
origin_commits_raw = run_cmd('git rev-list --no-merges ' + ' '.join(origin_refs)).strip().splitlines()
origin_merges_raw = run_cmd('git rev-list --merges ' + ' '.join(origin_refs)).strip().splitlines()
all_origin = list(dict.fromkeys(origin_commits_raw + origin_merges_raw))

print(f"Total commits in kkkk (origin): {len(all_origin)}")
print("--- Commits in kkkk ---")
for c in all_origin:
    info = run_cmd(f'git log -1 --pretty=format:"%h | %ai | %an | %s" {c}')
    print(info)

# 2. Inspect all neuronav branches
neuronav_branches = ['neuronav/main', 'neuronav/kunal', 'neuronav/amit', 'neuronav/panda', 'neuronav/Sumit', 'Kunal']
neuronav_commits_raw = run_cmd('git rev-list --no-merges ' + ' '.join(neuronav_branches)).strip().splitlines()
neuronav_merges_raw = run_cmd('git rev-list --merges ' + ' '.join(neuronav_branches)).strip().splitlines()
all_neuronav = list(dict.fromkeys(neuronav_commits_raw + neuronav_merges_raw))

unique_neuronav = [c for c in all_neuronav if c not in all_origin]

print(f"\nTotal commits across ALL neuronav branches: {len(unique_neuronav)}")
print("--- Commits in neuronav ---")
for c in unique_neuronav:
    info = run_cmd(f'git log -1 --pretty=format:"%h | %cd | %an | %s" --date=iso {c}')
    print(info)

