import json, subprocess, sys, time, os
task = sys.argv[1]; cwd = sys.argv[2]; out = sys.argv[3]
cmd = ["uv", "run", "python", "-m", "k2cascade.run", "--mode", "cascade", "--max-steps", "20",
       "--trace", out.replace(".jsonl", "-trace.jsonl"), "--cwd", cwd, "--task", task]
env = dict(os.environ, PYTHONUNBUFFERED="1")
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
with open(out, "w") as f:
    for line in p.stdout:
        f.write(json.dumps({"t": round(time.time() - t0, 3), "line": line.rstrip("\n")}) + "\n"); f.flush()
p.wait()
print("exit", p.returncode, "elapsed", round(time.time() - t0, 1))
