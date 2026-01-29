import docker
import time
import csv
import json
import os
import random
import uuid
import platform
import logging
import glob
from datetime import datetime

# ==========================================
# 🧪 Experimental Configuration
# ==========================================
# [Config] Batch Processing & Stability
BATCH_SIZE = 20        # Restart environment every 20 trials
MAX_RETRIES = 3        # Max retries per configuration
RETRY_DELAY_BASE = 5   # Base delay for exponential backoff (seconds)
BATCH_COOLDOWN = 15    # Extended cooldown to clear TIME_WAIT sockets

NETWORK_NAME = "cts_paper_net"
DATA_FILE_PREFIX = "pareto_results_batch"
FINAL_DATA_FILE = "pareto_results_final.csv"

CLIENT_SCRIPT = os.path.abspath("pareto_client.py")
NGINX_CONF = os.path.abspath("nginx.conf")
DATA_BIN = os.path.abspath("data.bin")

# [Factors]
SCENARIOS = {
    "Weak":   {"bw": 5,   "delay": 400, "loss": 1.0},
    "Edge":   {"bw": 20,  "delay": 50,  "loss": 0.1},
    "Cloud":  {"bw": 100, "delay": 10,  "loss": 0.0}
}
CPU_QUOTAS = ["1.0", "2.0"] 
THREADS = [1, 2, 4, 8, 16]
CHUNKS = [0.1, 1.0, 4.0]
FILE_SIZES = [10, 100, 400] 
BASE_PAYLOAD_SIZE = 500 
REPEAT_COUNT = 3 

client = docker.from_env()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_host():
    """Environment Preparation"""
    if not os.path.exists(DATA_BIN) or os.path.getsize(DATA_BIN) < BASE_PAYLOAD_SIZE * 1024 * 1024:
        logging.info(f"Generating {BASE_PAYLOAD_SIZE}MB random payload...")
        os.system(f"dd if=/dev/urandom of={DATA_BIN} bs=1M count={BASE_PAYLOAD_SIZE}")
    
    with open(NGINX_CONF, 'w') as f:
        f.write("""
events { worker_connections 4096; }
http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    server {
        listen 80;
        root /usr/share/nginx/html;
    }
}
""")

def get_system_metadata():
    return {
        "kernel": platform.release(),
        "arch": platform.machine(),
        "docker_ver": client.version()['Version'],
        "cpu_info": os.popen("lscpu | grep 'Model name'").read().strip()
    }

def start_nginx_server(net):
    # [Optimization] Apply sysctl to prevent TIME_WAIT accumulation
    sysctls = {
        "net.ipv4.tcp_tw_reuse": "1",
        "net.ipv4.tcp_fin_timeout": "30"
    }
    
    return client.containers.run(
        "nginx:alpine",
        name="cts_server",
        detach=True, 
        network=NETWORK_NAME,
        cpuset_cpus="0", 
        mem_limit="1g",
        sysctls=sysctls,
        volumes={
            DATA_BIN: {'bind': '/usr/share/nginx/html/data.bin', 'mode': 'ro'},
            NGINX_CONF: {'bind': '/etc/nginx/nginx.conf', 'mode': 'ro'}
        }
    )

def configure_network(server, params):
    # Ensure TC is installed (idempotent check)
    exit_code, _ = server.exec_run("which tc")
    if exit_code != 0:
        exit_code, _ = server.exec_run("apk add --no-cache iproute2")
        if exit_code != 0:
            logging.warning("TC install failed, retrying...")
            time.sleep(2)
            server.exec_run("apk add --no-cache iproute2")

    # Clean existing rules first
    server.exec_run("tc qdisc del dev eth0 root", check=False)
    
    # Apply new rules
    cmds = [
        f"tc qdisc add dev eth0 root handle 1: tbf rate {params['bw']}mbit burst 32kbit limit 100mb",
        f"tc qdisc add dev eth0 parent 1:1 handle 10: netem delay {params['delay']}ms loss {params['loss']}%"
    ]
    for cmd in cmds:
        res = server.exec_run(cmd)
        if res.exit_code != 0:
            logging.error(f"TC Error: {res.output.decode()}")

def run_trial(net, config, is_warmup=False):
    cpu_quota = int(float(config['cpu_limit']) * 100000)
    cpuset = "1" if float(config['cpu_limit']) <= 1.0 else "1,2"
    
    size = 10 if is_warmup else config['file_size']
    
    volumes = {CLIENT_SCRIPT: {'bind': '/app/run.py', 'mode': 'ro'}}
    cmd = f"python /app/run.py --url http://cts_server/data.bin --threads {config['threads']} --size {size} --buffer {config['chunk']}"

    try:
        container = client.containers.run(
            "python:3.9-slim",
            name="cts_client", command=cmd,
            detach=True, volumes=volumes, network=NETWORK_NAME,
            cpuset_cpus=cpuset,
            cpu_quota=cpu_quota, cpu_period=100000,
            mem_limit="512m",
            working_dir="/app"
        )
        
        # Timeout protection
        try:
            result = container.wait(timeout=600) 
        except Exception:
            logging.error("Client container timed out (hung)!")
            try: container.kill()
            except: pass
            container.remove()
            return None

        logs = container.logs().decode('utf-8')
        container.remove()
        
        for line in reversed(logs.strip().split('\n')):
            if line.startswith('{'):
                return json.loads(line)
    except Exception as e:
        logging.error(f"Trial execution exception: {e}")
    return None

def generate_configs():
    configs = []
    run_uuid = str(uuid.uuid4())[:8]
    
    for r in range(REPEAT_COUNT):
        for scene, params in SCENARIOS.items():
            for quota in CPU_QUOTAS:
                for t in THREADS:
                    for c in CHUNKS:
                        for size in FILE_SIZES:
                            # Pruning Rules
                            if size <= 10 and t >= 8: continue
                            if c > size: continue

                            configs.append({
                                "run_id": f"{run_uuid}-{len(configs)}",
                                "scene": scene, "net": params, 
                                "cpu_limit": quota, "threads": t, "chunk": c,
                                "file_size": size
                            })
    return configs

def write_batch_result(batch_idx, row):
    """Write to a batch-specific CSV file"""
    filename = f"{DATA_FILE_PREFIX}_{batch_idx}.csv"
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['RunID', 'Scenario', 'FileSize_MB', 'CPU_Quota', 'Threads', 'Chunk', 
                             'Throughput_Mbps', 'CPU_Cores', 'Throttle_Ratio', 'RTT_ms', 'Duration'])
        writer.writerow(row)

def merge_csv_files():
    """Merge all batch CSVs into one final file"""
    logging.info("🔄 Merging batch CSV files...")
    all_files = glob.glob(f"{DATA_FILE_PREFIX}_*.csv")
    all_files.sort()
    
    with open(FINAL_DATA_FILE, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['RunID', 'Scenario', 'FileSize_MB', 'CPU_Quota', 'Threads', 'Chunk', 
                         'Throughput_Mbps', 'CPU_Cores', 'Throttle_Ratio', 'RTT_ms', 'Duration'])
        
        for filename in all_files:
            with open(filename, 'r') as infile:
                reader = csv.reader(infile)
                header = next(reader, None) # Skip header
                for row in reader:
                    writer.writerow(row)
    logging.info(f"✅ Merged data saved to {FINAL_DATA_FILE}")

def cleanup_legacy():
    logging.info("🧹 Cleaning up legacy containers...")
    try: client.containers.get("cts_server").remove(force=True)
    except: pass
    try: client.containers.get("cts_client").remove(force=True)
    except: pass
    try: client.networks.get(NETWORK_NAME).remove()
    except: pass

def main():
    setup_host()
    cleanup_legacy()
    
    configs = generate_configs()
    random.shuffle(configs)
    
    total_trials = len(configs)
    total_batches = (total_trials + BATCH_SIZE - 1) // BATCH_SIZE
    
    logging.info(f"Metadata: {get_system_metadata()}")
    logging.info(f"🚀 Total Trials: {total_trials} | Batch Size: {BATCH_SIZE} | Total Batches: {total_batches}")

    # ==========================
    # 🔄 Batch Execution Loop
    # ==========================
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total_trials)
        current_batch = configs[start_idx:end_idx]
        
        logging.info(f"=== Starting Batch {batch_idx+1}/{total_batches} (Trials {start_idx+1}-{end_idx}) ===")
        
        server = None
        net = None
        try:
            # 1. Infrastructure Setup
            try:
                net = client.networks.get(NETWORK_NAME)
                net.remove()
            except: pass
            net = client.networks.create(NETWORK_NAME, driver="bridge")
            
            server = start_nginx_server(net)
            time.sleep(5) 
            
            # 2. Run Trials
            for idx, cfg in enumerate(current_batch):
                global_idx = start_idx + idx + 1
                logging.info(f"[{global_idx}/{total_trials}] {cfg['scene']} Size:{cfg['file_size']}MB Q:{cfg['cpu_limit']} T:{cfg['threads']}")
                
                success = False
                
                # 3. Retry Logic
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        configure_network(server, cfg['net'])
                        run_trial(net, cfg, is_warmup=True)
                        data = run_trial(net, cfg, is_warmup=False)
                        
                        if data:
                            write_batch_result(batch_idx, [
                                cfg['run_id'], cfg['scene'], cfg['file_size'], cfg['cpu_limit'], cfg['threads'], cfg['chunk'],
                                data['throughput_mbps'], data['cpu_cores_used'], 
                                data['cpu_throttle_ratio'], data['rtt_ms'], data['duration']
                            ])
                            success = True
                            break
                        else:
                            raise ValueError("Data returned None")
                            
                    except Exception as e:
                        if attempt < MAX_RETRIES:
                            wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                            logging.warning(f"   ⚠️ Fail (Attempt {attempt+1}): {e}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            
                            if attempt == MAX_RETRIES - 1:
                                logging.warning("   ⚠️ Critical: Restarting Server...")
                                try:
                                    server.restart()
                                    time.sleep(5)
                                    # [Fix] Ensure tools are re-installed and net configured
                                    server.exec_run("apk add --no-cache iproute2") 
                                    configure_network(server, cfg['net'])
                                except: pass
                        else:
                            logging.error(f"   ❌ Final Failure for {cfg['run_id']}")

                if not success:
                    write_batch_result(batch_idx, [
                        cfg['run_id'], cfg['scene'], cfg['file_size'], cfg['cpu_limit'], cfg['threads'], cfg['chunk'],
                        -1, -1, -1, -1, -1
                    ])

        except Exception as e:
            logging.critical(f"🔥 Batch {batch_idx+1} Infra Failure: {e}")
        
        finally:
            logging.info(f"🧹 Cleaning up Batch {batch_idx+1}...")
            if server:
                try: server.stop(); server.remove()
                except: pass
            if net:
                try: net.remove()
                except: pass
            
            # [Optimization] Extended Cooldown for TIME_WAIT
            logging.info(f"❄️ Cooldown for {BATCH_COOLDOWN}s...")
            time.sleep(BATCH_COOLDOWN)

    # Merge results at the end
    merge_csv_files()
    logging.info("🎉 All Batches Completed.")

if __name__ == "__main__":
    main()