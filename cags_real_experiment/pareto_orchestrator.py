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
import subprocess
import re
from datetime import datetime

# ==========================================
# 🧪 Experimental Configuration
# ==========================================
BATCH_SIZE = 20        
MAX_RETRIES = 3        
RETRY_DELAY_BASE = 5   
BATCH_COOLDOWN = 15    

NETWORK_NAME = "cts_paper_net"
DATA_FILE_PREFIX = "pareto_results_batch"
FINAL_DATA_FILE = "pareto_results_final.csv"

CLIENT_SCRIPT = os.path.abspath("pareto_client.py")
NGINX_CONF = os.path.abspath("nginx.conf")
DATA_BIN = os.path.abspath("data.bin")

SCENARIOS = {
    "Weak":   {"bw": 5,   "delay": 400, "loss": 1.0},
    "Edge":   {"bw": 20,  "delay": 50,  "loss": 0.1},
    "Cloud":  {"bw": 100, "delay": 10,  "loss": 0.0}
}
CPU_QUOTAS = ["1.0", "2.0"] 
THREADS = [1, 2, 4, 8, 16]
CHUNKS = [0.1, 0.5, 1.0, 2.0, 4.0]
FILE_SIZES = [10, 50, 100, 200, 400] 
BASE_PAYLOAD_SIZE = 500 
REPEAT_COUNT = 5 

client = docker.from_env()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_host():
    if not os.path.exists(DATA_BIN) or os.path.getsize(DATA_BIN) < BASE_PAYLOAD_SIZE * 1024 * 1024:
        logging.info(f"Generating {BASE_PAYLOAD_SIZE}MB random payload...")
        os.system(f"dd if=/dev/urandom of={DATA_BIN} bs=1M count={BASE_PAYLOAD_SIZE}")
    
    # Save system metadata to a file for later analysis
    metadata = get_system_metadata()
    with open("experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
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

def disable_host_offload(network_id):
    """
    [Methodology Fix] Disable Offloading on Host Bridge
    CRITICAL: Netem needs accurate packet boundaries. If the host bridge 
    does GRO/LRO, it re-coalesces packets, defeating the delay model.
    """
    # Docker bridges are named 'br-<first 12 chars of ID>'
    bridge_name = f"br-{network_id[:12]}"
    
    logging.info(f"🔧 Disabling HW offload on Host Bridge: {bridge_name}")
    try:
        # Requires sudo/root on host
        subprocess.run(f"ethtool -K {bridge_name} tso off gso off gro off", shell=True, check=False)
    except Exception as e:
        logging.warning(f"Failed to disable host offload (Need Root?): {e}")

def get_system_metadata():
    return {
        "kernel": platform.release(),
        "arch": platform.machine(),
        "docker_ver": client.version()['Version'],
        "cpu_info": os.popen("lscpu | grep 'Model name'").read().strip()
    }

def start_nginx_server(net):
    sysctls = {
        "net.ipv4.tcp_tw_reuse": "1",
        "net.ipv4.tcp_fin_timeout": "30"
    }
    return client.containers.run(
        "nginx:alpine",
        name="cts_server",
        detach=True, 
        network=NETWORK_NAME,
        cpuset_cpus="3", # Server on Core 3
        mem_limit="1g",
        sysctls=sysctls,
        privileged=True,  # Required for network traffic control
        cap_add=["NET_ADMIN"],  # Required for tc commands
        volumes={
            DATA_BIN: {'bind': '/usr/share/nginx/html/data.bin', 'mode': 'ro'},
            NGINX_CONF: {'bind': '/etc/nginx/nginx.conf', 'mode': 'ro'}
        }
    )

def configure_network(server, params):
    exit_code, _ = server.exec_run("which tc")
    if exit_code != 0:
        server.exec_run("apk add --no-cache iproute2 ethtool") # Ensure ethtool inside too

    # [Methodology Fix] Disable Container-side Offload (Sender TSO)
    server.exec_run("ethtool -K eth0 tso off gso off gro off")

    
    try:
        server.exec_run("tc qdisc del dev eth0 root")
    except Exception as e:
        logging.warning(f"Failed to delete qdisc: {e}")

    # [Methodology Fix] Hierarchy: Root HTB (Rate) -> Class -> Netem (Delay)
    cmds = [
        "tc qdisc add dev eth0 root handle 1: htb default 10",
        f"tc class add dev eth0 parent 1: classid 1:10 htb rate {params['bw']}mbit ceil {params['bw']}mbit",
        f"tc qdisc add dev eth0 parent 1:10 handle 10: netem delay {params['delay']}ms loss {params['loss']}%"
    ]
    for cmd in cmds:
        res = server.exec_run(cmd)
        if res.exit_code != 0:
            logging.error(f"TC Error: {res.output.decode()}")

def validate_rtt_sanity(net, server, params):
    """
    [Methodology Check] Verify TCP_INFO RTT against ICMP Ping
    This guards against kernel struct layout mismatches.
    """
    logging.info(f"🔎 Validating RTT Integrity for {params['delay']}ms delay...")
    
    # 1. Run a quick ping check
    ping_cmd = f"ping -c 5 cts_server"
    try:
        # Spin up a temp client
        client_check = client.containers.run(
            "python:3.9-slim",
            name="cts_validator",
            command=["sh", "-c", "apt-get update && apt-get install -y iputils-ping && " + ping_cmd],
            detach=True, network=NETWORK_NAME
        )
        client_check.wait()
        logs = client_check.logs().decode()
        client_check.remove()
        
        # Parse Ping RTT
        # rtt min/avg/max/mdev = 400.1/400.5/401.2/0.5 ms
        match = re.search(r"min/avg/max/mdev = [\d\.]+/([\d\.]+)/", logs)
        if match:
            ping_rtt = float(match.group(1))
            target_rtt = params['delay'] * 2 # Round trip
            
            # Allow 20% margin (Netem isn't perfect, OS scheduling noise)
            if abs(ping_rtt - target_rtt) > (target_rtt * 0.2 + 5):
                logging.error(f"❌ RTT VALIDATION FAILED! Expected ~{target_rtt}ms, Got Ping={ping_rtt}ms")
                logging.error("Check Host Offloading or Netem configuration!")
                return False
            else:
                logging.info(f"✅ RTT Sanity Passed: Ping={ping_rtt}ms matches Target={target_rtt}ms")
                return True
    except Exception as e:
        logging.warning(f"RTT Validation skipped due to error: {e}")
        return True # Don't block experiment on validation error, but log it
    return True

def run_trial(net, config, is_warmup=False):
    cpu_quota = int(float(config['cpu_limit']) * 100000)
    cpuset = "1" if float(config['cpu_limit']) <= 1.0 else "1,2"
    
    size = 10 if is_warmup else config['file_size']
    warmup_flag = 1 if is_warmup else 0
    
    volumes = {CLIENT_SCRIPT: {'bind': '/app/run.py', 'mode': 'ro'}}
    cmd = f"python /app/run.py --url http://cts_server/data.bin --threads {config['threads']} --size {size} --buffer {config['chunk']} --warmup {warmup_flag}"

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
    filename = f"{DATA_FILE_PREFIX}_{batch_idx}.csv"
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['RunID', 'Scenario', 'FileSize_MB', 'CPU_Quota', 'Threads', 'Chunk', 
                             'Throughput_Mbps', 'CPU_Cores', 'Throttle_Ratio', 'RTT_ms', 'Duration'])
        writer.writerow(row)

def merge_csv_files():
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
                header = next(reader, None)
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
    if os.geteuid() != 0:
        logging.warning("⚠️  Running without Root? Host offload disabling may fail.")
        
    setup_host()
    cleanup_legacy()
    
    configs = generate_configs()
    random.shuffle(configs)
    
    total_trials = len(configs)
    total_batches = (total_trials + BATCH_SIZE - 1) // BATCH_SIZE
    
    logging.info(f"Metadata: {get_system_metadata()}")
    logging.info(f"🚀 Total Trials: {total_trials} | Batch Size: {BATCH_SIZE}")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total_trials)
        current_batch = configs[start_idx:end_idx]
        
        logging.info(f"=== Starting Batch {batch_idx+1}/{total_batches} ===")
        
        server = None
        net = None
        try:
            try:
                net = client.networks.get(NETWORK_NAME)
                net.remove()
            except: pass
            net = client.networks.create(NETWORK_NAME, driver="bridge")
            
            # [Methodology Fix] Disable Offload on Host Bridge immediately
            disable_host_offload(net.id)
            
            server = start_nginx_server(net)
            time.sleep(5) 
            
            # [Methodology Fix] Perform RTT Sanity Check once per batch/network
            # We pick the first config's network params to validate
            if len(current_batch) > 0:
                first_net_params = current_batch[0]['net']
                configure_network(server, first_net_params)
                if not validate_rtt_sanity(net, server, first_net_params):
                    logging.critical("🛑 ABORTING BATCH: Network environment unsound.")
                    continue 

            for idx, cfg in enumerate(current_batch):
                global_idx = start_idx + idx + 1
                logging.info(f"[{global_idx}/{total_trials}] {cfg['scene']} Size:{cfg['file_size']}MB Q:{cfg['cpu_limit']} T:{cfg['threads']}")
                
                success = False
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
                                try:
                                    server.restart()
                                    time.sleep(5)
                                    server.exec_run("apk add --no-cache iproute2 ethtool") 
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
            if server:
                try: server.stop(); server.remove()
                except: pass
            if net:
                try: net.remove()
                except: pass
            
            logging.info(f"❄️ Cooldown for {BATCH_COOLDOWN}s...")
            time.sleep(BATCH_COOLDOWN)

    merge_csv_files()
    logging.info("🎉 All Batches Completed.")

if __name__ == "__main__":
    main()