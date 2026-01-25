import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import csv

class RealDownloader:
    """
    真实环境并发下载器 (防崩溃最终版)
    """
    def __init__(self, url, file_size, output_path):
        self.url = url
        self.total_size = file_size
        self.output_path = output_path
        self.lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.downloaded_bytes = 0
        
        # 预分配磁盘空间
        print(f"[Downloader] 正在预分配磁盘空间: {file_size/(1024*1024):.2f} MB")
        try:
            with open(self.output_path, 'wb') as f:
                f.seek(self.total_size - 1)
                f.write(b'\0')
        except Exception as e:
            print(f"[Downloader] ⚠️ 预分配空间失败: {e}")

    def _fetch_chunk(self, start, end, chunk_id, log_file):
        headers = {'Range': f'bytes={start}-{end}'}
        try:
            t0 = time.time()
            # timeout=15 适应极慢的弱网环境 (1.5s RTT)
            resp = requests.get(self.url, headers=headers, timeout=15)
            
            if resp.status_code == 206:
                data = resp.content
                duration = time.time() - t0
                with self.lock:
                    with open(self.output_path, 'r+b') as f:
                        f.seek(start)
                        f.write(data)
                with self.progress_lock:
                    self.downloaded_bytes += len(data)
                self._log_micro_data(log_file, time.time(), len(data), duration, 'SUCCESS')
                return len(data), duration, 'SUCCESS'
            else:
                return 0, 0, 'FAILED'
        except:
            # 任何错误都只记录，不抛出异常
            self._log_micro_data(log_file, time.time(), (end-start+1), 0, 'TIMEOUT')
            return 0, 0, 'TIMEOUT'

    def _log_micro_data(self, log_file, ts, size, duration, status):
        try:
            with open(log_file, 'a', newline='') as csvfile:
                inst_speed = (size/1024/1024) / duration if duration > 0 else 0
                csv.writer(csvfile).writerow([ts, size/1024, f"{inst_speed:.2f}", status])
        except:
            pass

    def download_with_chunks(self, initial_chunk_size, concurrency, correction_layer=None, log_file='microscopic_log.csv'):
        cursor = 0
        self.downloaded_bytes = 0
        start_time = time.time()
        
        with open(log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['Timestamp', 'Chunk_Size_KB', 'Speed_MB_s', 'Status'])

        print(f"📥 开始下载 | 目标: {self.total_size/(1024*1024):.2f}MB | 并发: {concurrency}")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            while cursor < self.total_size or futures:
                # 1. 提交任务
                while cursor < self.total_size and len(futures) < concurrency:
                    current_chunk_size = correction_layer.current_size if correction_layer else initial_chunk_size
                    end = min(cursor + current_chunk_size - 1, self.total_size - 1)
                    future = executor.submit(self._fetch_chunk, cursor, end, 0, log_file)
                    futures[future] = (cursor, end)
                    cursor += current_chunk_size
                
                # 2. 【核心修复】轮询状态 (去掉会导致崩溃的 as_completed)
                done_list = []
                for f in list(futures.keys()):
                    if f.done():
                        done_list.append(f)
                        try:
                            size, duration, status = f.result()
                            if status == 'SUCCESS':
                                progress = self.downloaded_bytes / self.total_size * 100
                                speed = (size/1024/1024) / duration if duration > 0 else 0
                                # 强制刷新显示
                                print(f"\r🚀 进度: {progress:.1f}% | 速度: {speed:.2f} MB/s ", end="", flush=True)
                            if correction_layer:
                                correction_layer.feedback(status, duration*1000)
                        except:
                            pass
                
                for f in done_list:
                    del futures[f]
                
                time.sleep(0.05) # 稍微等待，防止CPU空转

        total_time = time.time() - start_time
        print(f"\n✅ 下载流程结束 | 耗时: {total_time:.2f}s")
        return os.path.getsize(self.output_path) == self.total_size, total_time