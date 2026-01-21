import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import csv

class RealDownloader:
    def __init__(self, url, file_size, output_path):
        self.url = url
        self.total_size = file_size
        self.output_path = output_path
        self.lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.downloaded_bytes = 0
        
        # 初始化空文件
        with open(self.output_path, 'wb') as f:
            f.seek(self.total_size - 1)
            f.write(b'\0')

    # 注意：移除了 log_file 参数，因为日志是在主线程写的，不需要传进去
    def _fetch_chunk(self, start, end, chunk_id):
        """下载单个分片的工作函数"""
        headers = {'Range': f'bytes={start}-{end}'}
        try:
            t0 = time.time()
            resp = requests.get(self.url, headers=headers, timeout=10)
            
            if resp.status_code == 206:
                data = resp.content
                duration = time.time() - t0
                
                # 写入文件 (加锁防止冲突)
                with self.lock:
                    with open(self.output_path, 'r+b') as f:
                        f.seek(start)
                        f.write(data)
                
                # 更新下载进度
                with self.progress_lock:
                    self.downloaded_bytes += len(data)
                
                return len(data), duration, 'SUCCESS'
            else:
                return 0, 0, 'FAILED'
        except Exception as e:
            return 0, 0, 'TIMEOUT'

    def download_with_chunks(self, initial_chunk_size, concurrency, correction_layer=None, log_file=None):
        """
        执行分片下载
        :param correction_layer: 传入 CAGSCorrectionLayer 实例
        :param log_file: CSV日志文件路径
        """
        cursor = 0
        self.downloaded_bytes = 0
        start_time = time.time()
        
        # 初始化CSV日志文件 (写表头)
        if log_file:
            # 确保目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_exists = os.path.isfile(log_file)
            with open(log_file, 'w', newline='') as f: # 使用 'w' 覆盖旧日志，保证每次实验干净
                writer = csv.writer(f)
                writer.writerow(["Time_Offset_s", "Chunk_Size_KB", "Instant_Speed_MBs", "Status"])
        
        print(f"📥 开始下载 | 大小: {self.total_size/(1024*1024):.2f}MB | 并发: {concurrency} | 初始块: {initial_chunk_size/1024:.0f}KB")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            
            while cursor < self.total_size or futures:
                # 1. 提交新任务
                while cursor < self.total_size and len(futures) < concurrency:
                    # 动态获取当前切片大小
                    if correction_layer:
                        current_chunk_size = correction_layer.current_size
                    else:
                        current_chunk_size = initial_chunk_size # 静态模式
                    
                    end = min(cursor + current_chunk_size - 1, self.total_size - 1)
                    
                    # 提交任务
                    future = executor.submit(self._fetch_chunk, cursor, end, 0)
                    # 将 chunk_size 存入 futures 字典，方便后续记录日志
                    futures[future] = (cursor, end, current_chunk_size)
                    cursor += current_chunk_size
                
                # 2. 处理已完成的任务
                completed_futures = []
                for future in as_completed(list(futures.keys()), timeout=0.05): # 缩短 timeout 提高响应速度
                    size, duration, status = future.result()
                    start_pos, end_pos, chunk_size_used = futures[future]
                    completed_futures.append(future)
                    
                    # --- 微观数据记录核心区域 ---
                    speed = (size/1024/1024) / duration if duration > 0 else 0
                    time_offset = time.time() - start_time
                    
                    if log_file:
                        with open(log_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                f"{time_offset:.2f}",
                                f"{chunk_size_used/1024:.0f}", # 记录 KB
                                f"{speed:.2f}",
                                status
                            ])
                    # ---------------------------

                    if status == 'SUCCESS':
                        # 打印进度
                        with self.progress_lock:
                            if self.downloaded_bytes % (5*1024*1024) < size:
                                progress = self.downloaded_bytes / self.total_size * 100
                                print(f"\r🚀 进度: {progress:.1f}% | 速度: {speed:.2f} MB/s | 块: {chunk_size_used/1024:.0f}KB", end="")
                    
                    # 3. 反馈给 AIMD 修正层 (闭环控制)
                    if correction_layer:
                        correction_layer.feedback(status, rtt_ms=duration*1000)
                
                # 移除已完成的任务
                for future in completed_futures:
                    del futures[future]
                
                # 避免 CPU 空转
                if not completed_futures:
                    time.sleep(0.01)

        total_time = time.time() - start_time
        avg_speed = (self.total_size / (1024*1024)) / total_time
        print(f"\n✅ 下载完成 | 耗时: {total_time:.2f}s | 平均速度: {avg_speed:.2f} MB/s")
        
        # 验证文件大小
        if os.path.exists(self.output_path):
            actual_size = os.path.getsize(self.output_path)
            if actual_size == self.total_size:
                return True
            else:
                print(f"❌ 大小不匹配: {actual_size} != {self.total_size}")
                return False
        return False