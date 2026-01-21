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

    def _fetch_chunk(self, start, end, chunk_id, log_file):
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
                
                # 记录微观数据
                with open(log_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    inst_speed = (len(data)/1024/1024) / duration if duration > 0 else 0
                    writer.writerow([
                        time.time(),  # 时间戳
                        len(data)/1024,  # 当前分片大小KB
                        f"{inst_speed:.2f}",  # 瞬时速度MB/s
                        'SUCCESS'  # 状态
                    ])
                
                return len(data), duration, 'SUCCESS'
            else:
                # 记录失败情况
                with open(log_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        time.time(),  # 时间戳
                        (end-start+1)/1024,  # 分片大小KB
                        0,  # 瞬时速度
                        'FAILED'  # 状态
                    ])
                return 0, 0, 'FAILED'
        except Exception as e:
            # 记录超时情况
            with open(log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    time.time(),  # 时间戳
                    (end-start+1)/1024,  # 分片大小KB
                    0,  # 瞬时速度
                    'TIMEOUT'  # 状态
                ])
            return 0, 0, 'TIMEOUT'

    def download_with_chunks(self, initial_chunk_size, concurrency, correction_layer=None, log_file='microscopic_log.csv'):
        """
        执行分片下载
        :param correction_layer: 传入 CAGSCorrectionLayer 实例，如果为 None 则不调整
        :param log_file: 微观数据记录文件路径
        """
        cursor = 0
        self.downloaded_bytes = 0
        start_time = time.time()
        
        # 初始化日志文件
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Chunk_Size_KB', 'Speed_MB_s', 'Status'])  # 写表头

        print(f"📥 开始下载 | 大小: {self.total_size/(1024*1024):.2f}MB | 并发: {concurrency} | 初始块: {initial_chunk_size/(1024*1024):.2f}MB")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            active_count = 0
            
            while cursor < self.total_size or futures:
                # 1. 提交新任务 (如果还有剩余数据且并发未满)
                while cursor < self.total_size and len(futures) < concurrency:
                    # 动态获取当前切片大小
                    if correction_layer:
                        current_chunk_size = correction_layer.current_size
                    else:
                        current_chunk_size = initial_chunk_size # 静态模式
                    
                    end = min(cursor + current_chunk_size - 1, self.total_size - 1)
                    
                    # 记录决策日志
                    log_data.append((time.time()-start_time, current_chunk_size))
                    
                    # 提交
                    future = executor.submit(self._fetch_chunk, cursor, end, 0, log_file)
                    futures[future] = (cursor, end)
                    cursor += current_chunk_size
                
                # 2. 处理已完成的任务
                completed_futures = []
                for future in as_completed(list(futures.keys()), timeout=0.1):
                    size, duration, status = future.result()
                    start_pos, end_pos = futures[future]
                    completed_futures.append(future)
                    
                    if status == 'SUCCESS':
                        # 打印进度 (每 5MB 打印一次，避免刷屏)
                        with self.progress_lock:
                            progress = self.downloaded_bytes / self.total_size * 100
                        speed = (size/1024/1024) / duration if duration > 0 else 0
                        if self.downloaded_bytes % (5*1024*1024) < size:
                            print(f"\r🚀 进度: {progress:.1f}% | 瞬时速度: {speed:.2f} MB/s | 块: {size/1024:.0f}KB", end="")

                    # 3. 反馈给 AIMD 修正层
                    if correction_layer:
                        correction_layer.feedback(status, rtt_ms=duration*1000)
                
                # 移除已完成的任务
                for future in completed_futures:
                    del futures[future]
                
                # 避免 CPU 空转
                time.sleep(0.01)

        total_time = time.time() - start_time
        avg_speed = (self.total_size / (1024*1024)) / total_time
        print(f"\n✅ 下载完成 | 耗时: {total_time:.2f}s | 平均速度: {avg_speed:.2f} MB/s")
        
        # 验证文件大小
        actual_size = os.path.getsize(self.output_path)
        if actual_size == self.total_size:
            print("✅ 文件完整性验证通过!")
            return True
        else:
            print(f"❌ 文件完整性验证失败! 期望: {self.total_size}, 实际: {actual_size}")
            return False