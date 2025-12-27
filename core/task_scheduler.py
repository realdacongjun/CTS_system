"""
异步任务调度器（Task Scheduler）
目的：管理压缩任务的队列和状态
"""

import enum
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Callable, Any, Optional, List
from dataclasses import dataclass, field
from queue import Queue, Empty
import hashlib


class TaskStatus(enum.Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranscodeTask:
    """转码任务数据类"""
    task_id: str
    image_name: str
    layer_digest: str
    compression_method: str
    layer_data: bytes
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0


class TaskQueue:
    """任务队列管理器"""
    
    def __init__(self, max_workers: int = 4):
        self.task_queue = Queue()
        self.tasks: Dict[str, TranscodeTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
    
    def add_task(self, task: TranscodeTask) -> str:
        """添加任务到队列"""
        with self.lock:
            self.tasks[task.task_id] = task
            self.task_queue.put(task.task_id)
            return task.task_id
    
    def get_task(self, task_id: str) -> Optional[TranscodeTask]:
        """获取任务状态"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[TranscodeTask]:
        """获取指定状态的任务列表"""
        with self.lock:
            return [task for task in self.tasks.values() if task.status == status]
    
    def _process_queue(self):
        """处理队列中的任务"""
        while self.running:
            try:
                task_id = self.task_queue.get(timeout=1)
                with self.lock:
                    task = self.tasks.get(task_id)
                
                if task:
                    self._execute_task(task)
                
                self.task_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"Error processing task queue: {e}")
    
    def _execute_task(self, task: TranscodeTask):
        """执行单个任务"""
        try:
            # 更新任务状态
            with self.lock:
                task.status = TaskStatus.PROCESSING
            
            print(f"Starting task {task.task_id} for {task.image_name}:{task.layer_digest} with {task.compression_method}")
            
            # 这里调用实际的压缩逻辑
            # 为了演示，我们使用一个模拟的压缩函数
            compressed_data = self._compress_layer_data(
                task.layer_data, 
                task.compression_method
            )
            
            # 模拟进度更新
            task.progress = 1.0
            
            # 更新任务结果
            with self.lock:
                task.status = TaskStatus.COMPLETED
                task.result = compressed_data
                task.progress = 1.0
            
            # 执行回调（如果存在）
            if task.callback:
                try:
                    task.callback(task)
                except Exception as e:
                    print(f"Callback error for task {task.task_id}: {e}")
        
        except Exception as e:
            print(f"Task {task.task_id} failed: {e}")
            with self.lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)
    
    def _compress_layer_data(self, layer_data: bytes, compression_method: str) -> bytes:
        """压缩层数据 - 这里是模拟实现"""
        # 这里应该调用实际的压缩算法
        # 为了演示，我们简单地返回原始数据
        # 在实际实现中，这里会根据compression_method选择对应的压缩算法
        print(f"Compressing {len(layer_data)} bytes with {compression_method}")
        
        # 模拟压缩过程（实际实现中会使用zlib、zstd等库）
        
        # 这里是模拟压缩，实际实现中需要根据compression_method选择压缩算法
        if compression_method.startswith("gzip"):
            import zlib
            level = int(compression_method.split("-")[1]) if "-" in compression_method else 6
            return zlib.compress(layer_data, level)
        elif compression_method.startswith("zstd"):
            try:
                import zstandard as zstd
                level = int(compression_method.split("-")[1]) if "-" in compression_method else 3
                cctx = zstd.ZstdCompressor(level=level)
                return cctx.compress(layer_data)
            except ImportError:
                # 如果zstd不可用，使用gzip作为回退
                return zlib.compress(layer_data, 6)
        elif compression_method.startswith("lz4"):
            try:
                import lz4.frame
                # 使用一次性压缩，避免flush问题
                return lz4.frame.compress(layer_data)
            except ImportError:
                # 如果lz4不可用，使用gzip作为回退
                return zlib.compress(layer_data, 6)
        else:
            # 默认使用gzip
            return zlib.compress(layer_data, 6)
    
    def shutdown(self):
        """关闭任务队列"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.worker_thread.join()


class TaskScheduler:
    """任务调度器主类"""
    
    def __init__(self, max_workers: int = 4):
        self.task_queue = TaskQueue(max_workers)
        self.active_compressions = {}  # 记录正在运行的压缩任务，避免重复
        self.lock = threading.Lock()
    
    def submit_transcode_task(self, 
                            image_name: str, 
                            layer_digest: str, 
                            compression_method: str, 
                            layer_data: bytes,
                            callback: Optional[Callable] = None) -> str:
        """
        提交转码任务
        
        Args:
            image_name: 镜像名称
            layer_digest: 层摘要
            compression_method: 压缩方法
            layer_data: 层数据
            callback: 完成回调函数
            
        Returns:
            任务ID
        """
        # 创建唯一任务ID，避免重复任务
        task_key = f"{image_name}:{layer_digest}:{compression_method}"
        
        with self.lock:
            if task_key in self.active_compressions:
                # 如果已经有相同任务在运行，返回现有任务ID
                return self.active_compressions[task_key]
        
        # 创建新任务
        task = TranscodeTask(
            task_id=str(uuid.uuid4()),
            image_name=image_name,
            layer_digest=layer_digest,
            compression_method=compression_method,
            layer_data=layer_data,
            callback=callback
        )
        
        task_id = self.task_queue.add_task(task)
        
        with self.lock:
            self.active_compressions[task_key] = task_id
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[TranscodeTask]:
        """获取任务状态"""
        return self.task_queue.get_task(task_id)
    
    def is_task_running(self, image_name: str, layer_digest: str, compression_method: str) -> bool:
        """检查是否有相同任务正在运行"""
        task_key = f"{image_name}:{layer_digest}:{compression_method}"
        with self.lock:
            return task_key in self.active_compressions
    
    def cleanup_completed_tasks(self):
        """清理已完成的任务"""
        completed_tasks = self.task_queue.get_tasks_by_status(TaskStatus.COMPLETED)
        
        with self.lock:
            for task in completed_tasks:
                # 检查是否在活跃任务中
                task_key = f"{task.image_name}:{task.layer_digest}:{task.compression_method}"
                if task_key in self.active_compressions:
                    del self.active_compressions[task_key]
    
    def shutdown(self):
        """关闭调度器"""
        self.task_queue.shutdown()


# 全局任务调度器实例
task_scheduler = TaskScheduler(max_workers=4)


def main():
    """测试任务调度器"""
    print("Testing Task Scheduler...")
    
    # 创建测试数据
    test_data = b"Test layer data for compression. " * 1000
    
    def task_callback(task: TranscodeTask):
        print(f"Task {task.task_id} completed with {len(task.result)} bytes result")
    
    # 提交一个任务
    task_id = task_scheduler.submit_transcode_task(
        "test_image:latest",
        "sha256:test_digest",
        "gzip-6",
        test_data,
        task_callback
    )
    
    print(f"Submitted task: {task_id}")
    
    # 查询任务状态
    import time
    for i in range(10):  # 最多等待10秒
        task = task_scheduler.get_task_status(task_id)
        if task:
            print(f"Task {task_id} status: {task.status.value}, progress: {task.progress:.2f}")
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
        time.sleep(0.5)
    
    # 关闭调度器
    task_scheduler.shutdown()


if __name__ == "__main__":
    main()