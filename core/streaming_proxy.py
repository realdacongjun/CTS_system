"""
流式转码代理（Streaming Transcoding Proxy）
目的：实现边压缩边传输，减少首字节延迟
"""

import io
import threading
import time
import zlib
from typing import Generator, Optional, Callable
from abc import ABC, abstractmethod

try:
    import zstandard as zstd
except ImportError:
    zstd = None

try:
    import lz4.frame
except ImportError:
    lz4 = None

class CompressionStream(ABC):
    """压缩流抽象基类"""
    
    def __init__(self):
        self._finished = False
    
    @abstractmethod
    def compress_chunk(self, data_chunk: bytes) -> bytes:
        """压缩数据块"""
        pass
    
    @abstractmethod
    def finalize(self) -> bytes:
        """完成压缩并返回剩余数据"""
        pass


class GzipStream(CompressionStream):
    """Gzip压缩流"""
    
    def __init__(self, level: int = 6):
        super().__init__()
        self.compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    
    def compress_chunk(self, data_chunk: bytes) -> bytes:
        return self.compressor.compress(data_chunk)
    
    def finalize(self) -> bytes:
        if not self._finished:
            self._finished = True
            return self.compressor.flush()
        return b""


class ZstdStream(CompressionStream):
    """Zstd压缩流"""
    
    def __init__(self, level: int = 3):
        super().__init__()
        if zstd is None:
            raise ImportError("zstandard package is required for Zstd compression")
        self.cctx = zstd.ZstdCompressor(level=level).compressor_ctx()
    
    def compress_chunk(self, data_chunk: bytes) -> bytes:
        return self.cctx.compress(data_chunk)
    
    def finalize(self) -> bytes:
        if not self._finished:
            self._finished = True
            return self.cctx.flush()
        return b""


class Lz4Stream(CompressionStream):
    """LZ4压缩流 - 使用一次性压缩方法"""
    
    def __init__(self):
        super().__init__()
        if lz4 is None:
            raise ImportError("lz4 package is required for LZ4 compression")
        # 为了模拟流式处理，我们需要缓存数据直到finalize被调用
        self._buffer = bytearray()
    
    def compress_chunk(self, data_chunk: bytes) -> bytes:
        if self._finished:
            return b""
        
        # 将数据添加到缓冲区，实际压缩在finalize中进行
        self._buffer.extend(data_chunk)
        # 返回空字节串，因为我们正在缓存数据直到finalize
        return b""
    
    def finalize(self) -> bytes:
        if not self._finished:
            self._finished = True
            # 对整个缓冲区进行一次性压缩
            if self._buffer:
                compressed = lz4.frame.compress(self._buffer)
                self._buffer = bytearray()
                return compressed
        return b""


class StreamingTranscoder:
    """流式转码器"""
    
    def __init__(self, compression_method: str):
        """
        初始化流式转码器
        
        Args:
            compression_method: 压缩方法，格式如 "gzip-6", "zstd-3", "lz4-fast"
        """
        self.compression_method = compression_method
        self._setup_compressor()
    
    def _setup_compressor(self):
        """设置压缩器"""
        algo = self.compression_method.split('-')[0]
        
        if algo == "gzip":
            level = 6
            if "-" in self.compression_method:
                try:
                    level = int(self.compression_method.split("-")[1])
                except ValueError:
                    pass
            self.compressor = GzipStream(level)
        elif algo == "zstd":
            level = 3
            if "-" in self.compression_method:
                try:
                    level = int(self.compression_method.split("-")[1])
                except ValueError:
                    pass
            self.compressor = ZstdStream(level)
        elif algo == "lz4":
            self.compressor = Lz4Stream()
        else:
            raise ValueError(f"Unsupported compression algorithm: {algo}")
    
    def transcode_stream(self, input_stream: io.RawIOBase) -> Generator[bytes, None, None]:
        """
        流式转码
        
        Args:
            input_stream: 输入数据流
            
        Yields:
            压缩后的数据块
        """
        chunk_size = 64 * 1024  # 64KB chunks
        
        while True:
            chunk = input_stream.read(chunk_size)
            if not chunk:
                break
            
            compressed_chunk = self.compressor.compress_chunk(chunk)
            if compressed_chunk:
                yield compressed_chunk
        
        # 发送最终数据
        final_data = self.compressor.finalize()
        if final_data:
            yield final_data


class StreamingProxy:
    """流式代理"""
    
    def __init__(self):
        self.active_streams = {}
        self.lock = threading.Lock()
    
    def process_stream(self, 
                      input_stream: io.RawIOBase, 
                      compression_method: str,
                      on_progress: Optional[Callable[[int, int], None]] = None) -> Generator[bytes, None, None]:
        """
        处理流式数据
        
        Args:
            input_stream: 输入数据流
            compression_method: 压缩方法
            on_progress: 进度回调函数 (processed_bytes, total_bytes)
            
        Yields:
            压缩后的数据块
        """
        transcoder = StreamingTranscoder(compression_method)
        
        # 记录开始时间
        start_time = time.time()
        processed_bytes = 0
        
        for chunk in transcoder.transcode_stream(input_stream):
            processed_bytes += len(chunk)
            yield chunk
            
            # 如果提供了进度回调，调用它
            if on_progress:
                on_progress(processed_bytes, processed_bytes)  # 注意：这里total_bytes是近似值
        
        # 记录处理时间
        elapsed_time = time.time() - start_time
        print(f"Streaming compression completed in {elapsed_time:.2f}s for {processed_bytes} bytes")
    
    def create_proxy_response(self, 
                            input_stream: io.RawIOBase, 
                            compression_method: str,
                            content_type: str = "application/octet-stream") -> dict:
        """
        创建代理响应
        
        Args:
            input_stream: 输入数据流
            compression_method: 压缩方法
            content_type: 内容类型
            
        Returns:
            响应字典
        """
        def generate():
            for chunk in self.process_stream(input_stream, compression_method):
                yield chunk
        
        response = {
            "stream": generate(),
            "headers": {
                "Content-Type": content_type,
                "Content-Encoding": self._get_content_encoding(compression_method),
                "Transfer-Encoding": "chunked"
            }
        }
        
        return response
    
    def _get_content_encoding(self, compression_method: str) -> str:
        """获取内容编码类型"""
        algo = compression_method.split('-')[0]
        mapping = {
            "gzip": "gzip",
            "zstd": "zstd",
            "lz4": "lz4"
        }
        return mapping.get(algo, "identity")


# 示例用法
if __name__ == "__main__":
    import tempfile
    
    # 创建一个临时文件用于测试
    with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
        # 写入一些测试数据
        test_data = b"Hello, this is test data for streaming compression. " * 1000
        temp_file.write(test_data)
        temp_file_path = temp_file.name
    
    # 打开文件并进行流式压缩
    with open(temp_file_path, 'rb') as input_file:
        proxy = StreamingProxy()
        
        print("Testing streaming compression...")
        start_time = time.time()
        
        total_output_size = 0
        for chunk in proxy.process_stream(input_file, "gzip-6"):
            total_output_size += len(chunk)
            # 模拟处理输出
            pass
        
        elapsed_time = time.time() - start_time
        print(f"Compression completed in {elapsed_time:.2f}s")
        print(f"Output size: {total_output_size} bytes")
        print(f"Compression ratio: {total_output_size / len(test_data):.2f}")