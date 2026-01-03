"""
流式压缩代理模块
功能：实现实时流式压缩，支持多种压缩算法
输入：原始数据流
输出：压缩后的数据流
"""

import gzip
import io
import json
from typing import Generator, Tuple, Dict, Any
import zstandard as zstd
import lz4.frame
import brotli
from .compression_interface import CompressionInterface


class StreamingProxy:
    """流式压缩代理"""
    
    def __init__(self):
        """初始化流式代理"""
        self.compression_methods = {
            'gzip': GzipCompression(),
            'zstd': ZstdCompression(),
            'lz4': Lz4Compression(),
            'brotli': BrotliCompression(),
            'none': NoCompression()
        }
    
    def compress_stream(self, 
                      data_stream: Generator[bytes, None, None], 
                      method: str, 
                      client_profile: Dict[str, Any] = None) -> Generator[bytes, None, None]:
        """
        流式压缩数据
        
        Args:
            data_stream: 原始数据流
            method: 压缩方法
            client_profile: 客户端画像（用于自适应块大小）
            
        Yields:
            压缩后的数据块
        """
        if method not in self.compression_methods:
            raise ValueError(f"不支持的压缩方法: {method}")
        
        # 根据客户端画像调整块大小
        chunk_size = self._determine_chunk_size(client_profile)
        
        compressor = self.compression_methods[method]
        
        # 初始化压缩器
        compressor.init_compression()
        
        # 分块处理数据流
        for chunk in self._chunk_data(data_stream, chunk_size):
            compressed_chunk = compressor.compress_chunk(chunk)
            if compressed_chunk:
                yield compressed_chunk
        
        # 获取最终压缩数据
        final_chunk = compressor.finalize_compression()
        if final_chunk:
            yield final_chunk
    
    def _determine_chunk_size(self, client_profile: Dict[str, Any]) -> int:
        """
        根据客户端画像确定块大小（自适应块大小）
        
        Args:
            client_profile: 客户端画像
            
        Returns:
            块大小（字节）
        """
        if not client_profile:
            # 默认块大小
            return 256 * 1024  # 256KB
        
        # 根据带宽确定块大小
        bandwidth_mbps = client_profile.get('bandwidth_mbps', 100)
        
        if bandwidth_mbps >= 500:  # 高带宽
            return 1024 * 1024  # 1MB
        elif bandwidth_mbps >= 100:  # 中等带宽
            return 512 * 1024  # 512KB
        elif bandwidth_mbps >= 50:  # 低带宽
            return 256 * 1024  # 256KB
        else:  # 极低带宽
            return 64 * 1024  # 64KB
    
    def _chunk_data(self, data_stream: Generator[bytes, None, None], chunk_size: int) -> Generator[bytes, None, None]:
        """
        将数据流分块
        
        Args:
            data_stream: 数据流
            chunk_size: 块大小
            
        Yields:
            数据块
        """
        buffer = b""
        
        for data in data_stream:
            buffer += data
            
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                yield chunk
        
        # 返回剩余数据
        if buffer:
            yield buffer


class GzipCompression(CompressionInterface):
    """Gzip压缩实现"""
    
    def __init__(self, level: int = 6):
        self.level = level
        self.compressor = None
    
    def init_compression(self):
        """初始化压缩器"""
        self.compressor = gzip.GzipFile(fileobj=io.BytesIO(), mode='wb', compresslevel=self.level)
        self.buffer = io.BytesIO()
        # 重写write方法以支持流式压缩
        self.compressor.fileobj = self.buffer
    
    def compress_chunk(self, chunk: bytes) -> bytes:
        """压缩数据块"""
        if self.compressor:
            # 将数据写入压缩器
            self.buffer.write(chunk)
            # 获取压缩后的数据
            compressed_data = self.buffer.getvalue()
            self.buffer.seek(0)
            self.buffer.truncate()
            return compressed_data
        return b""
    
    def finalize_compression(self) -> bytes:
        """完成压缩"""
        if self.compressor:
            # 完成压缩
            self.compressor.close()
            compressed_data = self.buffer.getvalue()
            return compressed_data
        return b""


class ZstdCompression(CompressionInterface):
    """Zstd压缩实现"""
    
    def __init__(self, level: int = 3):
        self.level = level
        self.compressor = None
    
    def init_compression(self):
        """初始化压缩器"""
        self.compressor = zstd.ZstdCompressor(level=self.level).compressobj()
        self.output_size = 1024 * 1024  # 1MB
    
    def compress_chunk(self, chunk: bytes) -> bytes:
        """压缩数据块"""
        if self.compressor:
            return self.compressor.compress(chunk)
        return b""
    
    def finalize_compression(self) -> bytes:
        """完成压缩"""
        if self.compressor:
            return self.compressor.flush()
        return b""


class Lz4Compression(CompressionInterface):
    """LZ4压缩实现"""
    
    def __init__(self, mode: str = 'fast'):
        self.mode = mode
        self.compressor = None
        self.params = {}
    
    def init_compression(self):
        """初始化压缩器"""
        if self.mode == 'fast':
            self.params = {'compression_level': 1}
        elif self.mode == 'balanced':
            self.params = {'compression_level': 6}
        elif self.mode == 'high':
            self.params = {'compression_level': 12}
        
        self.compressor = lz4.frame.LZ4FrameCompressor()
    
    def compress_chunk(self, chunk: bytes) -> bytes:
        """压缩数据块"""
        if self.compressor:
            return self.compressor.compress(chunk)
        return b""
    
    def finalize_compression(self) -> bytes:
        """完成压缩"""
        if self.compressor:
            return self.compressor.flush()
        return b""


class BrotliCompression(CompressionInterface):
    """Brotli压缩实现"""
    
    def __init__(self, level: int = 6):
        self.level = level
        self.compressor = None
    
    def init_compression(self):
        """初始化压缩器"""
        self.compressor = brotli.Compressor(quality=self.level)
    
    def compress_chunk(self, chunk: bytes) -> bytes:
        """压缩数据块"""
        if self.compressor:
            return self.compressor.process(chunk)
        return b""
    
    def finalize_compression(self) -> bytes:
        """完成压缩"""
        if self.compressor:
            return self.compressor.finish()
        return b""


class NoCompression(CompressionInterface):
    """无压缩实现（直接传递数据）"""
    
    def init_compression(self):
        """初始化压缩器"""
        pass
    
    def compress_chunk(self, chunk: bytes) -> bytes:
        """压缩数据块（直接返回）"""
        return chunk
    
    def finalize_compression(self) -> bytes:
        """完成压缩"""
        return b""


def main():
    """主函数，用于测试流式压缩功能"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python streaming_proxy.py <压缩方法> [客户端带宽(Mbps)]")
        print("示例: python streaming_proxy.py gzip 100")
        return
    
    method = sys.argv[1]
    
    # 创建客户端画像
    client_profile = {
        'bandwidth_mbps': int(sys.argv[2]) if len(sys.argv) > 2 else 100
    }
    
    print(f"测试压缩方法: {method}")
    print(f"客户端带宽: {client_profile['bandwidth_mbps']} Mbps")
    
    # 模拟数据流
    def data_stream():
        for i in range(10):
            yield f"这是第{i}个数据块，用于测试流式压缩功能。" * 100
    
    # 创建流式代理
    proxy = StreamingProxy()
    
    # 测试流式压缩
    compressed_chunks = []
    for chunk in proxy.compress_stream(data_stream(), method, client_profile):
        compressed_chunks.append(chunk)
    
    original_size = sum(len(f"这是第{i}个数据块，用于测试流式压缩功能。" * 100) for i in range(10))
    compressed_size = sum(len(chunk) for chunk in compressed_chunks)
    
    print(f"原始大小: {original_size} bytes")
    print(f"压缩后大小: {compressed_size} bytes")
    print(f"压缩比: {compressed_size/original_size:.2%}")
    print(f"使用块大小: {proxy._determine_chunk_size(client_profile)} bytes")


if __name__ == "__main__":
    main()