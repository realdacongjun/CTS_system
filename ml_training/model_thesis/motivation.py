import os
import time
import gzip
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import sys

# 尝试导入 LZ4 / ZSTD
try:
    import lz4.frame
    import zstandard as zstd
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    print("⚠️ 警告: 未检测到 lz4/zstandard 库。将仅使用 Gzip 和 Python内置库(LZMA) 进行对比演示。")
    print("   强烈建议安装: pip install lz4 zstandard")

# --- 1. 字体与风格 ---
system_name = platform.system()
fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
for f in fonts:
    try:
        plt.rcParams['font.sans-serif'] = [f]
        break
    except:
        continue
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

class RealWorldMotivation:
    def __init__(self):
        self.results = []
        
    def is_binary(self, file_path):
        """简单判断文件是否为二进制"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk: return True
                return False
        except:
            return True

    def scan_real_files(self):
        """扫描系统真实文件"""
        target_files = {'Binary': [], 'Text': []}
        
        # 1. 找二进制文件 (系统命令/DLL)
        bin_dirs = ['/usr/bin', '/bin', r'C:\Windows\System32']
        count = 0
        for d in bin_dirs:
            if os.path.exists(d):
                for f in os.listdir(d):
                    fp = os.path.join(d, f)
                    if os.path.isfile(fp) and 50*1024 < os.path.getsize(fp) < 10*1024*1024: # 50KB - 10MB
                        if self.is_binary(fp):
                            target_files['Binary'].append(fp)
                            count += 1
                    if count > 30: break # 每个类别采30个样
            if count > 0: break
            
        # 2. 找文本文件 (Python源码/日志)
        text_dirs = [os.path.dirname(os.__file__), '/var/log'] 
        count = 0
        for d in text_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for f in files:
                        fp = os.path.join(root, f)
                        if f.endswith('.py') or f.endswith('.log') or f.endswith('.h'):
                            if 20*1024 < os.path.getsize(fp) < 5*1024*1024:
                                target_files['Text'].append(fp)
                                count += 1
                        if count > 30: break
                    if count > 30: break
        
        print(f"📂 数据准备完成: 采集到 {len(target_files['Binary'])} 个真实二进制文件, {len(target_files['Text'])} 个真实文本文件。")
        return target_files

    def run_benchmark(self):
        files_map = self.scan_real_files()
        if not files_map['Binary'] and not files_map['Text']:
            print("❌ 错误: 未能在系统中找到合适的文件进行测试。")
            return pd.DataFrame()

        print("🚀 开始真实压缩测试 (这可能需要几十秒)...")
        
        for f_type, file_list in files_map.items():
            for fp in file_list:
                try:
                    with open(fp, 'rb') as f:
                        raw_data = f.read()
                    
                    original_size = len(raw_data)
                    filename = os.path.basename(fp)
                    
                    # 定义要测试的算法
                    algos = ['Gzip']
                    if HAS_LIBS: algos += ['LZ4', 'Zstd']
                    else: algos += ['LZMA (Sim Zstd)'] # Fallback
                    
                    for algo in algos:
                        # --- 压缩 ---
                        t0 = time.perf_counter()
                        if algo == 'Gzip':
                            comp_data = gzip.compress(raw_data, compresslevel=6)
                        elif algo == 'LZ4':
                            comp_data = lz4.frame.compress(raw_data, compression_level=3)
                        elif algo == 'Zstd':
                            comp_data = zstd.ZstdCompressor(level=3).compress(raw_data)
                        elif 'LZMA' in algo:
                            import lzma
                            comp_data = lzma.compress(raw_data)
                        
                        # --- 解压 (关键指标) ---
                        t1 = time.perf_counter()
                        if algo == 'Gzip':
                            gzip.decompress(comp_data)
                        elif algo == 'LZ4':
                            lz4.frame.decompress(comp_data)
                        elif algo == 'Zstd':
                            zstd.ZstdDecompressor().decompress(comp_data)
                        elif 'LZMA' in algo:
                            lzma.decompress(comp_data)
                        t2 = time.perf_counter()
                        
                        # 记录数据
                        decomp_time = t2 - t1
                        ratio = len(comp_data) / original_size
                        
                        self.results.append({
                            'File Type': f_type,
                            'File Name': filename,
                            'Algorithm': algo.split(' ')[0], # 清理名字
                            'Size (KB)': original_size / 1024,
                            'Compression Ratio': ratio,
                            'Decomp Time (ms)': decomp_time * 1000
                        })
                        
                except Exception as e:
                    continue
                    
        return pd.DataFrame(self.results)

    def plot_three_views(self, df):
        if df.empty: return

        # 颜色配置
        colors = {'Gzip': '#d62728', 'LZ4': '#1f77b4', 'Zstd': '#2ca02c', 'LZMA': '#2ca02c'}
        
        # 创建大图
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # ==========================================
        # 图 1: 内容异构性 (Boxplot) - 证明压缩率随内容波动大
        # ==========================================
        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=df, x='File Type', y='Compression Ratio', hue='Algorithm', 
                    palette=colors, ax=ax1, linewidth=1.5)
        ax1.set_title('(a) 内容异构性：不同文件类型的压缩率分布', fontsize=12, fontweight='bold')
        ax1.set_ylabel('压缩率 (Compressed/Original)', fontsize=10)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # ==========================================
        # 图 2: 算力敏感性 (Barplot) - 证明解压时间差异大
        # ==========================================
        ax2 = fig.add_subplot(gs[0, 1])
        # 聚合取平均值
        avg_time = df.groupby(['Algorithm', 'File Type'])['Decomp Time (ms)'].mean().reset_index()
        sns.barplot(data=avg_time, x='File Type', y='Decomp Time (ms)', hue='Algorithm', 
                    palette=colors, ax=ax2)
        ax2.set_title('(b) 算力敏感性：解压时间开销对比 (单核模式)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('平均解压耗时 (ms)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        # 标注倍数差距
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.0f', padding=3, fontsize=9)

        # ==========================================
        # 图 3: 粒度影响 (Scatter) - 证明小文件收益不稳定
        # ==========================================
        ax3 = fig.add_subplot(gs[1, :]) # 占满下方
        # 只看 Gzip (作为基准)
        subset = df[df['Algorithm'] == 'Gzip']
        sns.scatterplot(data=subset, x='Size (KB)', y='Compression Ratio', hue='File Type', 
                        style='File Type', s=100, alpha=0.7, ax=ax3)
        
        ax3.set_xscale('log') # 这种图通常用对数轴
        ax3.set_title('(c) 粒度影响：文件大小与压缩收益的关系 (Gzip)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('文件大小 (KB, 对数刻度)', fontsize=10)
        ax3.set_ylabel('压缩率', fontsize=10)
        
        # 画一条 1.0 的线
        ax3.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='无收益基线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('motivation_3_views.png', dpi=300)
        print("\n✅ 核心动机图已生成: motivation_3_views.png")
        print("   这张图包含了：分布差异、计算代价差异、粒度不确定性。完美。")

if __name__ == "__main__":
    motivator = RealWorldMotivation()
    df_res = motivator.run_benchmark()
    motivator.plot_three_views(df_res)