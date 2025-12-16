"""
客户端探针模块测试
"""


import unittest
from modules.client_probe import ClientProbe


class TestClientProbe(unittest.TestCase):
    
    def setUp(self):
        self.probe = ClientProbe()
    
    def test_probe_cpu(self):
        """测试CPU探测功能"""
        cpu_score = self.probe.probe_cpu()
        self.assertIsInstance(cpu_score, int)
        self.assertGreater(cpu_score, 0)
    
    def test_probe_bandwidth(self):
        """测试带宽探测功能"""
        bandwidth = self.probe.probe_bandwidth()
        self.assertIsInstance(bandwidth, (int, float))
        self.assertGreater(bandwidth, 0)
    
    def test_probe_decompression_speed(self):
        """测试解压速度探测功能"""
        speed = self.probe.probe_decompression_speed()
        self.assertIsInstance(speed, (int, float))
        self.assertGreater(speed, 0)
    
    def test_get_client_profile(self):
        """测试获取完整客户端配置"""
        profile = self.probe.get_client_profile()
        self.assertIsInstance(profile, dict)
        self.assertIn("cpu_score", profile)
        self.assertIn("bandwidth_mbps", profile)
        self.assertIn("decompression_speed", profile)
        self.assertIn("latency_requirement", profile)


if __name__ == "__main__":
    unittest.main()