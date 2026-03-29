"""
Profiling utility for CARS.
"""

import time
import torch
from typing import Dict


class ProfileTimer:
    """Tracks time per named region, handling both CPU and GPU operations."""
    
    def __init__(self):
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._gpu_available = torch.cuda.is_available()
    
    def __call__(self, name: str, gpu_sync: bool = False):
        """Create a timing context.
        
        Args:
            name: Region name.
            gpu_sync: If True, synchronize CUDA before and after to capture
                      true GPU execution time. Use for operations that launch
                      GPU kernels (e.g., model.generate).
        """
        return _TimerContext(self, name, gpu_sync and self._gpu_available)
    
    def record(self, name: str, elapsed: float):
        if name not in self.totals:
            self.totals[name] = 0.0
            self.counts[name] = 0
        self.totals[name] += elapsed
        self.counts[name] += 1
    
    def reset(self):
        self.totals.clear()
        self.counts.clear()
    
    def report(self, title: str = "Profile Report"):
        if not self.totals:
            print("No profiling data collected.")
            return
        
        total_time = sum(self.totals.values())
        
        print()
        print(f"{'='*75}")
        print(f"  {title}")
        print(f"{'='*75}")
        print(f"  {'Operation':<30} {'Time (s)':>9} {'Calls':>7} "
              f"{'%':>7} {'Avg (ms)':>10}")
        print(f"  {'-'*30} {'-'*9} {'-'*7} "
              f"{'-'*7} {'-'*10}")
        
        for name in sorted(self.totals, key=lambda k: self.totals[k], reverse=True):
            t = self.totals[name]
            n = self.counts[name]
            pct = 100 * t / total_time if total_time > 0 else 0
            avg_ms = 1000 * t / n if n > 0 else 0
            
            print(f"  {name:<30} {t:>9.3f} {n:>7d} "
                  f"{pct:>6.1f}% {avg_ms:>10.3f}")
        
        print(f"  {'-'*30} {'-'*9} {'-'*7} "
              f"{'-'*7} {'-'*10}")
        print(f"  {'TOTAL':<30} {total_time:>9.3f}")
        print(f"{'='*75}")
        print()


class _TimerContext:
    def __init__(self, timer: ProfileTimer, name: str, gpu_sync: bool):
        self.timer = timer
        self.name = name
        self.gpu_sync = gpu_sync
    
    def __enter__(self):
        if self.gpu_sync:
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.gpu_sync:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self.timer.record(self.name, elapsed)