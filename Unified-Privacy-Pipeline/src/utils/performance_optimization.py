"""
Performance optimization utilities for the Unified Privacy Pipeline.

Provides tools and techniques to optimize memory usage, computational efficiency,
and scalability for large-scale privacy-preserving machine learning.
"""

import torch
import torch.nn as nn
import gc
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import time
import threading
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    # Memory optimization
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    
    # Computation optimization
    enable_compilation: bool = True
    enable_tensor_fusion: bool = True
    batch_size_auto_tune: bool = True
    
    # Parallel processing
    enable_data_parallel: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Caching and storage
    enable_gradient_caching: bool = True
    cache_influence_computations: bool = True
    checkpoint_frequency: int = 10
    
    # Monitoring
    enable_profiling: bool = False
    log_memory_usage: bool = True
    performance_monitoring_interval: int = 60  # seconds


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_history = []
        
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        memory_stats = {
            'cpu_memory_mb': memory_info.rss / 1024 / 1024,
            'cpu_memory_percent': process.memory_percent(),
        }
        
        if torch.cuda.is_available():
            memory_stats.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_memory_percent': torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            })
        
        return memory_stats
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """Context manager for monitoring memory usage during operations."""
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            end_time = time.time()
            
            memory_delta = {
                key: end_memory[key] - start_memory[key] 
                for key in start_memory.keys()
            }
            
            self.memory_history.append({
                'operation': operation_name,
                'duration': end_time - start_time,
                'memory_delta': memory_delta,
                'peak_memory': end_memory
            })
            
            if self.config.log_memory_usage:
                logger.debug(f"{operation_name}: Memory delta: {memory_delta}, Duration: {end_time - start_time:.2f}s")
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Optimize model for memory efficiency."""
        if not self.config.enable_memory_optimization:
            return model
        
        # Apply gradient checkpointing
        if self.config.gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)
        
        # Apply mixed precision optimization
        if self.config.mixed_precision and torch.cuda.is_available():
            model = model.half()  # Convert to FP16
            logger.info("Applied mixed precision (FP16) optimization")
        
        return model
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to reduce memory usage."""
        try:
            # Apply to transformer-like layers if available
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
                elif isinstance(module, (nn.TransformerEncoder, nn.TransformerEncoderLayer)):
                    # Wrap transformer layers
                    module.forward = torch.utils.checkpoint.checkpoint(module.forward)
                elif len(list(module.children())) > 0:
                    # For sequential-like modules, checkpoint every few layers
                    children = list(module.children())
                    if len(children) > 4:  # Only apply to deeper modules
                        for i in range(0, len(children), 3):  # Checkpoint every 3 layers
                            if i + 2 < len(children):
                                checkpointed_block = nn.Sequential(*children[i:i+3])
                                checkpointed_block.forward = torch.utils.checkpoint.checkpoint(checkpointed_block.forward)
            
            logger.info("Applied gradient checkpointing")
        except Exception as e:
            logger.warning(f"Could not apply gradient checkpointing: {e}")
        
        return model
    
    def clear_cache(self, aggressive: bool = False):
        """Clear various caches to free up memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        if aggressive:
            # More aggressive cleanup
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux only
            
        logger.debug("Cleared memory caches")
    
    def auto_tune_batch_size(self,
                           model: nn.Module,
                           sample_data: torch.Tensor,
                           max_memory_gb: Optional[float] = None) -> int:
        """Automatically determine optimal batch size based on memory constraints."""
        if not self.config.batch_size_auto_tune:
            return 32  # Default batch size
        
        max_memory = max_memory_gb or self.config.max_memory_usage_gb
        device = next(model.parameters()).device
        
        # Start with small batch size and increase
        batch_sizes_to_try = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        optimal_batch_size = 1
        
        model.eval()
        
        for batch_size in batch_sizes_to_try:
            try:
                # Create test batch
                test_batch = sample_data[:batch_size] if len(sample_data) >= batch_size else sample_data
                test_batch = test_batch.to(device)
                
                # Test memory usage
                with self.memory_context(f"batch_size_{batch_size}"):
                    with torch.no_grad():
                        _ = model(test_batch)
                
                # Check if memory usage is within limits
                current_memory = self.get_memory_usage()
                if torch.cuda.is_available():
                    gpu_memory_gb = current_memory['gpu_memory_allocated_mb'] / 1024
                    if gpu_memory_gb > max_memory:
                        break
                else:
                    cpu_memory_gb = current_memory['cpu_memory_mb'] / 1024
                    if cpu_memory_gb > max_memory:
                        break
                
                optimal_batch_size = batch_size
                
            except torch.cuda.OutOfMemoryError:
                break
            except Exception as e:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
                break
            finally:
                self.clear_cache()
        
        logger.info(f"Auto-tuned batch size: {optimal_batch_size}")
        return optimal_batch_size


class ComputationOptimizer:
    """Computation optimization utilities."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.compilation_cache = {}
        
    def optimize_model_computation(self, model: nn.Module) -> nn.Module:
        """Optimize model for computational efficiency."""
        if not torch.cuda.is_available():
            return model
        
        # Apply torch.compile if available (PyTorch 2.0+)
        if self.config.enable_compilation and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='default')
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"Could not apply torch.compile: {e}")
        
        # Apply tensor fusion optimizations
        if self.config.enable_tensor_fusion:
            model = self._apply_tensor_fusion(model)
        
        return model
    
    def _apply_tensor_fusion(self, model: nn.Module) -> nn.Module:
        """Apply tensor fusion optimizations."""
        try:
            # Fuse batch norm with conv layers
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)
            logger.info("Applied tensor fusion optimizations")
        except Exception as e:
            logger.warning(f"Could not apply tensor fusion: {e}")
        
        return model
    
    @staticmethod
    def optimize_data_loading(dataloader: torch.utils.data.DataLoader,
                            num_workers: int = 4,
                            pin_memory: bool = True) -> torch.utils.data.DataLoader:
        """Optimize data loading for better performance."""
        
        # Create optimized dataloader with same dataset
        optimized_loader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=dataloader.drop_last,  # Use existing settings
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
        )
        
        logger.info(f"Optimized data loading: {num_workers} workers, pin_memory={pin_memory}")
        return optimized_loader


class GradientCache:
    """Cache for gradient computations to avoid recomputation."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        
    def get_cache_key(self, model_state: Dict, data_hash: str) -> str:
        """Generate cache key for gradient computation."""
        # Simple hash of model parameters and data
        model_hash = hash(tuple(p.data.cpu().numpy().tobytes() for p in model_state.values()
                               if p.requires_grad)[:5])  # Only hash first few parameters
        return f"{model_hash}_{data_hash}"
    
    def get(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached gradients."""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key: str, gradients: Dict[str, torch.Tensor]):
        """Cache gradients."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        # Deep copy gradients to avoid issues
        cached_gradients = {name: grad.clone().detach() for name, grad in gradients.items()}
        self.cache[key] = cached_gradients
        self.access_count[key] = 1
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.metrics_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started performance monitoring")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = {
                    'timestamp': time.time(),
                    'memory': MemoryOptimizer.get_memory_usage(),
                    'cpu_percent': psutil.cpu_percent(),
                }
                
                if torch.cuda.is_available():
                    metrics['gpu_utilization'] = torch.cuda.utilization()
                
                self.metrics_history.append(metrics)
                
                # Keep only recent history
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                time.sleep(self.config.performance_monitoring_interval)
                
            except Exception as e:
                logger.warning(f"Error in performance monitoring: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics_history:
            return {}
        
        # Extract memory usage over time
        memory_usage = [m['memory']['cpu_memory_mb'] for m in self.metrics_history]
        cpu_usage = [m['cpu_percent'] for m in self.metrics_history]
        
        summary = {
            'memory_stats': {
                'avg_memory_mb': np.mean(memory_usage),
                'max_memory_mb': np.max(memory_usage),
                'min_memory_mb': np.min(memory_usage),
                'memory_std_mb': np.std(memory_usage)
            },
            'cpu_stats': {
                'avg_cpu_percent': np.mean(cpu_usage),
                'max_cpu_percent': np.max(cpu_usage)
            },
            'monitoring_duration': len(self.metrics_history) * self.config.performance_monitoring_interval
        }
        
        if torch.cuda.is_available() and 'gpu_utilization' in self.metrics_history[0]:
            gpu_usage = [m.get('gpu_utilization', 0) for m in self.metrics_history]
            summary['gpu_stats'] = {
                'avg_gpu_percent': np.mean(gpu_usage),
                'max_gpu_percent': np.max(gpu_usage)
            }
        
        return summary


class ProfilerContext:
    """Context manager for profiling operations."""
    
    def __init__(self, operation_name: str, enabled: bool = True):
        self.operation_name = operation_name
        self.enabled = enabled
        self.profiler = None
        
    def __enter__(self):
        if self.enabled and torch.cuda.is_available():
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler:
            self.profiler.stop()
            
            # Save profiler results
            trace_path = f"./profiler_traces/{self.operation_name}_trace.json"
            try:
                self.profiler.export_chrome_trace(trace_path)
                logger.info(f"Profiler trace saved to {trace_path}")
            except Exception as e:
                logger.warning(f"Could not save profiler trace: {e}")


def performance_decorator(operation_name: str):
    """Decorator to monitor performance of functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            memory_optimizer = MemoryOptimizer(PerformanceConfig())
            
            with memory_optimizer.memory_context(operation_name):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                logger.debug(f"{operation_name} completed in {end_time - start_time:.2f}s")
                return result
        
        return wrapper
    return decorator


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.computation_optimizer = ComputationOptimizer(self.config)
        self.gradient_cache = GradientCache() if self.config.enable_gradient_caching else None
        self.monitor = PerformanceMonitor(self.config)
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply all available optimizations to a model."""
        logger.info("Applying performance optimizations...")
        
        # Memory optimizations
        model = self.memory_optimizer.optimize_model_memory(model)
        
        # Computation optimizations
        model = self.computation_optimizer.optimize_model_computation(model)
        
        logger.info("Performance optimizations applied")
        return model
    
    def optimize_dataloader(self, dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """Optimize dataloader for better performance."""
        return ComputationOptimizer.optimize_data_loading(
            dataloader,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and get performance summary."""
        self.monitor.stop_monitoring()
        return self.monitor.get_performance_summary()
    
    def create_profiler_context(self, operation_name: str):
        """Create profiler context for detailed profiling."""
        return ProfilerContext(operation_name, self.config.enable_profiling)


def create_performance_config(**kwargs) -> PerformanceConfig:
    """Create performance configuration with sensible defaults."""
    return PerformanceConfig(**kwargs)


# Utility functions for easy access
def optimize_for_inference(model: nn.Module) -> nn.Module:
    """Quick optimization for inference workloads."""
    config = PerformanceConfig(
        enable_memory_optimization=True,
        mixed_precision=True,
        enable_compilation=True,
        gradient_checkpointing=False  # Not needed for inference
    )
    
    optimizer = PerformanceOptimizer(config)
    return optimizer.optimize_model(model)


def optimize_for_training(model: nn.Module, enable_checkpointing: bool = True) -> nn.Module:
    """Quick optimization for training workloads."""
    config = PerformanceConfig(
        enable_memory_optimization=True,
        gradient_checkpointing=enable_checkpointing,
        mixed_precision=True,
        enable_compilation=False,  # Can interfere with gradient computation
        enable_gradient_caching=True
    )
    
    optimizer = PerformanceOptimizer(config)
    return optimizer.optimize_model(model)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = create_performance_config(
        enable_memory_optimization=True,
        mixed_precision=True,
        enable_compilation=True,
        enable_profiling=True
    )
    
    # Create optimizer
    optimizer = PerformanceOptimizer(config)
    
    # Example model optimization
    dummy_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    optimized_model = optimizer.optimize_model(dummy_model)
    
    print("Performance optimization example completed!")