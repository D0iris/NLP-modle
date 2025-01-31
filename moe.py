"""Mixture of Experts implementation with elastic routing and quantization"""

import torch
import torch.nn as nn
from typing import Dict, Optional

class MoEConfig:
    """Configuration for Mixture of Experts with Elastic Shared Routing"""
    def __init__(self,
                 num_shared_experts: int = 2,
                 num_task_experts: int = 8,
                 expert_capacity: float = 0.2,
                 router_init: Optional[Dict] = None,
                 max_shared_experts: int = 8,
                 expansion_threshold: float = 0.8,
                 circuit_breaker_threshold: float = 0.95):
        """
        Args:
            num_shared_experts: Number of shared experts
            num_task_experts: Number of task-specific experts
            expert_capacity: Capacity factor for experts
            router_init: Router initialization parameters
            max_shared_experts: Maximum number of shared experts
            expansion_threshold: Load threshold for expert expansion (0.0-1.0)
            circuit_breaker_threshold: Load threshold for circuit breaker (0.0-1.0)
        """
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.expert_capacity = expert_capacity
        self.router_init = router_init or {}
        self.max_shared_experts = max_shared_experts
        self.expansion_threshold = expansion_threshold
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.current_load = 0.0
        self.circuit_breaker_active = False

class MixtureOfExperts(nn.Module):
    """Mixture of Experts implementation with elastic routing and quantization"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Shared experts with INT8 quantization and QAT
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 1024),
                nn.quantized.FloatFunctional(),
                torch.quantization.QuantStub(),
                torch.quantization.DeQuantStub()
            ) for _ in range(config.num_shared_experts)
        ])
        
        # Prepare for QAT
        for expert in self.shared_experts:
            torch.quantization.prepare_qat(expert, inplace=True)
        
        # Task experts remain in FP32/FP16
        self.task_experts = nn.ModuleList([
            nn.Linear(1024, 1024) for _ in range(config.num_task_experts)
        ])
        
        # Router in FP16 precision
        self.router = nn.Linear(1024, config.num_shared_experts + config.num_task_experts).half()
        
        # Initialize load tracking
        self.expert_loads = torch.zeros(config.num_shared_experts + config.num_task_experts)
        self.last_expansion_step = 0

    def update_load_tracking(self, expert_indices: torch.Tensor):
        """Update expert load tracking statistics"""
        for idx in expert_indices:
            self.expert_loads[idx] += 1
        self.config.current_load = self.expert_loads.max().item() / self.config.expert_capacity
        
    def expand_shared_experts(self):
        """Dynamically expand shared experts when load exceeds threshold"""
        if (self.config.current_load > self.config.expansion_threshold and
            len(self.shared_experts) < self.config.max_shared_experts and
            self.current_step > self.last_expansion_step + 1000):
            
            new_expert = nn.Linear(1024, 1024)
            self.shared_experts.append(new_expert)
            self.config.num_shared_experts += 1
            
            # Update router to account for new expert
            old_router = self.router
            self.router = nn.Linear(1024, self.config.num_shared_experts + self.config.num_task_experts)
            with torch.no_grad():
                self.router.weight[:old_router.out_features] = old_router.weight
                self.router.bias[:old_router.out_features] = old_router.bias
                
            self.last_expansion_step = self.current_step
            
    def check_circuit_breaker(self):
        """Activate circuit breaker if load exceeds threshold"""
        if self.config.current_load > self.config.circuit_breaker_threshold:
            self.config.circuit_breaker_active = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route input to appropriate experts with load balancing"""
        if self.config.circuit_breaker_active:
            # Fallback to basic routing when circuit breaker is active
            return self.shared_experts[0](x)
            
        # Calculate routing probabilities
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)
        
        # Select top-k experts
        expert_indices = torch.topk(probs, k=2).indices
        self.update_load_tracking(expert_indices)
        
        # Apply expert outputs
        outputs = []
        for idx in expert_indices:
            if idx < len(self.shared_experts):
                outputs.append(self.shared_experts[idx](x))
            else:
                task_idx = idx - len(self.shared_experts)
                outputs.append(self.task_experts[task_idx](x))
                
        return sum(outputs) / len(outputs)
