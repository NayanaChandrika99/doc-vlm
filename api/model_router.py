"""
Model Router - A/B testing and model versioning

Routing strategies:
- Weighted: Probabilistic routing (e.g., 80% modelA, 20% modelB)
- Epsilon-greedy: Explore/exploit (90% best model, 10% random)
- Sticky-session: Route same user to same model
"""
from typing import Dict, Optional, Tuple
import random
import hashlib
import logging

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Route requests to appropriate model variant.
    
    Supports:
    - A/B testing
    - Canary deployments
    - Shadow mode (run both, only return one)
    """
    
    def __init__(self, routing_config: Dict):
        """
        Initialize model router.
        
        Args:
            routing_config: Dict with routing strategy and model variants
                {
                    "strategy": "weighted",  # weighted, epsilon_greedy, sticky
                    "models": {
                        "olmocr-v1": {"weight": 0.8, "path": "..."},
                        "olmocr-v2": {"weight": 0.2, "path": "..."}
                    },
                    "default_model": "olmocr-v1",
                    "epsilon": 0.1  # For epsilon-greedy
                }
        """
        self.config = routing_config
        self.strategy = routing_config.get("strategy", "weighted")
        self.models = routing_config.get("models", {})
        self.default_model = routing_config.get("default_model")
        self.epsilon = routing_config.get("epsilon", 0.1)
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"ModelRouter initialized with strategy: {self.strategy}")
    
    def route(
        self,
        request_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """
        Select model variant for request.
        
        Args:
            request_id: Unique request identifier
            user_id: Optional user ID for sticky routing
            context: Optional context (e.g., region_type, difficulty)
        
        Returns:
            (model_id, model_config)
        """
        if self.strategy == "weighted":
            return self._weighted_routing()
        
        elif self.strategy == "epsilon_greedy":
            return self._epsilon_greedy_routing(context)
        
        elif self.strategy == "sticky":
            if user_id:
                return self._sticky_routing(user_id)
            else:
                logger.warning("Sticky routing requested but no user_id provided")
                return self._weighted_routing()
        
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using default")
            return self._default_routing()
    
    def _weighted_routing(self) -> Tuple[str, Dict]:
        """Weighted random selection."""
        model_ids = list(self.models.keys())
        weights = [self.models[m].get('weight', 1.0) for m in model_ids]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Select
        selected = random.choices(model_ids, weights=weights)[0]
        
        logger.debug(f"Weighted routing selected: {selected}")
        return selected, self.models[selected]
    
    def _epsilon_greedy_routing(self, context: Optional[Dict]) -> Tuple[str, Dict]:
        """
        Epsilon-greedy: exploit best model vs explore others.
        
        Best model determined by historical accuracy for this context.
        """
        # Exploration
        if random.random() < self.epsilon:
            selected = random.choice(list(self.models.keys()))
            logger.debug(f"Epsilon-greedy exploration: {selected}")
            return selected, self.models[selected]
        
        # Exploitation: select best model
        # Placeholder: Would query historical accuracy from database
        best_model = self.default_model or list(self.models.keys())[0]
        
        logger.debug(f"Epsilon-greedy exploitation: {best_model}")
        return best_model, self.models[best_model]
    
    def _sticky_routing(self, user_id: str) -> Tuple[str, Dict]:
        """
        Sticky-session: Same user always gets same model.
        
        Uses hash of user_id to deterministically select model.
        """
        model_ids = list(self.models.keys())
        
        # Hash user_id to index
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        index = hash_value % len(model_ids)
        
        selected = model_ids[index]
        
        logger.debug(f"Sticky routing for user {user_id}: {selected}")
        return selected, self.models[selected]
    
    def _default_routing(self) -> Tuple[str, Dict]:
        """Fallback to default model."""
        if self.default_model and self.default_model in self.models:
            return self.default_model, self.models[self.default_model]
        
        # Return first model
        first_model = list(self.models.keys())[0]
        return first_model, self.models[first_model]
    
    def _validate_config(self):
        """Validate routing configuration."""
        if not self.models:
            raise ValueError("No models configured")
        
        if self.strategy == "weighted":
            # Check weights sum to reasonable value
            total_weight = sum(m.get('weight', 1.0) for m in self.models.values())
            if total_weight <= 0:
                raise ValueError("Total model weights must be positive")
        
        if self.default_model and self.default_model not in self.models:
            raise ValueError(f"Default model {self.default_model} not in models")

