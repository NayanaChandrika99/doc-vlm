"""
Prompt Selector - Intelligent prompt selection based on region characteristics

Matches region type + difficulty to optimal prompt template.
Tracks historical performance for continuous improvement.
"""
from pathlib import Path
from typing import Dict, Optional, List
import yaml
import logging

logger = logging.getLogger(__name__)


class PromptSelector:
    """
    Select optimal prompt template based on region characteristics.
    
    Considers:
    - Region type (checkbox, table, signature, etc.)
    - Difficulty score
    - Historical accuracy per prompt
    """
    
    def __init__(self, prompts_dir: Path):
        """
        Initialize prompt selector.
        
        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        self.prompts_dir = Path(prompts_dir)
        self.prompts = self._load_prompts()
        
        logger.info(f"Loaded {len(self.prompts)} prompt templates")
    
    def select_prompt(
        self,
        region_type: str,
        difficulty_score: float,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Select best prompt for region.
        
        Args:
            region_type: Type of region (checkbox, table, etc.)
            difficulty_score: Difficulty score (0-1)
            metadata: Additional context
        
        Returns:
            Dict with 'template', 'version', 'metadata'
        """
        # Filter by region type
        candidates = [p for p in self.prompts 
                     if p.get('region_type') == region_type]
        
        if not candidates:
            logger.warning(f"No prompts found for region_type={region_type}, using default")
            return self._default_prompt(region_type)
        
        # Filter by difficulty range
        candidates = [p for p in candidates
                     if p.get('difficulty_range', [0, 1])[0] <= difficulty_score <= 
                        p.get('difficulty_range', [0, 1])[1]]
        
        if not candidates:
            # Use any prompt for this region type if no difficulty match
            candidates = [p for p in self.prompts 
                         if p.get('region_type') == region_type]
        
        # Select best by historical accuracy
        best = max(candidates, key=lambda p: p.get('historical_accuracy', 0.5))
        
        logger.debug(f"Selected prompt {best.get('version')} for {region_type} "
                    f"(difficulty={difficulty_score:.2f})")
        
        return best
    
    def _load_prompts(self) -> List[Dict]:
        """Load all prompt templates from directory."""
        prompts = []
        
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return prompts
        
        for prompt_file in self.prompts_dir.glob("*.yaml"):
            try:
                with open(prompt_file) as f:
                    prompt = yaml.safe_load(f)
                    prompt['filename'] = prompt_file.name
                    prompts.append(prompt)
            except Exception as e:
                logger.error(f"Failed to load prompt {prompt_file}: {e}")
        
        return prompts
    
    def _default_prompt(self, region_type: str) -> Dict:
        """Return default prompt when no match found."""
        return {
            "template": f"Extract all information from this {region_type} region and return as JSON.",
            "version": "default_v1.0",
            "region_type": region_type,
            "difficulty_range": [0, 1],
            "historical_accuracy": 0.5
        }
    
    def get_prompt_stats(self) -> Dict:
        """Get statistics about loaded prompts."""
        by_type = {}
        for prompt in self.prompts:
            region_type = prompt.get('region_type', 'unknown')
            by_type[region_type] = by_type.get(region_type, 0) + 1
        
        return {
            "total_prompts": len(self.prompts),
            "by_type": by_type,
            "avg_accuracy": sum(p.get('historical_accuracy', 0.5) 
                              for p in self.prompts) / len(self.prompts)
                           if self.prompts else 0
        }

