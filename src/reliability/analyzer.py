"""
Reliability Assessment and Evidence Fusion
Combines confidence, localization, and stability into composite score
"""
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import roc_auc_score, average_precision_score


class ReliabilityAnalyzer:
    """
    Fuses evidence from multiple sources to assess prediction reliability
    """
    
    def __init__(
        self,
        reliability_threshold: float = 0.70,
        weights: Optional[Dict[str, float]] = None,
        baseline_type: str = "confidence_only"
    ):
        self.reliability_threshold = reliability_threshold
        
        # Default weights for evidence fusion
        self.weights = weights or {
            'confidence': 0.3,
            'localization': 0.4,
            'stability': 0.3
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        self.baseline_type = baseline_type
        self.audit_history = []
    
    def fuse_evidence(
        self,
        confidence: float,
        localization_score: float,
        stability_score: float
    ) -> float:
        """
        Fuse evidence from confidence, localization, and stability
        
        Args:
            confidence: Model prediction confidence [0, 1]
            localization_score: Localization metric (IoU, Dice, etc.) [0, 1]
            stability_score: Stability metric (SSIM, Spearman) [0, 1]
        
        Returns:
            reliability_score: Composite reliability score [0, 1]
        """
        # Weighted fusion
        reliability_score = (
            self.weights['confidence'] * confidence +
            self.weights['localization'] * localization_score +
            self.weights['stability'] * stability_score
        )
        
        return reliability_score
    
    def assign_reliability_flag(self, score: float) -> str:
        """
        Assign reliability flag based on score
        
        Args:
            score: Reliability score [0, 1]
        
        Returns:
            flag: "RELIABLE" or "UNRELIABLE"
        """
        if score >= self.reliability_threshold:
            return "RELIABLE"
        else:
            return "UNRELIABLE"
    
    def assess_reliability(
        self,
        confidence: float,
        localization_metrics: Dict[str, float],
        stability_metrics: Dict[str, float]
    ) -> Tuple[float, str, Dict]:
        """
        Perform complete reliability assessment
        
        Args:
            confidence: Prediction confidence
            localization_metrics: Dict of localization scores
            stability_metrics: Dict of stability scores
        
        Returns:
            reliability_score: Composite score
            flag: Reliability flag
            breakdown: Detailed score breakdown
        """
        # Get composite scores
        loc_score = self._get_composite_localization(localization_metrics)
        stab_score = self._get_composite_stability(stability_metrics)
        
        # Fuse evidence
        reliability_score = self.fuse_evidence(confidence, loc_score, stab_score)
        
        # Assign flag
        flag = self.assign_reliability_flag(reliability_score)
        
        # Detailed breakdown
        breakdown = {
            'reliability_score': reliability_score,
            'confidence': confidence,
            'localization_score': loc_score,
            'stability_score': stab_score,
            'flag': flag,
            'weights': self.weights,
            'threshold': self.reliability_threshold
        }
        
        # Log to audit history
        self.audit_history.append(breakdown)
        
        return reliability_score, flag, breakdown
    
    def _get_composite_localization(self, metrics: Dict[str, float]) -> float:
        """Compute composite localization score"""
        # Average of IoU, Dice, Pointing Game, Energy Ratio
        relevant_metrics = ['iou', 'dice', 'pointing_game', 'energy_ratio']
        scores = [metrics.get(m, 0) for m in relevant_metrics if m in metrics]
        
        if len(scores) == 0:
            return 0.0
        
        return np.mean(scores)
    
    def _get_composite_stability(self, metrics: Dict[str, float]) -> float:
        """Compute composite stability score"""
        # Average of mean SSIM and mean Spearman
        ssim = metrics.get('mean_ssim', 0)
        spearman = metrics.get('mean_spearman', 0)
        
        return 0.5 * ssim + 0.5 * spearman
    
    def compute_auroc_auprc(
        self,
        y_true: np.ndarray,
        reliability_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute AUROC and AUPRC for reliability assessment
        
        Args:
            y_true: Binary labels (1 = correct prediction, 0 = incorrect)
            reliability_scores: Predicted reliability scores
        
        Returns:
            metrics: Dictionary with AUROC and AUPRC
        """
        try:
            auroc = roc_auc_score(y_true, reliability_scores)
            auprc = average_precision_score(y_true, reliability_scores)
        except Exception as e:
            print(f"Error computing AUROC/AUPRC: {e}")
            auroc = 0.0
            auprc = 0.0
        
        return {
            'auroc': auroc,
            'auprc': auprc
        }
    
    def risk_coverage_analysis(
        self,
        reliability_scores: np.ndarray,
        correctness: np.ndarray,
        coverage_levels: List[float] = [0.5, 0.7, 0.8, 0.9]
    ) -> Dict[str, Dict]:
        """
        Analyze risk-coverage trade-off
        
        Args:
            reliability_scores: Array of reliability scores
            correctness: Binary array (1 = correct, 0 = incorrect)
            coverage_levels: Coverage percentages to analyze
        
        Returns:
            analysis: Risk metrics at different coverage levels
        """
        # Sort by reliability score (descending)
        sorted_indices = np.argsort(reliability_scores)[::-1]
        sorted_correctness = correctness[sorted_indices]
        
        analysis = {}
        
        for coverage in coverage_levels:
            # Number of samples to cover
            n_samples = int(len(reliability_scores) * coverage)
            
            if n_samples == 0:
                continue
            
            # Accuracy of covered samples
            covered_correctness = sorted_correctness[:n_samples]
            accuracy = covered_correctness.mean()
            
            # Error rate (risk)
            error_rate = 1.0 - accuracy
            
            analysis[f'coverage_{int(coverage*100)}'] = {
                'coverage': coverage,
                'n_samples': n_samples,
                'accuracy': accuracy,
                'error_rate': error_rate
            }
        
        return analysis
    
    def compare_with_baseline(
        self,
        reliability_scores: np.ndarray,
        baseline_scores: np.ndarray,
        correctness: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare evidence-aware reliability with baseline
        
        Args:
            reliability_scores: Evidence-fused reliability scores
            baseline_scores: Baseline scores (e.g., confidence only)
            correctness: Binary correctness array
        
        Returns:
            comparison: Metrics comparing both approaches
        """
        # Compute AUROC for both
        auroc_evidence = roc_auc_score(correctness, reliability_scores)
        auroc_baseline = roc_auc_score(correctness, baseline_scores)
        
        # Compute AUPRC for both
        auprc_evidence = average_precision_score(correctness, reliability_scores)
        auprc_baseline = average_precision_score(correctness, baseline_scores)
        
        comparison = {
            'auroc_evidence_aware': auroc_evidence,
            'auroc_baseline': auroc_baseline,
            'auroc_improvement': auroc_evidence - auroc_baseline,
            'auprc_evidence_aware': auprc_evidence,
            'auprc_baseline': auprc_baseline,
            'auprc_improvement': auprc_evidence - auprc_baseline,
        }
        
        return comparison
    
    def get_audit_summary(self) -> Dict:
        """Get summary of all reliability assessments"""
        if len(self.audit_history) == 0:
            return {'message': 'No assessments recorded'}
        
        scores = [a['reliability_score'] for a in self.audit_history]
        flags = [a['flag'] for a in self.audit_history]
        
        summary = {
            'total_assessments': len(self.audit_history),
            'mean_reliability': np.mean(scores),
            'std_reliability': np.std(scores),
            'reliable_count': flags.count('RELIABLE'),
            'unreliable_count': flags.count('UNRELIABLE'),
            'reliable_percentage': 100.0 * flags.count('RELIABLE') / len(flags)
        }
        
        return summary
