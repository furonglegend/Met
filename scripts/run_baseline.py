"""
Baseline Experiment Runner
Run ROME/MEMIT/EMMET baseline experiments with configurable parameters

Usage:
    python scripts/run_baseline.py --method emmet --model gpt2 --num_edits 100
    python scripts/run_baseline.py --method emmet --model gpt2 --num_edits 500 --batch_size 32 --seed 42
    python scripts/run_baseline.py --method emmet --model gpt2 --num_edits 500 --batch_size 32 --replay_rate 0.3
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import random
from typing import Dict, List

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class ExperimentConfig:
    """Experiment configuration"""
    
    def __init__(self, args):
        self.method = args.method
        self.model = args.model
        self.num_edits = args.num_edits
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.output_dir = args.output_dir
        self.replay_rate = args.replay_rate if hasattr(args, 'replay_rate') else 0.0
        self.replay_buffer_size = args.replay_buffer_size if hasattr(args, 'replay_buffer_size') else 200
        self.replay_strategy = args.replay_strategy if hasattr(args, 'replay_strategy') else 'random'
        self.replay_weight = args.replay_weight if hasattr(args, 'replay_weight') else 1.0
        self.use_lora = args.use_lora if hasattr(args, 'use_lora') else False
        self.edit_mode = args.edit_mode if hasattr(args, 'edit_mode') else "raw"
        self.lora_rank = args.lora_rank if hasattr(args, 'lora_rank') else 8
        self.lora_alpha = args.lora_alpha if hasattr(args, 'lora_alpha') else 16.0
        self.lora_scale = args.lora_scale if hasattr(args, 'lora_scale') else 1.0
        self.lora_use_svd = args.lora_use_svd if hasattr(args, 'lora_use_svd') else True
        self.lora_fit_steps = args.lora_fit_steps if hasattr(args, 'lora_fit_steps') else 0
        # Trust / Rollback
        self.trust_enable = getattr(args, 'trust_enable', False)
        self.trust_threshold = getattr(args, 'trust_threshold', 0.3)
        self.trust_action = getattr(args, 'trust_action', 'rollback')
        self.trust_scale = getattr(args, 'trust_scale', 0.5)
        self.trust_heldout_samples = getattr(args, 'trust_heldout_samples', 0)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup paths
        self.hparams_path = PROJECT_ROOT / f"src/hparams/{self.method.upper()}/{self.model}.json"
        self.data_path = PROJECT_ROOT / f"data/{self.dataset}.json"
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_suffix = f"_replay{self.replay_rate}" if self.replay_rate > 0 else ""
        lora_suffix = f"_lora{self.lora_rank}" if self.use_lora else ""
        self.run_dir = Path(self.output_dir) / f"{self.method}_{self.model}_b{self.batch_size}{replay_suffix}{lora_suffix}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.run_dir / "experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def save_config(self):
        """Save experiment configuration"""
        config_dict = {
            "method": self.method,
            "model": self.model,
            "num_edits": self.num_edits,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "replay_rate": self.replay_rate,
            "replay_buffer_size": self.replay_buffer_size,
            "replay_strategy": self.replay_strategy,
            "replay_weight": self.replay_weight,
            "use_lora": self.use_lora,
            "edit_mode": self.edit_mode,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_scale": self.lora_scale,
            "lora_use_svd": self.lora_use_svd,
            "lora_fit_steps": self.lora_fit_steps,
            "trust_enable": self.trust_enable,
            "trust_threshold": self.trust_threshold,
            "trust_action": self.trust_action,
            "trust_scale": self.trust_scale,
            "trust_heldout_samples": self.trust_heldout_samples,
            "dataset": self.dataset,
            "device": self.device,
            "timestamp": datetime.now().isoformat(),
            "hparams_path": str(self.hparams_path),
            "data_path": str(self.data_path)
        }
        
        config_file = self.run_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration saved to {config_file}")
        return config_file
    
    def validate(self):
        """Validate configuration"""
        if not self.hparams_path.exists():
            raise FileNotFoundError(f"Hyperparameters file not found: {self.hparams_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        self.logger.info("Configuration validated successfully")


class BaselineRunner:
    """Run baseline experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = config.logger
        
    def load_data(self):
        """Load dataset and prepare requests"""
        self.logger.info(f"Loading dataset from {self.config.data_path}")
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different data formats
        if isinstance(data, dict):
            # Convert dict format to list
            data = list(data.values())
        
        # Sample num_edits examples
        if len(data) > self.config.num_edits:
            random.seed(self.config.seed)
            data = random.sample(data, self.config.num_edits)
        
        self.logger.info(f"Loaded {len(data)} examples")
        
        # Convert to request format expected by EMMET
        requests = []
        for idx, record in enumerate(data):
            if isinstance(record, dict) and "requested_rewrite" in record:
                rewrite = record["requested_rewrite"]
                request = {
                    "case_id": record.get("case_id", idx),
                    "subject": rewrite["subject"],
                    "prompt": rewrite["prompt"],
                    "target_new": rewrite["target_new"],
                    "target_true": rewrite.get("target_true", {"str": ""}),
                    "paraphrase_prompts": record.get("paraphrase_prompts", []),
                    "neighborhood_prompts": record.get("neighborhood_prompts", []),
                    "generation_prompts": record.get("generation_prompts", []),
                }
                requests.append(request)
            else:
                self.logger.warning(f"Skipping malformed record at index {idx}")
        
        self.logger.info(f"Prepared {len(requests)} editing requests")
        return requests, data
    
    def load_model(self):
        """Load model and tokenizer"""
        self.logger.info(f"Loading model: {self.config.model}")
        
        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(self.config.model)
        tok.pad_token = tok.eos_token
        self.logger.info(f"Tokenizer loaded: {len(tok)} tokens")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(self.config.model)
        model.to(self.config.device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model loaded: {total_params:,} parameters on {self.config.device}")
        
        return model, tok
    
    def run_editing(self, model, tokenizer, requests):
        """Run model editing"""
        self.logger.info(f"Starting {self.config.method.upper()} editing...")
        self.logger.info(f"Batch size: {self.config.batch_size}, Replay rate: {self.config.replay_rate}")
        
        # Load hyperparameters
        with open(self.config.hparams_path, 'r') as f:
            hparams_dict = json.load(f)
        
        # Import appropriate modules
        if self.config.method == "emmet":
            from emmet.emmet_hparams import EMMETHyperParams
            # Choose replay-enabled or standard EMMET
            if self.config.replay_rate > 0:
                from emmet.emmet_replay import apply_emmet_with_replay as apply_emmet_to_model
                self.logger.info(
                    f"Using EMMET with Memory Replay (rate={self.config.replay_rate}, "
                    f"buffer={getattr(self.config, 'replay_buffer_size', 200)}, "
                    f"strategy={getattr(self.config, 'replay_strategy', 'random')}, "
                    f"weight={getattr(self.config, 'replay_weight', 1.0)})"
                )
            else:
                from emmet.emmet_main import apply_emmet_to_model
                self.logger.info("Using standard EMMET (no replay)")
            hparams = EMMETHyperParams.from_json(self.config.hparams_path)
            # Override hparams for native LoRA if requested via CLI
            if getattr(self.config, "edit_mode", "raw") == "lora_native":
                hparams.edit_mode = "lora_native"
                hparams.lora_rank = getattr(self.config, "lora_rank", 8)
                hparams.lora_alpha = getattr(self.config, "lora_alpha", float(hparams.lora_rank))
                hparams.lora_scale = getattr(self.config, "lora_scale", 1.0)
                hparams.lora_use_svd = getattr(self.config, "lora_use_svd", True)
                hparams.lora_fit_steps = getattr(self.config, "lora_fit_steps", 0)
                hparams.allow_fallback = getattr(self.config, "allow_fallback", False)
                hparams.lora_residual_threshold = getattr(self.config, "lora_residual_threshold", None)
            # Trust overrides
            if getattr(self.config, "trust_enable", False):
                hparams.trust_enable = True
                hparams.trust_threshold = getattr(self.config, "trust_threshold", 0.3)
                hparams.trust_action = getattr(self.config, "trust_action", "rollback")
                hparams.trust_scale = getattr(self.config, "trust_scale", 0.5)
                hparams.trust_heldout_samples = getattr(self.config, "trust_heldout_samples", 0)
        elif self.config.method == "memit":
            from memit.memit_hparams import MEMITHyperParams
            from memit.memit_main import apply_memit_to_model
            hparams = MEMITHyperParams.from_json(self.config.hparams_path)
        elif self.config.method == "rome":
            from rome.rome_hparams import ROMEHyperParams
            from rome.rome_main import apply_rome_to_model
            hparams = ROMEHyperParams.from_json(self.config.hparams_path)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        self.logger.info(f"Loaded hyperparameters from {self.config.hparams_path}")
        
        # Process in batches
        all_results = []
        num_batches = (len(requests) + self.config.batch_size - 1) // self.config.batch_size
        
        events_fp = self.config.run_dir / "lora_native_events.jsonl"
        trust_fp = self.config.run_dir / "trust_events.jsonl"
        seq_metrics_fp = self.config.run_dir / "sequence_metrics.jsonl"
        cumulative_edits = 0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(requests))
            batch_requests = requests[start_idx:end_idx]
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Batch {batch_idx + 1}/{num_batches}: Editing examples {start_idx}-{end_idx-1}")
            self.logger.info(f"{'='*80}")
            
            # Apply editing
            try:
                if self.config.method == "emmet":
                    # Pass replay parameters if enabled
                    if self.config.replay_rate > 0:
                        edited_model, orig_weights, edit_distances = apply_emmet_to_model(
                            model, tokenizer, batch_requests, hparams,
                            copy=False, return_orig_weights=True,
                            use_replay=True,
                            replay_rate=self.config.replay_rate,
                            replay_buffer_size=getattr(self.config, 'replay_buffer_size', 200),
                            replay_strategy=getattr(self.config, 'replay_strategy', 'random'),
                            replay_weight=getattr(self.config, 'replay_weight', 1.0)
                        )
                    else:
                        edited_model, orig_weights, edit_distances = apply_emmet_to_model(
                            model, tokenizer, batch_requests, hparams,
                            copy=False, return_orig_weights=True
                        )
                elif self.config.method == "memit":
                    edited_model, orig_weights = apply_memit_to_model(
                        model, tokenizer, batch_requests, hparams,
                        copy=False, return_orig_weights=True
                    )
                    edit_distances = {}
                elif self.config.method == "rome":
                    edited_model, orig_weights = apply_rome_to_model(
                        model, tokenizer, batch_requests, hparams,
                        copy=False, return_orig_weights=True
                    )
                    edit_distances = {}
                
                self.logger.info(f"✅ Batch editing completed successfully")
                
                # Store batch results
                batch_result = {
                    "batch_idx": batch_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "num_edits": len(batch_requests),
                    "subjects": [req["subject"] for req in batch_requests],
                    "edit_distances": edit_distances,
                    "status": "success"
                }
                all_results.append(batch_result)

                # If native LoRA, append per-layer events for analysis
                try:
                    if getattr(self.config, 'edit_mode', 'raw') == 'lora_native' and isinstance(edit_distances, dict):
                        with open(events_fp, 'a', encoding='utf-8') as f:
                            for layer_key, info in edit_distances.items():
                                if isinstance(info, dict):
                                    event = {
                                        "run_dir": str(self.config.run_dir),
                                        "batch_idx": batch_idx,
                                        "layer": layer_key,
                                        "weight_name": info.get("weight_name"),
                                        "delta_norm": info.get("delta_norm"),
                                        "lora_residual_rel": info.get("lora_residual_rel"),
                                        "lora_fallback": info.get("lora_fallback"),
                                        "lora_fallback_reason": info.get("lora_fallback_reason"),
                                        "lora_rank": getattr(self.config, 'lora_rank', None),
                                        "lora_alpha": getattr(self.config, 'lora_alpha', None),
                                        "lora_scale": getattr(self.config, 'lora_scale', None),
                                        "lora_fit_steps": getattr(self.config, 'lora_fit_steps', None),
                                        "edit_mode": getattr(self.config, 'edit_mode', 'raw')
                                    }
                                    f.write(json.dumps(event) + "\n")
                except Exception:
                    pass

                # Trust events (for both raw and lora modes)
                try:
                    if isinstance(edit_distances, dict):
                        with open(trust_fp, 'a', encoding='utf-8') as ft:
                            for layer_key, info in edit_distances.items():
                                if isinstance(info, dict):
                                    tev = {
                                        "run_dir": str(self.config.run_dir),
                                        "batch_idx": batch_idx,
                                        "layer": layer_key,
                                        "weight_name": info.get("weight_name"),
                                        "delta_norm": info.get("delta_norm"),
                                        "trust_enable": info.get("trust_enable"),
                                        "trust_score": info.get("trust_score"),
                                        "trust_applied": info.get("trust_applied"),
                                        "trust_action": info.get("trust_action"),
                                        "trust_scale": info.get("trust_scale"),
                                    }
                                    ft.write(json.dumps(tev) + "\n")
                except Exception:
                    pass

                # Optional sequence metrics snapshot
                try:
                    cumulative_edits += len(batch_requests)
                    every = getattr(self.config, 'sequence_metrics_every', 0)
                    if every and cumulative_edits % every == 0:
                        # Sample up to 50 edited requests so far for quick eval
                        sample_requests = requests[:cumulative_edits]
                        if len(sample_requests) > 50:
                            import random as _rnd
                            _rnd.seed(self.config.seed + cumulative_edits)
                            sample_requests = _rnd.sample(sample_requests, 50)
                        es_list, ps_list, ns_list = [], [], []
                        for rq in sample_requests:
                            try:
                                subject = rq.get('subject', '')
                                target_new = rq.get('target_new', {}).get('str', '')
                                target_true = rq.get('target_true', {}).get('str', '')
                                rewrite_prompt = rq.get('prompt', '').format(subject)
                                es = self._test_prediction(model, tokenizer, rewrite_prompt, target_new, target_true)
                                if rq.get('paraphrase_prompts'):
                                    paraphrase_prompts = rq['paraphrase_prompts'][:3]
                                    ps_vals = [self._test_prediction(model, tokenizer, pp, target_new, target_true) for pp in paraphrase_prompts]
                                    ps = float(np.mean(ps_vals)) if ps_vals else 0.0
                                else:
                                    ps = 0.0
                                if rq.get('neighborhood_prompts'):
                                    neighbor_prompts = rq['neighborhood_prompts'][:3]
                                    ns_vals = [self._test_prediction(model, tokenizer, np_text, target_true, target_new) for np_text in neighbor_prompts]
                                    ns = float(np.mean(ns_vals)) if ns_vals else 1.0
                                else:
                                    ns = 1.0
                                es_list.append(es); ps_list.append(ps); ns_list.append(ns)
                            except Exception:
                                pass
                        snapshot = {
                            "cumulative_edits": cumulative_edits,
                            "batch_idx": batch_idx,
                            "es_mean": float(np.mean(es_list)) if es_list else 0.0,
                            "ps_mean": float(np.mean(ps_list)) if ps_list else 0.0,
                            "ns_mean": float(np.mean(ns_list)) if ns_list else 0.0,
                            "sample_size": len(sample_requests)
                        }
                        with open(seq_metrics_fp, 'a', encoding='utf-8') as fsm:
                            fsm.write(json.dumps(snapshot) + "\n")
                        self.logger.info(f"[SequenceMetrics] edits={cumulative_edits} ES={snapshot['es_mean']:.3f} PS={snapshot['ps_mean']:.3f} NS={snapshot['ns_mean']:.3f}")
                except Exception as _e_seq:
                    self.logger.debug(f"Sequence metrics snapshot failed: {_e_seq}")
                
            except Exception as e:
                self.logger.error(f"❌ Batch editing failed: {str(e)}", exc_info=True)
                batch_result = {
                    "batch_idx": batch_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "num_edits": len(batch_requests),
                    "status": "failed",
                    "error": str(e)
                }
                all_results.append(batch_result)
                # Continue to next batch rather than failing completely
                continue
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Editing completed: {len(all_results)} batches processed")
        self.logger.info(f"{'='*80}\n")
        
        # Post-hoc LoRA path (legacy). If native LoRA is enabled, skip this.
        if self.config.use_lora and getattr(self.config, "edit_mode", "raw") != "lora_native":
            self.logger.info(f"\n{'='*80}")
            self.logger.info("Applying LoRA to edited model...")
            self.logger.info(f"{'='*80}")
            
            try:
                from emmet.lora_wrapper import apply_lora_to_edited_model, get_lora_target_modules
                
                # Get recommended target modules for this model
                target_modules = get_lora_target_modules(self.config.model)
                self.logger.info(f"Target modules: {target_modules}")
                
                # Apply LoRA
                lora_wrapper = apply_lora_to_edited_model(
                    model=model,
                    target_modules=target_modules,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    freeze_base=True
                )
                
                # Get the LoRA-enhanced model
                model = lora_wrapper.model
                
                self.logger.info("✅ LoRA applied successfully")
                self.logger.info(f"{'='*80}\n")
                
            except Exception as e:
                self.logger.error(f"❌ LoRA application failed: {str(e)}", exc_info=True)
                self.logger.warning("Continuing with non-LoRA model...")
        
        return all_results, model, tokenizer
    
    def evaluate(self, model, tokenizer, requests, original_data):
        """Evaluate editing results - compute ES, PS, NS metrics"""
        self.logger.info("\n" + "="*80)
        self.logger.info("Evaluating editing results...")
        self.logger.info("="*80)
        
        model.eval()
        
        # Store detailed results for each example
        detailed_results = []
        
        # Aggregate metrics
        rewrite_success = []
        paraphrase_success = []
        neighborhood_success = []
        
        for idx, (request, record) in enumerate(zip(requests, original_data)):
            if idx % 50 == 0:
                self.logger.info(f"Evaluating example {idx+1}/{len(requests)}")
            
            try:
                # Extract test prompts
                subject = request["subject"]
                target_new = request["target_new"]["str"]
                target_true = request["target_true"]["str"]
                
                # Efficacy Score (ES): Test on rewrite prompt
                rewrite_prompt = request["prompt"].format(subject)
                es = self._test_prediction(model, tokenizer, rewrite_prompt, target_new, target_true)
                rewrite_success.append(es)
                
                # Paraphrase Score (PS): Test on paraphrase prompts
                paraphrase_prompts = request.get("paraphrase_prompts", [])
                if paraphrase_prompts:
                    ps_scores = [
                        self._test_prediction(model, tokenizer, pp, target_new, target_true)
                        for pp in paraphrase_prompts[:5]  # Limit to 5 to save time
                    ]
                    ps = np.mean(ps_scores) if ps_scores else 0.0
                    paraphrase_success.append(ps)
                else:
                    ps = 0.0
                
                # Neighborhood Score (NS): Test on neighborhood prompts
                neighborhood_prompts = request.get("neighborhood_prompts", [])
                if neighborhood_prompts:
                    ns_scores = [
                        self._test_prediction(model, tokenizer, np_text, target_true, target_new)
                        for np_text in neighborhood_prompts[:5]  # Should preserve original
                    ]
                    ns = np.mean(ns_scores) if ns_scores else 1.0
                    neighborhood_success.append(ns)
                else:
                    ns = 1.0
                
                detailed_results.append({
                    "example_id": idx,
                    "subject": subject,
                    "target_new": target_new,
                    "target_true": target_true,
                    "efficacy_score": float(es),
                    "paraphrase_score": float(ps),
                    "neighborhood_score": float(ns)
                })

                # If replay is enabled and strategy is priority, update buffer priority
                try:
                    if self.config.replay_rate > 0 and getattr(self.config, 'replay_strategy', 'random') == 'priority':
                        from emmet.emmet_replay import get_replay_buffer
                        buf = get_replay_buffer()
                        # Simple scheme: priority := 0.7*ES + 0.3*NS
                        pr = 0.7*float(es) + 0.3*float(ns)
                        _ = buf.update_priority_by_subject(subject, pr)
                except Exception:
                    pass
                
            except Exception as e:
                self.logger.warning(f"Evaluation failed for example {idx}: {str(e)}")
                detailed_results.append({
                    "example_id": idx,
                    "subject": request.get("subject", ""),
                    "error": str(e),
                    "efficacy_score": 0.0,
                    "paraphrase_score": 0.0,
                    "neighborhood_score": 0.0
                })
        
        # Compute aggregate metrics
        metrics = {
            "efficacy_success": float(np.mean(rewrite_success)) if rewrite_success else 0.0,
            "paraphrase_success": float(np.mean(paraphrase_success)) if paraphrase_success else 0.0,
            "neighborhood_specificity": float(np.mean(neighborhood_success)) if neighborhood_success else 0.0,
            "num_examples": len(requests),
            "num_evaluated": len(detailed_results)
        }
        
        # Composite score (S)
        metrics["composite_score"] = (
            metrics["efficacy_success"] + 
            metrics["paraphrase_success"] + 
            metrics["neighborhood_specificity"]
        ) / 3.0
        
        self.logger.info("\n" + "="*80)
        self.logger.info("EVALUATION RESULTS:")
        self.logger.info("="*80)
        self.logger.info(f"Efficacy Score (ES):           {metrics['efficacy_success']:.4f}")
        self.logger.info(f"Paraphrase Score (PS):         {metrics['paraphrase_success']:.4f}")
        self.logger.info(f"Neighborhood Specificity (NS): {metrics['neighborhood_specificity']:.4f}")
        self.logger.info(f"Composite Score (S):           {metrics['composite_score']:.4f}")
        self.logger.info("="*80 + "\n")
        
        return metrics, detailed_results
    
    def _test_prediction(self, model, tokenizer, prompt, target_new, target_true):
        """
        Test if model predicts target_new instead of target_true
        Returns 1.0 if correct (predicts target_new), 0.0 otherwise
        """
        try:
            # Prepare inputs
            prompt_new = f"{prompt} {target_new}"
            prompt_true = f"{prompt} {target_true}"
            
            # Tokenize
            inputs_new = tokenizer(prompt_new, return_tensors="pt").to(model.device)
            inputs_true = tokenizer(prompt_true, return_tensors="pt").to(model.device)
            
            # Get log probabilities
            with torch.no_grad():
                outputs_new = model(**inputs_new, labels=inputs_new["input_ids"])
                outputs_true = model(**inputs_true, labels=inputs_true["input_ids"])
                
                # Use negative loss as proxy for log probability
                prob_new = -outputs_new.loss.item()
                prob_true = -outputs_true.loss.item()
            
            # Return 1 if new target is more likely
            return 1.0 if prob_new > prob_true else 0.0
            
        except Exception as e:
            self.logger.debug(f"Prediction test failed: {str(e)}")
            return 0.0
    
    def save_results(self, edit_results, metrics, detailed_results):
        """Save results and metrics"""
        # Save editing results
        edit_results_file = self.config.run_dir / "edit_results.json"
        with open(edit_results_file, 'w', encoding='utf-8') as f:
            json.dump(edit_results, f, indent=2)
        
        # Save detailed evaluation results
        detailed_file = self.config.run_dir / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save aggregate metrics
        metrics_file = self.config.run_dir / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        # Save metrics CSV for easy aggregation
        csv_file = self.config.run_dir / "metrics.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("metric,value\n")
            for key, value in metrics.items():
                f.write(f"{key},{value}\n")
        
        # Save detailed results as CSV
        detailed_csv = self.config.run_dir / "detailed_results.csv"
        with open(detailed_csv, 'w', encoding='utf-8') as f:
            f.write("example_id,subject,target_new,target_true,efficacy_score,paraphrase_score,neighborhood_score\n")
            for result in detailed_results:
                if "error" not in result:
                    f.write(f"{result['example_id']},{result['subject']},"
                           f"{result.get('target_new', '')},"
                           f"{result.get('target_true', '')},"
                           f"{result['efficacy_score']},"
                           f"{result['paraphrase_score']},"
                           f"{result['neighborhood_score']}\n")
        
        self.logger.info(f"\n✅ Results saved to {self.config.run_dir}")
        self.logger.info(f"   - Edit results: {edit_results_file.name}")
        self.logger.info(f"   - Detailed results: {detailed_file.name}")
        self.logger.info(f"   - Metrics: {metrics_file.name}")
        self.logger.info(f"   - CSV files: metrics.csv, detailed_results.csv\n")
    
    def run(self):
        """Run complete experiment"""
        start_time = datetime.now()
        
        try:
            # Validate configuration
            self.config.validate()
            self.config.save_config()
            
            # Set random seeds for reproducibility
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            
            self.logger.info(f"Random seed set to: {self.config.seed}")
            
            # Load data and model
            requests, original_data = self.load_data()
            model, tokenizer = self.load_model()
            
            # Run editing
            edit_results, edited_model, tokenizer = self.run_editing(model, tokenizer, requests)
            
            # Evaluate
            metrics, detailed_results = self.evaluate(edited_model, tokenizer, requests, original_data)
            
            # Add timing info
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            metrics["duration_seconds"] = duration
            metrics["duration_formatted"] = str(end_time - start_time)
            
            # Save results
            self.save_results(edit_results, metrics, detailed_results)
            
            self.logger.info("\n" + "="*80)
            self.logger.info("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
            self.logger.info("="*80)
            self.logger.info(f"Total duration: {metrics['duration_formatted']}")
            self.logger.info(f"Results directory: {self.config.run_dir}")
            self.logger.info("="*80 + "\n")
            
            return True
            
        except Exception as e:
            self.logger.error("\n" + "="*80)
            self.logger.error("❌ EXPERIMENT FAILED")
            self.logger.error("="*80)
            self.logger.error(f"Error: {str(e)}", exc_info=True)
            self.logger.error("="*80 + "\n")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline experiments (ROME/MEMIT/EMMET)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--method", type=str, required=True,
                       choices=["rome", "memit", "emmet"],
                       help="Editing method")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (gpt2, gpt2-xl, llama3.2-3b)")
    
    # Optional arguments
    parser.add_argument("--num_edits", type=int, default=100,
                       help="Number of edits to perform")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for editing")
    parser.add_argument("--replay_rate", type=float, default=0.0,
                       help="Replay rate for memory replay (0.0 = no replay)")
    parser.add_argument("--replay_buffer_size", type=int, default=200,
                       help="Replay buffer max size")
    parser.add_argument("--replay_strategy", type=str, default="random",
                       choices=["random", "priority", "recent"],
                       help="Replay sampling strategy")
    parser.add_argument("--replay_weight", type=float, default=1.0,
                       help="Weight for replay samples relative to current batch (0.0-1.0)")
    parser.add_argument("--use_lora", action="store_true",
                       help="Apply LoRA after EMMET editing (post-hoc mode; prefer --edit_mode lora_native)")
    parser.add_argument("--edit_mode", type=str, default="raw",
                       choices=["raw", "lora_native"],
                       help="Editing application mode: raw updates or native LoRA overlays")
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank (number of low-rank dimensions)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                       help="LoRA alpha scaling factor")
    parser.add_argument("--lora_scale", type=float, default=1.0,
                       help="Additional scaling applied to ΔW before low-rank mapping")
    parser.add_argument("--lora_use_svd", dest="lora_use_svd", action="store_true",
                       help="Use SVD mapping for LoRA factors")
    parser.add_argument("--no_lora_use_svd", dest="lora_use_svd", action="store_false",
                       help="Disable SVD mapping for LoRA factors")
    parser.set_defaults(lora_use_svd=True)
    parser.add_argument("--lora_fit_steps", type=int, default=0,
                       help="Optional tiny fitting steps to refine LoRA factors")
    parser.add_argument("--allow_fallback", action="store_true",
                       help="Allow residual-guard fallback to raw updates when mapping is poor or fails")
    parser.add_argument("--lora_residual_threshold", type=float, default=None,
                       help="Residual threshold to trigger fallback; if omitted, only failures trigger fallback")
    parser.add_argument("--dataset", type=str, default="counterfact_sampled_unique_cf_10_20000",
                       help="Dataset name (without .json extension)")
    parser.add_argument("--output_dir", type=str, default="results/baseline",
                       help="Output directory for results")
    parser.add_argument("--sequence_metrics_every", type=int, default=0,
                       help="Collect sequence ES/PS/NS snapshot every N cumulative edits (0=disable)")
    # Trust / Rollback (Phase 4)
    parser.add_argument("--trust_enable", action="store_true", help="Enable trust/rollback mechanism")
    parser.add_argument("--trust_threshold", type=float, default=0.3, help="Threshold in [0,1] below which apply rollback/scale")
    parser.add_argument("--trust_action", type=str, default="rollback", choices=["rollback","scale"], help="Action when trust below threshold")
    parser.add_argument("--trust_scale", type=float, default=0.5, help="Scale factor when trust_action=scale")
    parser.add_argument("--trust_heldout_samples", type=int, default=0, help="Reserved for future held-out eval quick check")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 <= args.replay_rate <= 1:
        parser.error("replay_rate must be between 0.0 and 1.0")
    
    # Print experiment configuration
    print("="*80)
    print("EMMET Baseline Experiment Runner")
    print("="*80)
    print(f"Method:       {args.method.upper()}")
    print(f"Model:        {args.model}")
    print(f"Num edits:    {args.num_edits}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Replay rate:  {args.replay_rate}")
    if args.replay_rate > 0:
        print(f"Replay buf:   size={args.replay_buffer_size}, strategy={args.replay_strategy}, weight={args.replay_weight}")
    print(f"Use LoRA:     {args.use_lora}")
    print(f"Edit mode:    {args.edit_mode}")
    if args.use_lora:
        print(f"LoRA rank:    {args.lora_rank}")
        print(f"LoRA alpha:   {args.lora_alpha}")
    if args.edit_mode == "lora_native":
        print(f"LoRA-native:  rank={args.lora_rank}, alpha={args.lora_alpha}, scale={args.lora_scale}, use_svd={args.lora_use_svd}, fit_steps={args.lora_fit_steps}, allow_fallback={args.allow_fallback}, residual_thr={args.lora_residual_threshold}")
    print(f"Seed:         {args.seed}")
    print(f"Dataset:      {args.dataset}")
    print(f"Output dir:   {args.output_dir}")
    if args.sequence_metrics_every:
        print(f"Seq metrics:  every {args.sequence_metrics_every} edits")
    if args.trust_enable:
        print(f"Trust:        enabled (thr={args.trust_threshold}, action={args.trust_action}, scale={args.trust_scale}, heldout={args.trust_heldout_samples})")
    print("="*80 + "\n")
    
    # Create and run experiment
    config = ExperimentConfig(args)
    runner = BaselineRunner(config)
    success = runner.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
