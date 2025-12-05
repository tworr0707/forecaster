import datetime as dt
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import torch
import numpy as np
from database import Database
import pandas as pd
import os
import gc
import math
from logger import setup_logger
from vllm import LLM, SamplingParams

from config import VLLM_CONFIG, FORECAST_MODEL_PATHS_VLLM, DEFAULT_MODEL_VLLM, NUMBER_LOGPROB_TOP_K, CONTEXT_TOKENS
from oom_utils import is_oom_error, wrap_oom, ForecastingOOMError

logger = setup_logger(__name__)

logger.info("CUDA available: %s", torch.cuda.is_available())

class Agent:
    """Agent for forecasting and optional logic generation using vLLM."""
    def __init__(
        self,
        model: str = DEFAULT_MODEL_VLLM,
        verbose: bool = False,
        # vLLM parallelism and memory controls
        tensor_parallel_size: Optional[int] = None,
        pipeline_parallel_size: Optional[int] = None,
        distributed_executor_backend: Optional[str] = None,
        gpu_memory_utilization: Optional[float] = None,
        swap_space_gb: Optional[float] = None,
        max_model_len: Optional[int] = None,
    ):
        self.verbose = verbose
        self.db = Database()

        self.context_limit = CONTEXT_TOKENS  # Max tokens for prompt input
        self.char_limit = int(self.context_limit * 4)  # rough chars per token heuristic for slicing

        if model in FORECAST_MODEL_PATHS_VLLM:
            self.forecast_model_path = FORECAST_MODEL_PATHS_VLLM[model]
        else:
            raise ValueError(f'Unknown model: {model}.')

        # Resolve parallelism/memory settings (config-driven, optional overrides via args)
        auto_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.tensor_parallel_size = tensor_parallel_size or VLLM_CONFIG.get("tensor_parallel_size") or auto_gpus
        self.pipeline_parallel_size = pipeline_parallel_size or VLLM_CONFIG.get("pipeline_parallel_size", 1)
        self.distributed_executor_backend = (
            distributed_executor_backend
            or VLLM_CONFIG.get("distributed_executor_backend")
            or "mp"
        )
        self.gpu_memory_utilization = gpu_memory_utilization or VLLM_CONFIG.get("gpu_memory_utilization", 0.90)
        self.swap_space_gb = swap_space_gb if swap_space_gb is not None else VLLM_CONFIG.get("swap_space_gb")
        self.max_model_len = max_model_len if max_model_len is not None else VLLM_CONFIG.get("max_model_len")
        self.load_format = VLLM_CONFIG.get("load_format")
        # copy to avoid mutating global config dict
        self.model_loader_extra_config = dict(VLLM_CONFIG.get("model_loader_extra_config", {}))

        logger.info(
            "vLLM parallelism => TP=%s, PP=%s, backend=%s, gpu_mem_util=%.2f, swap=%s, max_len=%s",
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
            self.distributed_executor_backend,
            self.gpu_memory_utilization,
            self.swap_space_gb,
            self.max_model_len,
        )

        # Placeholder for forecast LLM engine
        self._forecast_llm_engine: Optional[LLM] = None
        self._forecast_vocab_size: Optional[int] = None
        self.eos_token_id: Optional[int] = None # For the forecast model
        self._number_token_variants: Dict[int, List[List[int]]] = {}
        self._percent_token_id: Optional[int] = None
        self.single_token_mode: bool = True
        self.number_logprob_top_k = NUMBER_LOGPROB_TOP_K
        if _vllm_import_error:
            raise ImportError(
                "vLLM is required for Agent but is not installed or failed to import: %s"
                % _vllm_import_error
            )

    def _load_llm_engine(self, model_path: str, is_forecast_model: bool = False) -> LLM:
        logger.info(
            "Loading vLLM engine for model=%s TP=%s PP=%s gpu_mem_util=%.2f",
            model_path,
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
            self.gpu_memory_utilization,
        )
        try:
            llm_kwargs = dict(
                model=model_path,
                tokenizer=model_path,  # Usually same as model
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                distributed_executor_backend=self.distributed_executor_backend,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )

            if self.load_format:
                llm_kwargs["load_format"] = self.load_format
                if self.load_format == "runai_streamer":
                    import importlib
                    if importlib.util.find_spec("runai_model_streamer") is None:
                        msg = (
                            "runai_model_streamer not found. "
                            "Install vllm[runai] and runai-model-streamer, "
                            "or set load_format=None to use the default loader."
                        )
                        logger.error(msg)
                        raise ImportError(msg)
            if self.model_loader_extra_config:
                llm_kwargs["model_loader_extra_config"] = self.model_loader_extra_config
            if self.swap_space_gb is not None:
                llm_kwargs["swap_space"] = self.swap_space_gb
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = self.max_model_len
            download_dir = os.environ.get("HF_HOME")
            if download_dir:
                llm_kwargs["download_dir"] = download_dir

            logger.info("vLLM load args: %s", llm_kwargs)
            llm = LLM(**llm_kwargs)
            if is_forecast_model:
                tokenizer = llm.get_tokenizer()
                self._forecast_vocab_size = tokenizer.vocab_size
                self.eos_token_id = tokenizer.eos_token_id
                logger.info(
                    "Forecast model vocab size: %s, EOS ID: %s", self._forecast_vocab_size, self.eos_token_id
                )
                self._precompute_number_tokens(tokenizer)
            logger.info(f"Successfully loaded vLLM engine for {model_path}")
            return llm
        except Exception as e:
            if is_oom_error(e):
                logger.error("CUDA OOM while loading model %s", model_path, exc_info=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise wrap_oom(e)
            logger.error(f"Failed to load vLLM engine for '{model_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load vLLM engine '{model_path}': {e}") from e

    def _precompute_number_tokens(self, tokenizer) -> None:
        """Precompute token ids for 0-100 and simple punctuation to avoid hot-path tokenization."""
        self._number_token_variants.clear()
        self._percent_token_id = None

        pct_ids = tokenizer.encode("%", add_special_tokens=False) or []
            
        if pct_ids:
            self._percent_token_id = pct_ids[0]

        empty_encodings = []
        multi_token = 0
        for num in range(101):
            variants: List[List[int]] = []
            for s in (str(num), " " + str(num)):
                ids = tokenizer.encode(s, add_special_tokens=False) or []
                if ids:
                    variants.append(ids)
            if not variants:
                empty_encodings.append(num)
                continue
            if any(len(v) > 1 for v in variants):
                multi_token += 1
            self._number_token_variants[num] = variants

        if empty_encodings:
            logger.warning("Tokenizer returned empty ids for numbers (no-space or leading-space): %s", empty_encodings)

        self.single_token_mode = multi_token == 0 and not empty_encodings
        if not self.single_token_mode:
            logger.warning(
                "Multi-token number regime detected for %d numbers; using multi-step probability path.",
                multi_token,
            )

        logger.info(
            "Precomputed number tokens (single_token_mode=%s, percent_token=%s)",
            self.single_token_mode,
            self._percent_token_id,
        )

    def start_forecast(self) -> None:
        if self._forecast_llm_engine is None:
            self._forecast_llm_engine = self._load_llm_engine(self.forecast_model_path, is_forecast_model=True)

    def stop_forecast(self) -> None:
        if self._forecast_llm_engine is not None:
            logger.info(f'Unloading forecast vLLM engine: {self.forecast_model_path}')
            del self._forecast_llm_engine
            self._forecast_llm_engine = None
            self._forecast_vocab_size = None
            self.eos_token_id = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def forecast(self, query: str, context: Optional[str] = None, logic_str: Optional[str] = None) -> pd.DataFrame:
        self.start_forecast()
        if self._forecast_llm_engine is None or self.eos_token_id is None or self._forecast_vocab_size is None:
            raise RuntimeError("Forecast model not loaded properly before forecast call.")

        usable_context = context[: self.char_limit] if context else ''
        if usable_context:
            tok = self._forecast_llm_engine.get_tokenizer()
            ctx_tokens = tok.encode(usable_context, add_special_tokens=False)
            if len(ctx_tokens) > self.context_limit:
                ctx_tokens = ctx_tokens[: self.context_limit]
                usable_context = tok.decode(ctx_tokens)
        context_line = f"Knowledge: {usable_context}\n\n" if len(usable_context) > 0 else ''
        logic_text = logic_str or ''
        logic_line = f"Argument: {logic_text}\n\n" if logic_text.strip() else ''

        prompt_prefix = (
            "You are an expert AI superforecaster, trained to predict the future using all available knowledge. "
            f"The current date is {dt.datetime.now().strftime('%Y-%m-%d')}. "
            "Your task is to provide the most accurate prediction based on the given query and knowledge. "
            "You MUST follow these rules carefully:\n\n"
            "1. Use all available knowledge as context to inform your prediction.\n"
            "2. Answer with a single numerical percentage between 0 and 100.\n"
            "3. Do NOT include any additional text, symbols, or explanations in your response.\n\n"
            "For example:\n"
            "If the prediction is 50%, simply respond with:\n"
            "50\n\n"    
            f"{logic_line}"     
            f"{context_line}"
            f"Query: {query}\n\n"
            "Now provide your prediction:\n"
            "Answer: "
        )

        forecast_tokenizer = self._forecast_llm_engine.get_tokenizer()

        logger.info(f'Model ({self.forecast_model_path}) running forecast...')

        try:
            if self.single_token_mode:
                probs = self._number_distribution_single_token(prompt_prefix)
            else:
                probs = self._number_distribution_multi_token(prompt_prefix)
        except ForecastingOOMError:
            raise
        except Exception as e:
            logger.error("Failed to compute number distribution: %s", e, exc_info=True)
            raise

        probs_series = pd.Series(probs)
        probs_series.index = range(101)
        df = pd.DataFrame({'probs': probs_series})
        return df

    def _get_logprob_dict(self, prompt: str) -> Dict[int, float]:
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=1,
            logprobs=self.number_logprob_top_k,
        )
        try:
            outputs = self._forecast_llm_engine.generate([prompt], params)
        except Exception as e:
            if is_oom_error(e):
                raise wrap_oom(e)
            raise
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no outputs for logprob step")
        lp_dict = outputs[0].outputs[0].logprobs[0]
        return {int(k): v.logprob for k, v in lp_dict.items()}

    def _get_logprob_batch(self, prompts: List[str]) -> List[Dict[int, float]]:
        if not prompts:
            return []
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=1,
            logprobs=self.number_logprob_top_k,
        )
        try:
            outputs = self._forecast_llm_engine.generate(prompts, params)
        except Exception as e:
            if is_oom_error(e):
                raise wrap_oom(e)
            raise
        if not outputs:
            raise RuntimeError("vLLM returned no outputs for batch logprob step")
        result: List[Dict[int, float]] = []
        for out in outputs:
            if not out.outputs:
                result.append({})
                continue
            lp_dict = out.outputs[0].logprobs[0]
            result.append({int(k): v.logprob for k, v in lp_dict.items()})
        return result

    def _number_distribution_single_token(self, prompt: str) -> List[float]:
        """Approximate distribution over 0–100 using one-step top-K logprobs.

        vLLM returns only the top-K logprobs; any missing token is assigned a
        small floor probability (1e-12) before renormalising across 101 values,
        so every integer 0–100 retains non-zero mass.
        """
        logprob_dict = self._get_logprob_dict(prompt)
        log_floor = math.log(1e-12)
        logs: List[float] = []
        for num in range(101):
            variants = self._number_token_variants.get(num, [])
            variant_lps = []
            for ids in variants:
                if len(ids) != 1:
                    continue
                lp = logprob_dict.get(ids[0], log_floor)
                variant_lps.append(lp)
            if variant_lps:
                logs.append(self._logsumexp(variant_lps))
            else:
                logs.append(log_floor)
        return self._normalize_log_probs(logs)

    def _number_distribution_multi_token(self, prompt: str) -> List[float]:
        """Multi-token numbers: accumulate top-K logprobs per token + terminator.

        Missing tokens/terminators receive a log-floor (1e-12) before
        renormalisation across 0–100 so mass is conserved in the 101-point grid.
        """
        tokenizer = self._forecast_llm_engine.get_tokenizer()
        log_floor = math.log(1e-12)
        log_floor = math.log(1e-12)
        # Each item: (num, variant_tokens, context_str, acc_logprob)
        active = []
        for num, variants in self._number_token_variants.items():
            for v in variants:
                if not v:
                    continue
                active.append({"num": num, "tokens": v, "ctx": prompt, "lp": 0.0})
        if not active:
            return [1/101.0 for _ in range(101)]

        # Step through tokens positionally, batching logprob calls per step
        max_len = max(len(a["tokens"]) for a in active)
        for pos in range(max_len):
            prompts = []
            idx_map = []
            for idx, item in enumerate(active):
                if pos >= len(item["tokens"]):
                    continue
                prompts.append(item["ctx"])
                idx_map.append(idx)
            if not prompts:
                continue
            lp_batch = self._get_logprob_batch(prompts)
            for lp_dict, a_idx in zip(lp_batch, idx_map):
                tid = active[a_idx]["tokens"][pos]
                lp = lp_dict.get(tid, log_floor)
                active[a_idx]["lp"] += lp
                active[a_idx]["ctx"] += tokenizer.decode([tid])

        # Terminator step
        prompts = [item["ctx"] for item in active]
        term_lp_batch = self._get_logprob_batch(prompts) if prompts else []
        for lp_dict, item in zip(term_lp_batch, active):
            terminators = []
            if self.eos_token_id is not None:
                terminators.append(lp_dict.get(self.eos_token_id, log_floor))
            if self._percent_token_id is not None:
                terminators.append(lp_dict.get(self._percent_token_id, log_floor))
            if terminators:
                item["lp"] += self._logsumexp(terminators)

        # Aggregate logprobs per number
        num_to_lps: Dict[int, List[float]] = {n: [] for n in range(101)}
        for item in active:
            num_to_lps[item["num"]].append(item["lp"])

        probs = []
        for num in range(101):
            lps = num_to_lps.get(num, [])
            if lps:
                probs.append(self._logsumexp(lps))
            else:
                probs.append(log_floor)
        return self._normalize_log_probs(probs)

    @staticmethod
    def _logsumexp(vals: List[float]) -> float:
        if not vals:
            return float("-inf")
        t = torch.tensor(vals, dtype=torch.float32)
        return float(torch.logsumexp(t, dim=0))

    def _normalize_log_probs(self, log_probs: List[float]) -> List[float]:
        m = max(log_probs)
        exp_probs = [math.exp(lp - m) for lp in log_probs]
        s = sum(exp_probs)
        if s == 0:
            logger.warning("Log-prob normalization sum is zero; returning uniform distribution")
            return [1/101.0 for _ in log_probs]
        return [p / s for p in exp_probs]

    # Logic generation removed: logic is provided externally via LogicClient in Ensemble.
