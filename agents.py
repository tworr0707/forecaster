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
from config import VLLM_CONFIG, FORECAST_MODEL_PATHS_VLLM, DEFAULT_MODEL_VLLM, NUMBER_LOGPROB_TOP_K
from oom_utils import is_oom_error, wrap_oom, ForecastingOOMError

logger = setup_logger(__name__)

logger.info("CUDA available: %s", torch.cuda.is_available())

class Agent:
    """Agent for forecasting and optional logic generation using vLLM."""
    def __init__(
        self,
        model: str = DEFAULT_MODEL_VLLM,
        use_logic: bool = False,
        logic_model_path: Optional[str] = 'Qwen/QwQ-32B', 
        verbose: bool = False,
        # vLLM parallelism and memory controls
        tensor_parallel_size: Optional[int] = None,
        pipeline_parallel_size: Optional[int] = None,
        distributed_executor_backend: Optional[str] = None,
        gpu_memory_utilization: Optional[float] = None,
        swap_space_gb: Optional[float] = None,
        max_model_len: Optional[int] = None,
    ):
        self.use_logic = use_logic
        self.verbose = verbose
        self.db = Database()

        self.context_limit = 30000 # Max tokens for prompt input / logic generation output

        if model in FORECAST_MODEL_PATHS_VLLM:
            self.forecast_model_path = FORECAST_MODEL_PATHS_VLLM[model]
        else:
            raise ValueError(f'Unknown model: {model}.')

        self.logic_model_path = logic_model_path
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

        # Placeholders for forecast and logic LLM engines
        self._forecast_llm_engine: Optional[LLM] = None
        self._logic_llm_engine: Optional[LLM] = None
        self._forecast_vocab_size: Optional[int] = None
        self.eos_token_id: Optional[int] = None # For the forecast model
        self._number_token_map: Dict[int, List[int]] = {}
        self._percent_token_id: Optional[int] = None
        self.single_token_mode: bool = True
        self.number_logprob_top_k = NUMBER_LOGPROB_TOP_K

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
        self._number_token_map.clear()
        self._percent_token_id = None

        pct_ids = tokenizer.encode("%", add_special_tokens=False) or []
            
        if pct_ids:
            self._percent_token_id = pct_ids[0]

        empty_encodings = []
        multi_token = 0
        for num in range(101):
            ids = tokenizer.encode(str(num), add_special_tokens=False) or []
            if not ids:
                empty_encodings.append(num)
                continue
            if len(ids) > 1:
                multi_token += 1
            self._number_token_map[num] = ids

        if empty_encodings:
            logger.warning("Tokenizer returned empty ids for numbers: %s", empty_encodings)

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
        if self._logic_llm_engine is not None:
            self.stop_logic()
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

    def start_logic(self) -> None:
        if self._forecast_llm_engine is not None:
            self.stop_forecast()
        if self._logic_llm_engine is None:
            if self.logic_model_path is None:
                raise ValueError("Logic model path is not set, but logic generation was requested.")
            self._logic_llm_engine = self._load_llm_engine(self.logic_model_path)
    
    def stop_logic(self) -> None:
        if self._logic_llm_engine is not None:
            logger.info(f'Unloading logic vLLM engine: {self.logic_model_path}')
            del self._logic_llm_engine
            self._logic_llm_engine = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def forecast(self, query: str, context: Optional[str] = None, logic_str: Optional[str] = None) -> pd.DataFrame:
        self.start_forecast()
        if self._forecast_llm_engine is None or self.eos_token_id is None or self._forecast_vocab_size is None:
            raise RuntimeError("Forecast model not loaded properly before forecast call.")

        usable_context = context[:self.context_limit] if context else ''
        logic_text = logic_str if (self.use_logic and logic_str) else ''

        prompt_prefix = (
            "You are an expert AI superforecaster, trained to predict the future using all available knowledge. "
            f"The current date is {dt.datetime.now().strftime('%Y-%m-%d')}. "
            "Your task is to provide the most accurate prediction based on the given query and knowledge. "
            "Follow these rules carefully:\n\n"
            "1. Use all available knowledge as context to inform your prediction.\n"
            "2. Answer with a single numerical percentage between 0 and 100.\n"
            "3. Do not include any additional text, symbols, or explanations in your response.\n\n"
            "For example:\n"
            "If the prediction is 75%, simply respond with:\n"
            "75\n\n"    
            f"Chain of thought: {logic_text}\n\n"     
            f"Knowledge: {usable_context}\n\n"
            f"Query: {query}\n\n"
            "Now, provide your prediction:\n"
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
            token_ids = self._number_token_map.get(num, [])
            if not token_ids:
                logs.append(log_floor)
                continue
            token_id = token_ids[0]
            lp = logprob_dict.get(token_id, log_floor)
            if lp == log_floor:
                logger.debug("Number %d not in top-K logprobs; using floor", num)
            logs.append(lp)
        return self._normalize_log_probs(logs)

    def _number_distribution_multi_token(self, prompt: str) -> List[float]:
        """Multi-token numbers: accumulate top-K logprobs per token + terminator.

        Missing tokens/terminators receive a log-floor (1e-12) before
        renormalisation across 0–100 so mass is conserved in the 101-point grid.
        """
        tokenizer = self._forecast_llm_engine.get_tokenizer()
        log_floor = math.log(1e-12)
        probs: List[float] = []
        for num in range(101):
            token_ids = self._number_token_map.get(num, [])
            if not token_ids:
                probs.append(log_floor)
                continue
            context = prompt
            total_lp = 0.0
            for tid in token_ids:
                step_lp = self._get_logprob_dict(context)
                lp = step_lp.get(tid, log_floor)
                total_lp += lp
                context += tokenizer.decode([tid])
            # terminator step combining EOS and optional %
            term_lp_dict = self._get_logprob_dict(context)
            terminators = []
            if self.eos_token_id is not None:
                terminators.append(term_lp_dict.get(self.eos_token_id, log_floor))
            if self._percent_token_id is not None:
                terminators.append(term_lp_dict.get(self._percent_token_id, log_floor))
            if terminators:
                total_lp += self._logsumexp(terminators)
            probs.append(total_lp)
        return self._normalize_log_probs(probs)

    @staticmethod
    def _logsumexp(vals: List[float]) -> float:
        m = max(vals)
        return m + math.log(sum(math.exp(v - m) for v in vals)) if vals else float('-inf')

    def _normalize_log_probs(self, log_probs: List[float]) -> List[float]:
        m = max(log_probs)
        exp_probs = [math.exp(lp - m) for lp in log_probs]
        s = sum(exp_probs)
        if s == 0:
            logger.warning("Log-prob normalization sum is zero; returning uniform distribution")
            return [1/101.0 for _ in log_probs]
        return [p / s for p in exp_probs]

    def generate_logic(self, query: str, context: Optional[str] = None) -> str:
        self.start_logic()
        if self._logic_llm_engine is None:
            raise RuntimeError("Logic model not loaded properly before generate_logic call.")

        usable_context = context[:self.context_limit] if context else ''
        
        try:
            base = os.path.dirname(__file__)
            sats_path = os.path.join(base, "sats.txt")
            if os.path.exists(sats_path):
                with open(sats_path) as f:
                    sats = f.read()
            else:
                logger.warning(f'sats.txt not found at {sats_path}. Proceeding without SATs guidelines.')
                sats = "Standard Analytical Techniques (SATs) guidelines not available."
        except Exception as e:
            logger.warning(f'Error reading sats.txt: {e}')
            sats = "Error loading SATs guidelines."

        messages = [
            {
                'role': 'system',
                'content': (
                    "You are an expert AI superforecaster, trained to predict the future using a large body of knowledge and data. "
                    f"The current date is {dt.datetime.now().strftime('%Y-%m-%d')}. "
                    'Your responses should be exceptionally detailed and elaborate. Use as many tokens as possible to fully explore every nuance of your analysis. '
                    'Do not hold back on any detail—explain every point thoroughly, provide multiple examples and perspectives, and ensure that your response is exhaustive. '
                    'Brevity is not a concern; your goal is to generate a maximally comprehensive and in-depth explanation. '
                    f'Approach each query methodically, ensuring your response is rigorous, objective, and unabridged. Specifically think through the SATs Guidelines:\n{sats}\n'
                    'Critically evaluate assumptions, consider multiple perspectives, and clearly identify any gaps in the available information. '
                    'Your analysis should be articulate, well-organised, and demonstrate a high standard of intellectual inquiry.'
                    "Focus solely on the underlying logic required to answer the query. IMPORTANT: Do NOT attempt to answer the query or suggest a likelihood."
                )
            },
            {
                'role': 'user',
                'content': f'Knowledge: {usable_context}.\nQuery: {query}'
            }
        ]
        
        logic_tokenizer = self._logic_llm_engine.get_tokenizer()
        try:
            # Fallback to manual prompt formatting if template API is unavailable
            prompt_string = logic_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Applying chat template failed: {e}. Falling back to manual prompt formatting.")
            system_prompt = messages[0]['content']
            user_prompt = messages[1]['content']
            prompt_string = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

        sampling_params = SamplingParams(
            max_tokens=self.context_limit,
            temperature=0.666,
            top_p=0.95,
            top_k=40,
            stop_token_ids=[logic_tokenizer.eos_token_id] if logic_tokenizer.eos_token_id is not None else [],
        )

        logger.info("Generating logic stream...")
        logic_chunks = []
        full_generated_text = ""
        request_id = f"logic_gen_{hash(prompt_string)}"

        try:
            results_stream = self._logic_llm_engine.generate([prompt_string], sampling_params, request_id=request_id, stream=True)
            for request_output in results_stream:
                current_total_text = request_output.outputs[0].text
                new_text_chunk = current_total_text[len(full_generated_text):]
                if self.verbose:
                    print(new_text_chunk, end='', flush=True)
                logic_chunks.append(new_text_chunk)
                full_generated_text = current_total_text
        except Exception as e:
            if is_oom_error(e):
                logger.error("CUDA OOM during logic generation", exc_info=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise wrap_oom(e)
            logger.error(f"Error during vLLM streaming generation for logic: {e}", exc_info=True)
            raise RuntimeError(f"vLLM streaming error in generate_logic: {e}")
        finally:
            # Ensure engines are in a known state
            self.stop_logic()
            self.start_forecast()

        if self.verbose:
            print()

        final_logic_text = "".join(logic_chunks).strip()
        
        self.stop_logic()
        self.start_forecast()
        return final_logic_text
