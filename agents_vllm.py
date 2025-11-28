import datetime as dt
from typing import Optional, Dict, Any
import hashlib
import torch
import numpy as np
from database import Database
import pandas as pd
import os
import gc
from functools import lru_cache
from logger import setup_logger
from vllm import LLM, SamplingParams
from config import VLLM_CONFIG, FORECAST_MODEL_PATHS_VLLM, DEFAULT_MODEL_VLLM

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
        self.max_cache_size = VLLM_CONFIG.get("MAX_LOGPROB_CACHE", 128)

    def _load_llm_engine(self, model_path: str, is_forecast_model: bool = False) -> LLM:
        logger.info(f"Loading vLLM engine for model: {model_path} with TP={self.tensor_parallel_size}")
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
                self._forecast_vocab_size = llm.get_tokenizer().vocab_size
                self.eos_token_id = llm.get_tokenizer().eos_token_id
                logger.info(f"Forecast model vocab size: {self._forecast_vocab_size}, EOS ID: {self.eos_token_id}")
            logger.info(f"Successfully loaded vLLM engine for {model_path}")
            return llm
        except Exception as e:
            logger.error(f"Failed to load vLLM engine for '{model_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load vLLM engine '{model_path}': {e}") from e

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

    def next_token_probs(self, prompt_string: str, llm_engine: LLM, cache: Dict[str, torch.Tensor], max_cache_size: Optional[int] = None) -> torch.Tensor:
        prompt_bytes = prompt_string.encode('utf-8')
        prompt_key = hashlib.sha256(prompt_bytes).hexdigest()

        if prompt_key in cache:
            return cache[prompt_key]

        if self._forecast_vocab_size is None: # Should be set when forecast_llm_engine is loaded
             tokenizer = llm_engine.get_tokenizer()
             self._forecast_vocab_size = tokenizer.vocab_size

        sampling_params = SamplingParams(
            temperature=0.0, # Deterministic for getting actual probabilities
            top_p=1.0,
            max_new_tokens=1,
            logprobs=self._forecast_vocab_size # Request logprobs for all tokens
        )

        try:
            # Prepare single-prompt generation for probability extraction
            request_id = f"next_token_probs_{prompt_key}" # Unique request ID
            outputs = llm_engine.generate([prompt_string], sampling_params, request_id=request_id)
        except Exception as e:
            logger.error(f"vLLM generation failed for next_token_probs: {e}", exc_info=True)
            # Fallback to a uniform distribution or raise error
            # For now, let's raise to make issues visible
            raise RuntimeError(f"vLLM generation error in next_token_probs: {e}")


        if not outputs or not outputs[0].outputs:
            logger.error("vLLM returned no output for next_token_probs.")
            raise ValueError("vLLM returned no output in next_token_probs.")

        # Extract log probabilities for the first generated token
        logprobs_at_first_step = outputs[0].outputs[0].logprobs[0]

        # Initialize log-probability tensor on CPU
        log_probs_tensor = torch.full((self._forecast_vocab_size,), -float('inf'), dtype=torch.float32, device='cpu')

        for token_id, logprob_obj in logprobs_at_first_step.items():
            log_probs_tensor[token_id] = logprob_obj.logprob
        
        # Compute normalized token probabilities via softmax
        probs_tensor = torch.softmax(log_probs_tensor, dim=-1)
        
        cache[prompt_key] = probs_tensor
        # Simple FIFO eviction to avoid unbounded growth
        limit = max_cache_size or self.max_cache_size
        if limit and len(cache) > limit:
            first_key = next(iter(cache))
            cache.pop(first_key, None)
        return probs_tensor

    def forecast(self, query: str, context: Optional[str] = None) -> pd.DataFrame:
        self.start_forecast()
        if self._forecast_llm_engine is None or self.eos_token_id is None or self._forecast_vocab_size is None:
            raise RuntimeError("Forecast model not loaded properly before forecast call.")

        usable_context = context[:self.context_limit] if context else ''
        logic_str = ''
        if self.use_logic:
            logic_str = self.generate_logic(query=query, context=usable_context)
            logger.info(f'Logic:\n{logic_str}')

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
            f"Chain of thought: {logic_str}\n\n"     
            f"Knowledge: {usable_context}\n\n"
            f"Query: {query}\n\n"
            "Now, provide your prediction:\n"
            "Answer: "
        )

        forecast_tokenizer = self._forecast_llm_engine.get_tokenizer()

        logger.info(f'Model ({self.forecast_model_path}) running forecast...')

        cache: Dict[str, torch.Tensor] = {}
        number_probs: Dict[int, float] = {}

        single_token_integer_generation = self.assure_agent()

        if single_token_integer_generation:
            logger.info('Using single token approach for number probabilities...')
            # Compute probabilities assuming one token per number
            probs_after_prompt = self.next_token_probs(prompt_prefix, self._forecast_llm_engine, cache)
            
            for num in range(101):
                num_str = str(num)
                try:
                    # Encode number as single token
                    token_ids = forecast_tokenizer.encode(num_str, add_special_tokens=False)
                    if len(token_ids) == 1:
                        token_id = token_ids[0]
                        number_probs[num] = probs_after_prompt[token_id].item()
                    else:
                        logger.warning(f"Number {num_str} tokenized to multiple tokens ({token_ids}) in single-token mode. Assigning 0 prob.")
                        number_probs[num] = 0.0 
                except Exception as e:
                    logger.error(f"Error tokenizing or getting prob for number {num_str} in single-token mode: {e}")
                    number_probs[num] = 0.0
        
        else:
            logger.info('Using multi-token approach for number probabilities...')
            for num in range(101):
                candidate_str = str(num)
                # Encode multi-token number candidates
                candidate_token_ids = forecast_tokenizer.encode(candidate_str, add_special_tokens=False)
                
                current_log_prob = 0.0
                current_prompt_context = prompt_prefix

                # Accumulate log-probabilities for each digit token
                for i, token_id in enumerate(candidate_token_ids):
                    probs_dist = self.next_token_probs(current_prompt_context, self._forecast_llm_engine, cache)
                    token_log_prob = torch.log(probs_dist[token_id] + 1e-12).item()
                    current_log_prob += token_log_prob
                    
                    token_str = forecast_tokenizer.decode([token_id])
                    current_prompt_context += token_str

                # Include terminator probability (EOS or '%')
                probs_after_number = self.next_token_probs(current_prompt_context, self._forecast_llm_engine, cache)
                
                eos_prob = probs_after_number[self.eos_token_id].item()
                
                percent_token_id = forecast_tokenizer.convert_tokens_to_ids('%')
                percent_prob = 0.0
                if percent_token_id != forecast_tokenizer.unk_token_id:
                     percent_prob = probs_after_number[percent_token_id].item()

                # Combine terminator probabilities
                terminator_log_prob = torch.log(torch.tensor(eos_prob + percent_prob + 1e-12)).item()
                current_log_prob += terminator_log_prob
                
                number_probs[num] = np.exp(current_log_prob)
                if num % 10 == 0 and self.verbose:
                    logger.info(f'Multi-token: P({num}) = {np.exp(current_log_prob):.4e}')

        total_prob = sum(number_probs.values())
        logger.info(f'Sum of probabilities for 0-100: {total_prob:.4f}. Normalizing...')
        if total_prob > 0:
            normalized_probs = {num: prob / total_prob for num, prob in number_probs.items()}
        else:
            logger.warning("Total probability sum is 0. Cannot normalize. Returning uniform distribution.")
            # Default to uniform distribution on zero-sum
            uniform_prob = 1.0 / 101
            normalized_probs = {num: uniform_prob for num in range(101)}

        data = pd.DataFrame.from_dict(normalized_probs, orient="index", columns=["probs"])
        data.index.name = "value"
        return data

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
            # Stream logic generation iteratively
            results_stream = self._logic_llm_engine.generate([prompt_string], sampling_params, request_id=request_id, stream=True)
            for request_output in results_stream:
                current_total_text = request_output.outputs[0].text
                new_text_chunk = current_total_text[len(full_generated_text):]
                
                if self.verbose:
                    print(new_text_chunk, end='', flush=True)
                logic_chunks.append(new_text_chunk)
                full_generated_text = current_total_text
        except Exception as e:
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

    @lru_cache(maxsize=1)
    def assure_agent(self) -> bool:
        """Check if tokenizer encodes numbers 0–100 as single tokens."""
        if self._forecast_llm_engine is None:
            logger.warning("Forecast model not loaded, cannot perform assure_agent check.")
            raise RuntimeError("Forecast model not loaded.")
            
        tokenizer = self._forecast_llm_engine.get_tokenizer()
        vocab = tokenizer.get_vocab() # Returns Dict[str, int] (token_str to token_id)

        integer_token_strings = [str(i) for i in range(101)]
        
        all_single_tokens = True
        for num_str in integer_token_strings:
            # Check 1: Is the number string itself a single token in the vocab?
            if num_str not in vocab:
                all_single_tokens = False
                if self.verbose:
                    logger.info(f"Number '{num_str}' is not a direct single token in vocab.")
                break 
            
            # Check 2: Does encoding the number string result in a single token ID corresponding to that string?
            # (This is a stronger check, as some tokenizers might have '10' in vocab but encode "10" as ['1','0'])
            try:
                encoded_ids = tokenizer.encode(num_str, add_special_tokens=False)
                if len(encoded_ids) != 1:
                    all_single_tokens = False
                    if self.verbose:
                        logger.info(f"Number '{num_str}' encodes to multiple tokens: {encoded_ids}.")
                    break
                # Optional: verify that this single token decodes back to num_str if needed,
                # but len(encoded_ids) == 1 and num_str in vocab should be sufficient.
            except Exception as e:
                logger.error(f"Error encoding '{num_str}' with tokenizer: {e}")
                all_single_tokens = False # Treat error as failure for single-token assumption
                break
        
        if all_single_tokens:
            logger.info("Tokenizer represents all numbers 0-100 as single tokens.")
        else:
            logger.info("Tokenizer does NOT represent all numbers 0-100 as single tokens. Multi-token approach will be used.")
        return all_single_tokens
