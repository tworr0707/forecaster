import datetime as dt
from typing import Optional, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import threading
import hashlib
import torch
import numpy as np
from database_torch import Database
import pandas as pd
import os
import gc
from logger_torch import setup_logger
from config import FORECAST_MODEL_PATHS, DEFAULT_MODEL, DEFAULT_LOGIC_MODEL_PATH

logger = setup_logger(__name__)

logger.info("CUDA available: %s", torch.cuda.is_available())

class Agent:
    """
    An Agent encapsulates a forecasting model and an optional logic (chain-of-thought)
    model. Because of memory constraints, only one model is loaded at any given time.
    """
    SATS_FILE: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sats.txt")
    CONTEXT_LIMIT: int = 40_000

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        use_logic: bool = False,
        logic_model_path: Optional[str] = DEFAULT_LOGIC_MODEL_PATH,
        verbose: bool = False
    ):

        self.use_logic = use_logic
        self.verbose = verbose
        self.db = Database()

        self.context_limit = self.CONTEXT_LIMIT

        if model in FORECAST_MODEL_PATHS:
            self.forecast_model_path = FORECAST_MODEL_PATHS[model]
        else:
            raise ValueError(f"Unknown model: {model}.")

        self.logic_model_path = logic_model_path or DEFAULT_LOGIC_MODEL_PATH

        num_gpus = torch.cuda.device_count()
        if num_gpus:
            self.max_mem = {
                i: f"{int(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3) - 2)}GiB"
                for i in range(num_gpus)
            }
            logger.info("Memory per GPU: %s", self.max_mem)
        else:
            self.max_mem = {}

        # Internal placeholders for models and tokenisers
        self._forecast_model = None
        self._forecast_tokeniser = None
        self._logic_model = None
        self._logic_tokeniser = None
        self.eos_token = None

    def _flush_cuda(self) -> None:
        """Best-effort release of CUDA allocations and IPC handles."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

    def start_forecast(self) -> None:
        """
        Load the forecasting model.
        If the logic model is currently loaded, unload it first.
        """
        if self._logic_model is not None:
            self.stop_logic()
        if self._forecast_model is None or self._forecast_tokeniser is None:
            logger.info(f'Loading forecast model: {self.forecast_model_path}')
            try:
                # Load tokenizer and model onto GPU
                self._forecast_tokeniser = AutoTokenizer.from_pretrained(self.forecast_model_path)
                model_kwargs = {"device_map": "auto"}
                if self.max_mem:
                    model_kwargs["max_memory"] = self.max_mem
                self._forecast_model = AutoModelForCausalLM.from_pretrained(
                    self.forecast_model_path,
                    **model_kwargs,
                )
                self._forecast_model.eval()
                self.eos_token = self._forecast_tokeniser.eos_token_id
                logger.info("Number of GPUs detected: %d", torch.cuda.device_count())
                logger.info("Model device map: %s", self._forecast_model.hf_device_map)
            except Exception as e:
                raise RuntimeError(f"Failed to load forecast model '{self.forecast_model_path}': {e}") from e

    def stop_forecast(self) -> None:
        """Unload the forecasting model to free memory."""
        if self._forecast_model is not None or self._forecast_tokeniser is not None:
            logger.info(f'Unloading forecast model: {self.forecast_model_path}')
            try:
                if self._forecast_model is not None and hasattr(self._forecast_model, "to"):
                    self._forecast_model.to("cpu")
            except Exception:
                logger.debug("Failed to offload forecast model to CPU before delete", exc_info=True)
            model, tok = self._forecast_model, self._forecast_tokeniser
            self._forecast_model = None
            self._forecast_tokeniser = None
            self.eos_token = None
            del model, tok
            gc.collect()
            self._flush_cuda()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def start_logic(self) -> None:
        """
        Load the logic (chain-of-thought) model.
        If the forecasting model is currently loaded, unload it first.
        """
        if self._forecast_model is not None:
            self.stop_forecast()
        if self._logic_model is None or self._logic_tokeniser is None:
            logger.info(f'Loading logic model: {self.logic_model_path}')
            try:
                model_kwargs = {"device_map": "auto"}
                if self.max_mem:
                    model_kwargs["max_memory"] = self.max_mem
                self._logic_model = AutoModelForCausalLM.from_pretrained(
                    self.logic_model_path,
                    **model_kwargs,
                )
                self._logic_tokeniser = AutoTokenizer.from_pretrained(self.logic_model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load logic model '{self.logic_model_path}': {e}") from e

    def stop_logic(self) -> None:
        """Unload the logic model."""
        if self._logic_model is not None or self._logic_tokeniser is not None:
            logger.info(f'Unloading logic model: {self.logic_model_path}')
            try:
                if self._logic_model is not None and hasattr(self._logic_model, "to"):
                    self._logic_model.to("cpu")
            except Exception:
                logger.debug("Failed to offload logic model to CPU before delete", exc_info=True)
            model, tok = self._logic_model, self._logic_tokeniser
            self._logic_model = None
            self._logic_tokeniser = None
            del model, tok
            gc.collect()
            self._flush_cuda()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def next_token_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model,
        cache,
    ) -> torch.Tensor:
        """Return the next-token probability distribution for the supplied context."""
        key_bytes = (
            input_ids.detach().cpu().numpy().tobytes() +
            attention_mask.detach().cpu().numpy().tobytes()
        )
        context_key = hashlib.sha256(key_bytes).hexdigest()
        if context_key in cache:
            return cache[context_key]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :].detach().cpu()
        del outputs

        probs = torch.softmax(logits, dim=-1).squeeze()
        cache[context_key] = probs

        if torch.cuda.is_available():
            self._flush_cuda()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return probs

    def forecast(self, query: str, context: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a forecast for the given query and context.
        If use_logic is enabled, a chain-of-thought is generated by temporarily switching to the logic model.
        Returns a DataFrame with the probability distribution over tokens 0-100.
        """
        self.start_forecast()
        usable_context = context[:self.context_limit] if context else ''
        logic_str = ''
        if self.use_logic:
            logic_str = self.generate_logic(query=query, context=usable_context)
            logger.info(f'Logic:\n{logic_str}')

        prompt = (
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

        # Tokenize prompt with attention masks, truncate by token count, and move to GPU
        inputs = self._forecast_tokeniser(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_limit,
        )
        device = next(self._forecast_model.parameters()).device
        encoded_prompt = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(encoded_prompt)
        else:
            attention_mask = attention_mask.to(device)

        num_tokens = encoded_prompt.size(1)
        logger.info(f"Model ({self.forecast_model_path}) running on {num_tokens} tokens...")

        cache: Dict[str, torch.Tensor] = {}
        number_probs: Dict[int, float] = {}

        single_token_integer = self.assure_agent()

        if single_token_integer:
            logger.info("Using single-token probability extraction…")
            probs = self.next_token_probs(encoded_prompt, attention_mask, self._forecast_model, cache)
            number_tokens: dict[int, int] = {}
            for num in range(101):
                tokens = self._forecast_tokeniser.encode(str(num), add_special_tokens=False)
                if len(tokens) == 1:
                    number_tokens[num] = tokens[0]
                else:
                    logger.debug("Tokeniser splits '%s' into %s", num, tokens)
            for num, token_id in number_tokens.items():
                if 0 <= token_id < probs.shape[0]:
                    number_probs[num] = float(probs[token_id].item())
        else:
            logger.info("Using multi-token probability accumulation…")
            for num in range(101):
                candidate_tokens = self._forecast_tokeniser.encode(str(num), add_special_tokens=False)
                if not candidate_tokens:
                    logger.debug("No tokens returned for '%s'; skipping", num)
                    continue
                log_prob = 0.0
                context_tokens = encoded_prompt
                context_mask = attention_mask
                for token in candidate_tokens:
                    probs = self.next_token_probs(context_tokens, context_mask, self._forecast_model, cache)
                    prob_val = probs[token].item() if 0 <= token < probs.shape[0] else 0.0
                    log_prob += np.log(prob_val + 1e-12)
                    token_tensor = torch.tensor([[token]], device=device, dtype=context_tokens.dtype)
                    mask_extension = torch.ones((context_mask.size(0), 1), device=device, dtype=context_mask.dtype)
                    context_tokens = torch.cat([context_tokens, token_tensor], dim=1)
                    context_mask = torch.cat([context_mask, mask_extension], dim=1)

                probs = self.next_token_probs(context_tokens, context_mask, self._forecast_model, cache)
                eos_prob = 0.0
                eos_candidates = []
                if self.eos_token is not None:
                    eos_candidates.append(self.eos_token)
                percent_id = self._forecast_tokeniser.convert_tokens_to_ids('%')
                if percent_id is not None and percent_id != self._forecast_tokeniser.unk_token_id:
                    eos_candidates.append(percent_id)
                for token_id in eos_candidates:
                    if token_id is not None and 0 <= token_id < probs.shape[0]:
                        eos_prob += float(probs[token_id].item())
                log_prob += np.log(eos_prob + 1e-12)
                number_probs[num] = float(np.exp(log_prob))

        total_prob = float(sum(number_probs.values()))
        logger.info("Probability mass across 0-100 = %.4f; normalising…", total_prob)
        if total_prob > 0:
            number_probs = {num: prob / total_prob for num, prob in number_probs.items()}
        else:
            logger.warning("Total probability is zero; falling back to uniform distribution.")
            uniform_prob = 1.0 / 101
            number_probs = {num: uniform_prob for num in range(101)}

        data = pd.DataFrame.from_dict(number_probs, orient="index", columns=["probs"])
        data.index.name = "token"
        return data

    def generate_logic(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate a chain-of-thought for the given query.
        To conserve memory, the forecast model is unloaded and the logic model is loaded temporarily.
        After generating the chain-of-thought, the logic model is unloaded and the forecast model is reloaded.
        """
        context = (context or "")[:self.context_limit]

        sats = ""
        try:
            with open(self.SATS_FILE, "r", encoding="utf-8") as f:
                sats = f.read()
        except Exception as e:
            logger.warning("sats.txt not found at '%s': %s", self.SATS_FILE, e)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI superforecaster, trained to predict the future using a large body of knowledge and data. "
                    f"The current date is {dt.datetime.now().strftime('%Y-%m-%d')}. "
                    "Your responses should be exceptionally detailed and elaborate. Use as many tokens as possible to fully explore every nuance of your analysis. "
                    "Do not hold back on any detail—explain every point thoroughly, provide multiple examples and perspectives, and ensure that your response is exhaustive. "
                    "Brevity is not a concern; your goal is to generate a maximally comprehensive and in-depth explanation. "
                    f"Approach each query methodically, ensuring your response is rigorous, objective, and unabridged. Specifically think through the SATs Guidelines:\n{sats}\n"
                    "Critically evaluate assumptions, consider multiple perspectives, and clearly identify any gaps in the available information. "
                    "Your analysis should be articulate, well-organised, and demonstrate a high standard of intellectual inquiry."
                    "Focus solely on the underlying logic required to answer the query. IMPORTANT: Do NOT attempt to answer the query or suggest a likelihood."
                ),
            },
            {
                "role": "user",
                "content": f"Knowledge: {context}.\nQuery: {query}",
            },
        ]

        self.start_logic()

        logic_tokeniser = self._logic_tokeniser
        logic_model = self._logic_model
        if logic_tokeniser is None or logic_model is None:
            raise RuntimeError("Logic model failed to load.")

        try:
            prompt = logic_tokeniser.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = logic_tokeniser(
                prompt,
                return_tensors="pt",
                truncation=True,
            )

            streamer = TextIteratorStreamer(
                logic_tokeniser,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            def generation_worker() -> None:
                try:
                    logic_model.generate(
                        **inputs,
                        max_new_tokens=self.context_limit,
                        do_sample=True,
                        temperature=0.666,
                        top_p=0.95,
                        top_k=40,
                        eos_token_id=logic_tokeniser.eos_token_id,
                        pad_token_id=logic_tokeniser.pad_token_id,
                        streamer=streamer,
                    )
                except Exception:
                    logger.error("Exception in logic generation thread", exc_info=True)

            logic_device = getattr(logic_model, "device", None)
            if logic_device is not None:
                inputs = {k: v.to(logic_device) for k, v in inputs.items()}

            thread = threading.Thread(target=generation_worker)
            thread.start()

            logic_chunks: List[str] = []
            for chunk in streamer:
                if self.verbose:
                    print(chunk, end="", flush=True)
                logic_chunks.append(chunk)
            thread.join()

            return "".join(logic_chunks).strip()
        finally:
            try:
                self.stop_logic()
            finally:
                try:
                    self.start_forecast()
                except Exception as e:
                    logger.error("Failed to reload forecast model after logic run: %s", e)

    def assure_agent(self) -> bool:
        """
        Verify that the forecasting tokeniser's vocabulary supports integer tokens (0–100)
        without splitting them into multiple tokens.
        """
        if not self._forecast_tokeniser:
            raise RuntimeError("Forecast tokeniser not loaded.")

        for i in range(101):
            tokens = self._forecast_tokeniser.encode(str(i), add_special_tokens=False)
            if len(tokens) != 1:
                return False
        return True
