import datetime as dt
import gc
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from logger_torch import setup_logger


logger = setup_logger(__name__)


class VisionAgent:
    def __init__(self, model_name: str = "gemma3", image_dir: str = "images") -> None:
        if model_name == "gemma3":
            self.model_path = "google/gemma-3n-E4B-it"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.forecast_model_path = self.model_path
        self.image_dir = image_dir

        self.model: Optional[AutoModelForCausalLM] = None
        self.processor = None
        self.tokenizer = None
        self.eos_token: Optional[int] = None
        self.pad_token_id: Optional[int] = None

        self.max_new_tokens = 256
        self.context_limit = 1_000_000

        self._has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self._has_mps:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_path = os.path.join(project_root, self.image_dir)

    def start_forecast(self) -> None:
        if self.model is None or self.processor is None:
            logger.info("Loading %s...", self.forecast_model_path)
            try:
                self.processor = AutoProcessor.from_pretrained(self.forecast_model_path)
                model_kwargs: Dict[str, object] = {}
                if torch.cuda.is_available():
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["torch_dtype"] = (
                        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    )
                elif self._has_mps:
                    model_kwargs["torch_dtype"] = torch.float16
                else:
                    model_kwargs["torch_dtype"] = torch.float32

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.forecast_model_path,
                    **model_kwargs,
                )

                if not (torch.cuda.is_available() or self._has_mps):
                    self.model.to(self.device)

                self.model.eval()

                tokenizer = getattr(self.processor, "tokenizer", None)
                if tokenizer is None:
                    raise RuntimeError("Processor does not expose a tokenizer.")
                if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                self.tokenizer = tokenizer
                self.eos_token = tokenizer.eos_token_id
                self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            except Exception as e:
                self.model = None
                self.processor = None
                self.tokenizer = None
                self.eos_token = None
                self.pad_token_id = None
                raise RuntimeError(f"Failed to load {self.forecast_model_path}: {e}") from e

    def stop_forecast(self) -> None:
        if self.model is not None or self.processor is not None:
            logger.info("Unloading %s...", self.forecast_model_path)
        try:
            if self.model is not None and hasattr(self.model, "to"):
                self.model.to("cpu")
        except Exception:
            logger.debug("Failed to offload vision model to CPU before delete", exc_info=True)
        model, processor, tokenizer = self.model, self.processor, self.tokenizer
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.eos_token = None
        self.pad_token_id = None
        del model, processor, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        if self._has_mps and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    def _resolve_image_paths(self, image_files: List[str]) -> List[str]:
        resolved: List[str] = []
        for filename in image_files:
            path = filename if os.path.isabs(filename) else os.path.join(self.image_path, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            resolved.append(path)
        return resolved

    def _apply_chat_template(self, prompt: str, num_images: int) -> str:
        if self.processor is None:
            return prompt

        template_fn = None
        if hasattr(self.processor, "apply_chat_template"):
            template_fn = self.processor.apply_chat_template
        elif self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            template_fn = self.tokenizer.apply_chat_template

        if template_fn is None:
            return prompt

        content = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        try:
            return template_fn(conversation, add_generation_prompt=True)
        except TypeError:
            return template_fn(conversation)

    def _prepare_inputs(self, prompt: str, image_files: List[str]) -> Dict[str, torch.Tensor]:
        kwargs = {"text": prompt, "return_tensors": "pt", "padding": True}
        images: List[Image.Image] = []
        if image_files:
            image_paths = self._resolve_image_paths(image_files)
            images = [Image.open(path).convert("RGB") for path in image_paths]
            kwargs["images"] = images

        try:
            inputs = self.processor(**kwargs)
        finally:
            for image in images:
                image.close()

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        return {k: v for k, v in inputs.items()}

    def query(self, query: str, image_files: List[str], verbose: bool = False) -> str:
        image_files = image_files or []
        try:
            self.start_forecast()
            formatted_prompt = self._apply_chat_template(query, len(image_files))
            inputs = self._prepare_inputs(formatted_prompt, image_files)
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "eos_token_id": self.eos_token,
                "pad_token_id": self.pad_token_id,
            }
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **generate_kwargs)
            if hasattr(self.processor, "batch_decode"):
                outputs = self.processor.batch_decode(output_ids, skip_special_tokens=True)
            else:
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            result = outputs[0].strip() if outputs else ""
            if verbose:
                logger.info("Model output: %s", result)
            return result
        except Exception as e:
            logger.error("Failed to generate image: %s", e)
            return ""
        finally:
            self.stop_forecast()

    def next_token_probs(self, prompts: List[str], image_files: List[str], cache: Dict) -> torch.Tensor:
        self.start_forecast()
        key = (tuple(prompts), tuple(image_files))
        if key in cache:
            return cache[key]

        if not prompts:
            raise ValueError("Prompts must not be empty.")

        prompt = prompts[0]
        inputs = self._prepare_inputs(prompt, image_files)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].detach()
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
        cache[key] = probs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._has_mps and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        return probs

    def forecast(
        self,
        query: str,
        image_files: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> pd.DataFrame:
        image_files = image_files or []
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
            f"Query: {query}\n\n"
            "Now, provide your prediction:\n"
            "Answer: "
        )

        try:
            self.start_forecast()
            formatted_prompt = self._apply_chat_template(prompt, len(image_files))
            cache: Dict = {}
            number_probs: Dict[int, float] = {}
            single_token_integer = self.assure_agent()

            if self.tokenizer is None:
                raise RuntimeError("Tokenizer is not available after loading the model.")

            if single_token_integer:
                number_tokens = {
                    num: self.tokenizer.encode(str(num), add_special_tokens=False)[0]
                    for num in range(101)
                }
                probs = self.next_token_probs([formatted_prompt], image_files, cache)
                if probs.ndim > 1:
                    probs = probs.squeeze()
                for num, token_id in number_tokens.items():
                    number_probs[num] = float(probs[token_id].item())
            else:
                for num in tqdm(range(101), desc="Computing probabilities"):
                    candidate = str(num)
                    candidate_tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
                    log_prob = 0.0
                    context_text = formatted_prompt
                    for token_id in candidate_tokens:
                        probs = self.next_token_probs([context_text], image_files, cache)
                        token_prob = float(probs[token_id].item())
                        log_prob += float(np.log(token_prob + 1e-12))
                        context_text += self.tokenizer.decode([token_id])
                    probs = self.next_token_probs([context_text], image_files, cache)
                    eos_candidates = [self.eos_token]
                    percent_id = None
                    if hasattr(self.tokenizer, "convert_tokens_to_ids"):
                        try:
                            percent_id = self.tokenizer.convert_tokens_to_ids("%")
                        except KeyError:
                            percent_id = None
                    if percent_id is None:
                        percent_tokens = self.tokenizer.encode("%", add_special_tokens=False)
                        if percent_tokens:
                            percent_id = percent_tokens[0]
                    if percent_id is not None:
                        eos_candidates.append(percent_id)
                    eos_prob = sum(float(probs[e].item()) for e in eos_candidates if e is not None)
                    log_prob += float(np.log(eos_prob + 1e-12))
                    number_probs[num] = float(np.exp(log_prob))

            total = sum(number_probs.values())
            if total > 0:
                for k in number_probs:
                    number_probs[k] /= total

            data = pd.DataFrame.from_dict(number_probs, orient="index", columns=["probs"])
            data.index.name = "token"
            return data
        except Exception as e:
            logger.error("Failed to generate forecast: %s", e)
            return pd.DataFrame()
        finally:
            self.stop_forecast()

    def assure_agent(self) -> bool:
        self.start_forecast()
        if self.tokenizer is None:
            return False
        try:
            for i in range(101):
                tokens = self.tokenizer.encode(str(i), add_special_tokens=False)
                if len(tokens) != 1:
                    return False
            return True
        except Exception:
            return False
