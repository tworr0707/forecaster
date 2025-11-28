import datetime as dt
import json
import logging
import re
from typing import Optional
import torch
import threading
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
#from semanticretriever import SemanticRetriever


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_forecast(user_question: str) -> str:
    """Estimate the probability and normalised entropy of a future event.

    Args:
        user_question (str): A clearly worded question about a potential future event.

    Returns:
        str: JSON string with expected_value (int) and normalised_entropy (float).
    """
    try:
        # TODO: Replace mock_result with actual forecasting logic
        mock_result = json.dumps(
            {
                "expected_value": 27,
                "normalised_entropy": 0.256,
            }
        )
        return mock_result
    except Exception as e:
        logging.error("Error in get_forecast: %s", e, exc_info=True)
        return f"Error: {e}"
    
def calculate(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression.
    """
    try:
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "Error: invalid characters."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


FUNCTION_DEFS = [
    {
        "name": "get_forecast",
        "description": (
            "Estimates the probability (0–100) that a future event will occur and "
            "returns the associated normalised entropy (0–1), based on the user question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_question": {
                    "type": "string",
                    "description": (
                        "A clearly worded question about a potential future event which has a binary (YES/NO) outcome "
                        "(e.g. 'Will it rain in London on Friday?')"
                    ),
                }
            },
            "required": ["user_question"],
        },
    },
    {
        "name": "calculate",
        "description": "Compute a simple arithmetic expression, e.g. '2+2*3'.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    }
]

FUNCTIONS_MAP = {
    "get_forecast": get_forecast,
    "calculate": calculate
}

MODEL_PATH: str = "Qwen/Qwen3-14B-Instruct"

SYS_PROMPT = f"""
You are an expert AI superforecaster trained to predict future outcomes. The current date is {dt.datetime.now().strftime('%Y-%m-%d')}.

You have access to the following tool functions:

{json.dumps(FUNCTION_DEFS, indent=2)}

To assess a user's query, you MUST ALWAYS call this tool first. The tool will return:
- an expected probability value for the event
- a normalised entropy score (a proxy for confidence; lower = higher confidence)

Your workflow:
1. Parse and understand the user's question.
2. Emit a function call in the EXACT format shown below to retrieve the forecast:
   <function_call>
   {{
    "name": "get_forecast", 
    "arguments": {{
        "user_question": "..."
        }} 
    }}
   </function_call>
3. Wait for the tool's result before continuing your answer.

The first sentence must read EXACTLY like: "It is **<Likelihood label> (<probability>%)** with **<Confidence label>** that <event> will occur."  
Always include the numeric probability (rounded to the nearest whole percent), and phrase the event in the indicative (“will occur”), not “may”.

ONLY after the function call, use the result to formulate a final response as a short prose paragraph:

The first sentence should be the **Key Judgement**, e.g. "It is highly likely with MEDIUM confidence that event X will occur." 
Follow immediately with the **Evidence** using the provided knowledge, e.g. "This assessment is based on Y."

Your answer should be a single, clear paragraph in natural language.

Use only these standardised likelihood labels:
- **Remote chance** (0–5%)
- **Highly unlikely** (10–20%)
- **Unlikely** (25–35%)
- **Realistic possibility** (40–50%)
- **Likely** (55–75%)
- **Highly likely** (80–90%)
- **Almost certain** (95–100%)

Use only these standardised confidence labels:
- **HIGH confidence** (entropy 0.0–0.2)
- **MEDIUM-HIGH confidence** (0.2–0.4)
- **MEDIUM confidence** (0.4–0.6)
- **LOW-MEDIUM confidence** (0.6–0.8)
- **LOW confidence** (0.8–1.0)

Do not include your own speculation on the likelihood or confidence. Always base your final answer on the tool output and any relevant evidence you know.
"""


class ToolAgent:
    """Agent that interacts with a language model to forecast event probabilities.

    Attributes:
        model: Loaded language model instance.
        tokeniser: Tokeniser associated with the model.
        messages (list): List of conversation messages.
        response (str): Generated response text.
    """

    def __init__(self) -> None:
        """Initialize the ToolAgent by loading the language model and tokeniser."""
        try:
            # Select dtype for NVIDIA GPUs
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
            self.tokeniser = AutoTokenizer.from_pretrained(
                MODEL_PATH, use_fast=True, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            if self.tokeniser.pad_token_id is None:
                self.tokeniser.pad_token = self.tokeniser.eos_token
            self.messages: list[dict] = []
            self.response: str = ""
            logging.info("Model and tokeniser loaded successfully!")
        except Exception as e:
            logging.error("Failed to load model or tokeniser: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to load model or tokeniser: {e}") from e

    def close(self) -> None:
        """Explicitly free CUDA memory held by the model/tokeniser."""
        try:
            if hasattr(self, "model") and self.model is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    logging.debug("Failed to move ToolAgent model to CPU before delete", exc_info=True)
            model = getattr(self, "model", None)
            tok = getattr(self, "tokeniser", None)
            self.model = None
            self.tokeniser = None
            del model, tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
        except Exception:
            logging.warning("ToolAgent close() encountered an error", exc_info=True)

    def __del__(self) -> None:
        self.close()

    def build_prompt(self, enable_thinking: bool = True, zero_temp: bool = False):
        """Build the prompt string from the current messages and select the sampler.

        Args:
            enable_thinking (bool): Whether to enable 'thinking' mode in prompt/template.
            zero_temp (bool): If True, set temperature to zero for deterministic sampling.

        Returns:
            tuple: (prompt, sampler)
        """
        prompt = self.tokeniser.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        # Sampler configuration map for clarity
        sampler_configs = {
            (True, True):  {"temp": 0.0, "top_p": 0.95, "top_k": 20, "min_p": 0},
            (True, False): {"temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
            (False, True): {"temp": 0.0, "top_p": 0.8,  "top_k": 20, "min_p": 0},
            (False, False):{"temp": 0.7, "top_p": 0.8,  "top_k": 20, "min_p": 0},
        }
        config = sampler_configs[(enable_thinking, zero_temp)]

        return prompt, config

    def process_messages(self) -> Optional[str]:

        self.response = ""
        prompt, config = self.build_prompt(zero_temp=True)

        # Tokenize and move to the model's device
        inputs = self.tokeniser(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokeniser, skip_prompt=True, skip_special_tokens=True
        )

        temperature = config.get("temperature", 0.0)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=None,
            do_sample=temperature is not None and temperature > 0.0,
            temperature=temperature if temperature and temperature > 0.0 else None,
            top_p=config.get("top_p", None),
            top_k=config.get("top_k", None),
            eos_token_id=self.tokeniser.eos_token_id,
            pad_token_id=self.tokeniser.pad_token_id,
            use_cache=True,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for piece in streamer:
            self.response += piece
            print(piece, end="", flush=True)

        thread.join()

        if "<function_call>" in self.response and "</function_call>" in self.response:
            match = re.search(r"<function_call>(.*?)</function_call>", self.response, re.S)
            if match:
                func_json = match.group(1).strip()
                return func_json
        else:
            return
        
    def process_tool(self, func_json):            
        try:
            func_call = json.loads(func_json)
        except json.JSONDecodeError as e:
            logging.error("Failed to parse function call JSON: %s", e, exc_info=True)
            return f"Error: failed to parse function call JSON - {e}"

        func = FUNCTIONS_MAP.get(func_call.get("name"))
        if func is None:
            error_msg = f"Error: unknown function {func_call.get('name')}"
            logging.error(error_msg)
            result = error_msg
        else:
            try:
                result = func(**func_call.get("arguments", {}))
            except Exception as e:
                logging.error("Error during function execution: %s", e, exc_info=True)
                result = f"Error during function execution: {e}"

        logging.info("Function response: %s", result)

        self.messages.append(
            {
                "role": "assistant",
                "content": self.response,
                "function_call": func_json,
            }
        )
        self.messages.append(
            {
                "role": "tool",
                "name": func_call.get("name"),
                "content": result,
            }
        )

    def query(self, query: str) -> str:
        """Query the language model to generate a forecast answer.

        Args:
            query (str): The user's question about a future event.

        Returns:
            str: The final answer generated by the model.
        """
        # TODO: Implement retrieval of relevant knowledge/context
        try:
            #ret = SemanticRetriever()
            knowledge = "" #ret.get_context(query, top_n=1)
            logging.info("Retrieved knowledge context for query.")

            self.messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "assistant", "content": f"Relevant evidence: {knowledge}"},
                {"role": "user", "content": f"Query: {query}"},
            ]

            while True:

                # Test whether model response contains tool call
                func_json = self.process_messages()

                # If it contains tool call, process it
                if func_json is not None:
                    self.process_tool(func_json)
                # Else get final answer
                else:
                    break

            final_answer = self.response.split("</think>")[-1].strip()

            return final_answer

        except Exception as e:
            logging.error("Error in query method: %s", e, exc_info=True)
            return f"Error processing query: {e}"
