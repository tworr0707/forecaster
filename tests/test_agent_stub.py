import types
import pandas as pd
from agents_vllm import Agent


class FakeTokenizer:
    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        # Map string numbers to ids 0-100 (wrap if vocab smaller)
        self._vocab = {str(i): i % vocab_size for i in range(101)}

    def get_vocab(self):
        return self._vocab

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        # Simple digit-wise encoding
        return [self._vocab.get(ch, 1) for ch in text.split()]

    def decode(self, ids):
        # Return first id as string for simplicity
        return str(ids[0]) if ids else ""

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, 1)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Minimal stub to concatenate contents
        return "\n".join([m["content"] for m in messages])


class FakeLogProb:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeOutput:
    def __init__(self, vocab_size):
        # uniform logprob over vocab
        self.logprobs = [{i: FakeLogProb(-1.0) for i in range(vocab_size)}]
        self.text = ""


class FakeRequestOutput:
    def __init__(self, vocab_size):
        self.outputs = [FakeOutput(vocab_size)]


class FakeLLM:
    def __init__(self, vocab_size=128):
        self._tokenizer = FakeTokenizer(vocab_size)

    def get_tokenizer(self):
        return self._tokenizer

    def generate(self, prompts, sampling_params, request_id=None, stream=False):
        if stream:
            # generator yielding one chunk
            yield FakeRequestOutput(self._tokenizer.vocab_size)
            return
        return [FakeRequestOutput(self._tokenizer.vocab_size)]


def test_assure_agent_single_token(monkeypatch):
    agent = Agent(model="llama-3B")
    agent._forecast_llm_engine = FakeLLM(vocab_size=256)
    agent._forecast_vocab_size = agent._forecast_llm_engine.get_tokenizer().vocab_size
    agent.eos_token_id = agent._forecast_llm_engine.get_tokenizer().eos_token_id
    assert agent.assure_agent() is True


def test_next_token_cache_eviction(monkeypatch):
    agent = Agent(model="llama-3B")
    fake_llm = FakeLLM(vocab_size=16)
    cache = {}
    # Insert three distinct prompts with max_cache_size=2
    for i in range(3):
        agent.next_token_probs(f"prompt-{i}", fake_llm, cache, max_cache_size=2)
    assert len(cache) == 2


def test_ensemble_stub(monkeypatch):
    # Stub Agent.forecast to avoid real model load
    def fake_forecast(self, query, context=None):
        s = pd.Series([0.01] * 101)
        s.iloc[50] = 0.5
        return pd.DataFrame({"probs": s / s.sum()})

    monkeypatch.setattr(Agent, "forecast", fake_forecast)
    monkeypatch.setattr(Agent, "start_forecast", lambda self: None)
    monkeypatch.setattr(Agent, "stop_forecast", lambda self: None)

    from ensemble import Ensemble

    ens = Ensemble()
    ens.forecast("test query")
    assert ens.ensemble_df is not None
    assert "ensemble_forecast" in ens.ensemble_df.columns
