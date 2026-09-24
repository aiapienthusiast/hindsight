"""Tests for the Cheaper Inference OpenAI-compatible LLM provider."""

import pytest


def test_cheaperinference_llm_provider_from_env_has_expected_default_model(monkeypatch):
    """LLMProvider.from_env should use the cheaperinference default model and base URL."""
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.llm_wrapper import LLMProvider

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "cheaperinference")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "test-key")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_BASE_URL", raising=False)
    clear_config_cache()

    try:
        llm = LLMProvider.from_env()
        assert llm.provider == "cheaperinference"
        assert llm.model == "gpt-5.4-mini"
        assert llm.base_url == "https://api.cheaperinference.com/v1"
    finally:
        clear_config_cache()


def test_cheaperinference_uses_openai_compatible_provider_with_default_base_url():
    """The provider factory should route cheaperinference to OpenAICompatibleLLM."""
    from hindsight_api.engine.llm_wrapper import LLMProvider
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = LLMProvider(
        provider="cheaperinference",
        api_key="test-key",
        base_url="",
        model="gpt-5.4-mini",
    )

    assert isinstance(llm._provider_impl, OpenAICompatibleLLM)
    assert llm._provider_impl.base_url == "https://api.cheaperinference.com/v1"


def test_cheaperinference_rejects_missing_api_key():
    """cheaperinference should fail fast without an API key, matching the other cloud providers."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with pytest.raises(ValueError, match="API key is required for cheaperinference"):
        LLMProvider(
            provider="cheaperinference",
            api_key="",
            base_url="",
            model="gpt-5.4-mini",
        )
