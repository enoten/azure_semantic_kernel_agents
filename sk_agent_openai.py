import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion


def normalize_openai_base_url(raw_url: str | None) -> str | None:
    """Normalize Azure AI Foundry OpenAI-compatible base URL (…/openai/v1)."""
    if not raw_url:
        return raw_url
    normalized = raw_url.strip().strip('"').rstrip("/")
    for suffix in ("/responses", "/chat/completions", "/completions"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


async def main() -> None:
    load_dotenv()

    endpoint = normalize_openai_base_url(os.getenv("AZURE_OPENAI_ENDPOINT"))
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not endpoint or not deployment_name or not api_key:
        raise EnvironmentError(
            "Missing required env vars: AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY"
        )

    # Azure AI Foundry uses an OpenAI-compatible /openai/v1 endpoint (not classic Azure OpenAI URLs).
    client = AsyncOpenAI(base_url=endpoint, api_key=api_key)

    agent = ChatCompletionAgent(
        service=OpenAIChatCompletion(
            ai_model_id=deployment_name,
            api_key=api_key,
            async_client=client,
        ),
        name="SK-Assistant",
        instructions="You are a helpful assistant.",
    )

    response = await agent.get_response(messages="Write a haiku about Semantic Kernel.")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())