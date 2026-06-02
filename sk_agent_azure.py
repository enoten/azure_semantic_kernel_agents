import asyncio
import os

from dotenv import load_dotenv
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion


def normalize_azure_endpoint(raw_url: str | None) -> str | None:
    """Convert .env OpenAI-compatible URL to Azure resource endpoint."""
    if not raw_url:
        return raw_url
    normalized = raw_url.strip().strip('"').rstrip("/")
    for suffix in (
        "/openai/v1",
        "/openai",
        "/responses",
        "/chat/completions",
        "/completions",
    ):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized.rstrip("/")


async def main() -> None:
    load_dotenv()

    endpoint = normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT"))
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint or not deployment_name or not api_key:
        raise EnvironmentError(
            "Missing required env vars: AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY"
        )

    agent = ChatCompletionAgent(
        service=AzureChatCompletion(
            deployment_name=deployment_name,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
        ),
        name="SK-Assistant",
        instructions="You are a helpful assistant.",
    )

    response = await agent.get_response(messages="Write a haiku about Semantic Kernel.")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
