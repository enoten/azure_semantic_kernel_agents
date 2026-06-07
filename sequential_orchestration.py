import asyncio
import os

from dotenv import load_dotenv

from semantic_kernel.agents import Agent, ChatCompletionAgent, SequentialOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent


def normalize_azure_endpoint(raw_url: str | None) -> str | None:
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


def get_chat_service() -> AzureChatCompletion:
    load_dotenv()

    endpoint = normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT"))
    deployment_name = os.getenv("AZURE_AGENT_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_AGENT_MODEL_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint or not deployment_name or not api_key:
        raise EnvironmentError(
            "Missing AZURE_OPENAI_ENDPOINT, AZURE_AGENT_MODEL (or AZURE_OPENAI_DEPLOYMENT_NAME), "
            "and AZURE_OPENAI_API_KEY."
        )

    return AzureChatCompletion(
        deployment_name=deployment_name,
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version,
    )


def get_agents() -> list[Agent]:
    service = get_chat_service()

    concept_extractor_agent = ChatCompletionAgent(
        name="ConceptExtractorAgent",
        instructions=(
            "You are a marketing analyst. Given a product description, identify:\n"
            "- Key features\n"
            "- Target audience\n"
            "- Unique selling points\n\n"
        ),
        service=service,
    )
    writer_agent = ChatCompletionAgent(
        name="WriterAgent",
        instructions=(
            "You are a marketing copywriter. Given a block of text describing features, audience, and USPs, "
            "compose a compelling marketing copy (like a newsletter section) that highlights these points. "
            "Output should be short (around 150 words), output just the copy as a single text block."
        ),
        service=service,
    )
    format_proof_agent = ChatCompletionAgent(
        name="FormatProofAgent",
        instructions=(
            "You are an editor. Given the draft copy, correct grammar, improve clarity, ensure consistent tone, "
            "give format and make it polished. Output the final improved copy as a single text block."
        ),
        service=service,
    )
    return [concept_extractor_agent, writer_agent, format_proof_agent]


def agent_response_callback(message: ChatMessageContent) -> None:
    print(f"# {message.name}\n{message.content}\n")


async def main() -> None:
    agents = get_agents()
    sequential_orchestration = SequentialOrchestration(
        members=agents,
        agent_response_callback=agent_response_callback,
    )

    runtime = InProcessRuntime()
    runtime.start()

    try:
        orchestration_result = await sequential_orchestration.invoke(
            task="An eco-friendly stainless steel water bottle that keeps drinks cold for 24 hours",
            runtime=runtime,
        )

        value = await orchestration_result.get(timeout=120)
        print(f"***** Final Result *****\n{value}")
    finally:
        await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
