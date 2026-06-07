import asyncio
import os

from dotenv import load_dotenv

from semantic_kernel.agents import Agent, ChatCompletionAgent, SequentialOrchestration, ConcurrentOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.agents import GroupChatOrchestration, RoundRobinGroupChatManager


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

    writer = ChatCompletionAgent(
        name="Writer",
        description="A content writer.",
        instructions=(
            "You are an excellent content writer. You create new content and edit contents based on the feedback."
        ),
        service=service,
    )
    reviewer = ChatCompletionAgent(
        name="Reviewer",
        description="A content reviewer.",
        instructions=(
            "You are an excellent content reviewer. You review the content and provide feedback to the writer."
        ),
        service=service,
    )
    return [writer, reviewer]


def agent_response_callback(message: ChatMessageContent) -> None:
    print(f"# {message.name}\n{message.content}\n")


async def main() -> None:
    agents = get_agents()
    
    group_chat_orchestration = GroupChatOrchestration(
        members=agents,
        manager=RoundRobinGroupChatManager(max_rounds=5),  # Odd number so writer gets the last word
        agent_response_callback=agent_response_callback,
    )

    runtime = InProcessRuntime()
    runtime.start()

    try:
        orchestration_result = await group_chat_orchestration.invoke(
            task="Create a slogan for a new electric SUV that is affordable and fun to drive.",
            runtime=runtime,
        )

        value = await orchestration_result.get(timeout=120)
        print(f"***** Final Result *****\n{value}")
    finally:
        await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())