import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from semantic_kernel.agents import ChatCompletionAgent, HandoffOrchestration, OrchestrationHandoffs
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import AuthorRole, ChatMessageContent
from semantic_kernel.functions import kernel_function


class OrderStatusPlugin:
    @kernel_function
    def check_order_status(self, order_id: str) -> str:
        """Check the status of an order."""
        return f"Order {order_id} is shipped and will arrive in 2-3 days."


class OrderRefundPlugin:
    @kernel_function
    def process_refund(self, order_id: str, reason: str) -> str:
        """Process a refund for an order."""
        print(f"Processing refund for order {order_id} due to: {reason}")
        return f"Refund for order {order_id} has been processed successfully."


class OrderReturnPlugin:
    @kernel_function
    def process_return(self, order_id: str, reason: str) -> str:
        """Process a return for an order."""
        print(f"Processing return for order {order_id} due to: {reason}")
        return f"Return for order {order_id} has been processed successfully."


def normalize_openai_base_url(raw_url: str | None) -> str | None:
    if not raw_url:
        return raw_url
    normalized = raw_url.strip().strip('"').rstrip("/")
    for suffix in ("/responses", "/chat/completions", "/completions"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def get_chat_service() -> OpenAIChatCompletion:
    load_dotenv()

    endpoint = normalize_openai_base_url(os.getenv("AZURE_OPENAI_ENDPOINT"))
    deployment_name = os.getenv("AZURE_AGENT_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_AGENT_MODEL_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")

    if not endpoint or not deployment_name or not api_key:
        raise EnvironmentError(
            "Missing AZURE_OPENAI_ENDPOINT, AZURE_AGENT_MODEL (or AZURE_OPENAI_DEPLOYMENT_NAME), "
            "and API key."
        )

    client = AsyncOpenAI(base_url=endpoint, api_key=api_key)
    return OpenAIChatCompletion(
        ai_model_id=deployment_name,
        api_key=api_key,
        async_client=client,
    )


def agent_response_callback(message: ChatMessageContent) -> None:
    print(f"# {message.name}\n{message.content}\n")


def human_response_function() -> ChatMessageContent:
    user_input = input("User: ")
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)


async def main() -> None:
    service = get_chat_service()

    triage_agent = ChatCompletionAgent(
        name="TriageAgent",
        description="A customer support agent that triages issues.",
        instructions="Handle customer requests and route to the right specialist when needed.",
        service=service,
    )

    refund_agent = ChatCompletionAgent(
        name="RefundAgent",
        description="A customer support agent that handles refunds.",
        instructions="Handle refund requests.",
        service=service,
        plugins=[OrderRefundPlugin()],
    )

    order_status_agent = ChatCompletionAgent(
        name="OrderStatusAgent",
        description="A customer support agent that checks order status.",
        instructions="Handle order status requests.",
        service=service,
        plugins=[OrderStatusPlugin()],
    )

    order_return_agent = ChatCompletionAgent(
        name="OrderReturnAgent",
        description="A customer support agent that handles order returns.",
        instructions="Handle order return requests.",
        service=service,
        plugins=[OrderReturnPlugin()],
    )

    handoffs = (
        OrchestrationHandoffs()
        .add_many(
            source_agent=triage_agent,
            target_agents={
                refund_agent.name: "Transfer to this agent if the issue is refund related",
                order_status_agent.name: "Transfer to this agent if the issue is order status related",
                order_return_agent.name: "Transfer to this agent if the issue is order return related",
            },
        )
        .add(
            source_agent=refund_agent,
            target_agent=triage_agent,
            description="Transfer to this agent if the issue is not refund related",
        )
        .add(
            source_agent=order_status_agent,
            target_agent=triage_agent,
            description="Transfer to this agent if the issue is not order status related",
        )
        .add(
            source_agent=order_return_agent,
            target_agent=triage_agent,
            description="Transfer to this agent if the issue is not order return related",
        )
    )

    handoff_orchestration = HandoffOrchestration(
        members=[triage_agent, refund_agent, order_status_agent, order_return_agent],
        handoffs=handoffs,
        agent_response_callback=agent_response_callback,
        human_response_function=human_response_function,
    )

    runtime = InProcessRuntime()
    runtime.start()

    try:
        orchestration_result = await handoff_orchestration.invoke(
            task="A customer is on the line.",
            runtime=runtime,
        )

        value = await orchestration_result.get(timeout=300)
        print(f"***** Final Result *****\n{value}")
    finally:
        await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
