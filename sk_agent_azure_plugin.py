import asyncio
import os

from dotenv import load_dotenv

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import kernel_function, KernelArguments

from typing import Annotated
from pydantic import BaseModel

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

class MenuPlugin:
    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder Soup
        Special Salad: Cobb Salad
        Special Drink: Chai Tea Drink
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, 
        menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        if "soup" in menu_item.lower():
            return 9.99
        elif "salad" in menu_item.lower():
            return 7.99
        elif "drink" in menu_item.lower():
            return 2.99
        else:
            return 0.00

class MenuItem(BaseModel):
    price: float
    name: str

async def main() -> None:
    load_dotenv()

    endpoint = normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT"))
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    # Configure structured output format
    settings = OpenAIChatPromptExecutionSettings()
    settings.response_format = MenuItem

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
        plugins=[MenuPlugin()],
        arguments=KernelArguments(settings)
    )
 
    while True:
        #response = await agent.get_response(messages="What is the price of the soup special?")
        #print(response.content)
        user_input = input("User > ")
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        response = await agent.get_response(messages=user_input)
        print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
