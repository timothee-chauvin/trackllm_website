import asyncio

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Config


async def main():
    config = Config()
    print(config)
    print(config.api)
    openrouter_client = OpenRouterClient()
    for endpoint in config.endpoints:
        response = await openrouter_client.query(endpoint=endpoint, prompt="Hi")
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
