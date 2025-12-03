import asyncio
import json

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Config, logger
from trackllm_website.storage import ResultsStorage


async def main():
    """Query all endpoints with configured prompts and store results."""
    config = Config()
    storage = ResultsStorage(data_dir=config.data_dir)
    openrouter_client = OpenRouterClient()

    all_responses = []

    # Query all endpoints with all prompts
    for endpoint in config.endpoints:
        for prompt in config.prompts:
            logger.info(
                f"Querying {endpoint.model} ({endpoint.provider}) with prompt: {repr(prompt[:50])}"
            )
            response = await openrouter_client.query(endpoint=endpoint, prompt=prompt)

            if response.error:
                logger.error(f"Error: {response.error}")
            else:
                logger.info(f"Success! Cost: ${response.cost:.6f}")

            all_responses.append(response)
            storage.store_response(response)

    # Print summary
    summary = storage.get_summary(all_responses)
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(json.dumps(summary, indent=2))

    total_cost = sum(s["total_cost"] for s in summary.values())
    total_success = sum(s["success"] for s in summary.values())
    total_errors = sum(s["error"] for s in summary.values())

    logger.info(f"\nTotal queries: {total_success + total_errors}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Total cost: ${total_cost:.6f}")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
