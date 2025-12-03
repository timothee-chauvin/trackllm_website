import asyncio
import json

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Config, logger
from trackllm_website.storage import Response, ResultsStorage
from trackllm_website.util import gather_with_concurrency_streaming, trim_to_length


def get_summary(responses: list[Response]) -> dict:
    """Generate a summary of responses grouped by endpoint.

    Args:
        responses: List of Response objects

    Returns:
        Dictionary mapping endpoint identifiers to summary statistics
    """
    summary = {}
    for response in responses:
        key = f"{response.endpoint.model}#{response.endpoint.provider}"
        if key not in summary:
            summary[key] = {
                "success": 0,
                "error": 0,
                "total_cost": 0.0,
            }

        if response.error:
            summary[key]["error"] += 1
        else:
            summary[key]["success"] += 1

        summary[key]["total_cost"] += response.cost

    return summary


async def main():
    """Query all endpoints with configured prompts and store results."""
    config = Config()
    storage = ResultsStorage(data_dir=config.data_dir)
    openrouter_client = OpenRouterClient()

    # Create all tasks upfront for concurrent execution
    tasks = [
        openrouter_client.query(endpoint=endpoint, prompt=prompt)
        for endpoint in config.endpoints
        for prompt in config.prompts
    ]

    logger.info(
        f"{len(tasks)} requests to send with max_workers={config.api.max_workers}"
    )

    all_responses = []
    i = 0
    async for response in gather_with_concurrency_streaming(
        config.api.max_workers, *tasks
    ):
        all_responses.append(response)
        storage.store_response(response)

        success_or_error = (
            "SUCCESS" if not response.error else f"ERROR: {response.error}"
        )
        logger.info(
            f"{i + 1}/{len(tasks)}: {response.endpoint.model} ({response.endpoint.provider}) "
            f"({repr(trim_to_length(response.prompt, 50))}) {success_or_error}"
        )
        i += 1

    # Print summary
    summary = get_summary(all_responses)
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
