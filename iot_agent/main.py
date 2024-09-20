import os
import time
import signal
from typing import Optional, Tuple
from openai import OpenAI, OpenAIError
from loguru import logger
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
RATE_LIMIT_WAIT_TIME = 10  # Time to wait on rate limit error
GLOBAL_TIMEOUT = 300  # 5 minutes timeout for the entire process

if not API_KEY:
    raise ValueError("OpenAI API key must be set as an environment variable.")

client = OpenAI(api_key=API_KEY)

console = Console()

class IterationOfThought:
    def __init__(self, model: str = MODEL, max_iterations: int = 5, timeout: int = 30):
        self.model = model
        self.max_iterations = max_iterations
        self.timeout = timeout

    def _call_openai(self, prompt: str, temperature: float = 0.5, max_retries: int = 3) -> str:
        logger.debug(f"Calling OpenAI API with prompt: {prompt}")

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            except OpenAIError as e:
                logger.error(f"An error occurred while calling OpenAI API: {e}")
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Rate limit exceeded. Waiting for {RATE_LIMIT_WAIT_TIME} seconds."
                    )
                    time.sleep(RATE_LIMIT_WAIT_TIME)
                else:
                    logger.error(f"Max retries reached. Error: {e}")
                    return ""
        logger.error("Failed to get a response from OpenAI API after max retries")
        return ""

    def inner_dialogue_agent(self, query: str, previous_response: str) -> str:
        prompt = (
            f"Given the original query: '{query}' and the previous response: '{previous_response}', "
            "generate an instructive and context-specific prompt to refine and improve the answer. "
            "Ensure that the new prompt encourages deeper reasoning or addresses any gaps in the previous response."
        )
        logger.debug(f"Generated prompt for IDA: {prompt}")
        return self._call_openai(prompt)

    def llm_agent(self, query: str, prompt: str) -> str:
        full_prompt = f"Query: {query}\nPrompt: {prompt}\nResponse:"
        logger.debug(f"Full prompt for LLM: {full_prompt}")
        return self._call_openai(full_prompt)

    def stopping_criterion(self, response: str) -> bool:
        lower_response = response.lower()
        return "answer:" in lower_response or "final answer:" in lower_response

    def aiot(self, query: str) -> str:
        logger.info("Starting AIoT...")
        current_response = self.llm_agent(query, "Initial Prompt")

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\nIteration {iteration}:")
            logger.info(f"LLMA Response:\n{current_response}\n")

            if self.stopping_criterion(current_response):
                logger.info("Stopping criterion met.")
                break

            new_prompt = self.inner_dialogue_agent(query, current_response)
            logger.info(f"IDA Generated Prompt:\n{new_prompt}\n")
            current_response = self.llm_agent(query, new_prompt)
            time.sleep(self.timeout)  # To avoid hitting rate limits

        logger.info("AIoT completed.\n")
        return current_response

    def giot(self, query: str, fixed_iterations: int) -> str:
        logger.info("Starting GIoT...")
        current_response = self.llm_agent(query, "Initial Prompt")

        for iteration in range(1, fixed_iterations + 1):
            logger.info(f"\nIteration {iteration}:")
            logger.info(f"LLMA Response:\n{current_response}\n")

            new_prompt = self.inner_dialogue_agent(query, current_response)
            logger.info(f"IDA Generated Prompt:\n{new_prompt}\n")
            current_response = self.llm_agent(query, new_prompt)
            time.sleep(self.timeout)  # To avoid hitting rate limits

        logger.info("GIoT completed.\n")
        return current_response

def get_user_query() -> str:
    sample_query = (
        "A textile dye containing an extensively conjugated pi-electrons emits light with energy of 2.3393 eV. "
        "What color of light is absorbed by the organic compound? Pick an answer from the following options:\n"
        "A. Red\nB. Yellow\nC. Blue\nD. Violet"
    )

    console.print(Panel.fit("Enter your query (or press Enter to use the sample query):"))
    user_input = input().strip()
    return user_input if user_input else sample_query

def timeout_handler(signum, frame):
    raise TimeoutError("Process took too long")

def interrupt_handler(signum, frame):
    raise KeyboardInterrupt("User interrupted the process")

def run_iot(iot: IterationOfThought, query: str, method: str) -> str:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Running {method}...", total=None)
        result = iot.aiot(query) if method == "AIoT" else iot.giot(query, fixed_iterations=3)
        progress.update(task, completed=True)
    return result

@click.command()
@click.option('--method', type=click.Choice(['AIoT', 'GIoT', 'both']), default='both', help='Choose the method to run')
def main(method: str) -> None:
    logger.add("debug.log", level="DEBUG")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.signal(signal.SIGINT, interrupt_handler)

    iot = IterationOfThought(model=MODEL, max_iterations=5, timeout=2)
    query = get_user_query()
    console.print(Panel(f"Using query: {query}", title="Query", expand=False))

    try:
        signal.alarm(GLOBAL_TIMEOUT)

        if method in ['AIoT', 'both']:
            final_response_aiot = run_iot(iot, query, "AIoT")
            console.print(Panel(final_response_aiot, title="Final AIoT Response", expand=False))

        if method in ['GIoT', 'both']:
            final_response_giot = run_iot(iot, query, "GIoT")
            console.print(Panel(final_response_giot, title="Final GIoT Response", expand=False))

        signal.alarm(0)
    except TimeoutError:
        console.print("⚠️ Process timed out. Please try again.", style="bold red")
    except KeyboardInterrupt:
        console.print("✋ Process interrupted by user.", style="bold yellow")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        console.print(f"❌ An unexpected error occurred: {e}", style="bold red")

if __name__ == "__main__":
    main()