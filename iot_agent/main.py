"""
    This script implements the Iteration of Thought (IoT) and Generative Iteration of Thought (GIoT) models.
"""

import os
import time
import signal
from typing import Optional
from loguru import logger
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.markdown import Markdown
from litellm import completion

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"  # Updated to a more recent model
RATE_LIMIT_WAIT_TIME = 10
GLOBAL_TIMEOUT = 300

if not API_KEY:
    raise ValueError("OpenAI API key must be set as an environment variable.")

console = Console()


class IterationOfThought:
    """
    Class for performing Iteration of Thought (IoT) and Generative Iteration of Thought (GIoT).
    """

    def __init__(self, model: str = MODEL, max_iterations: int = 5, timeout: int = 30):
        """
        Initialize the IterationOfThought class.

        Args:
            model (str): The model to use for the LLM.
            max_iterations (int): The maximum number of iterations to perform.
            timeout (int): The timeout for each iteration in seconds.
        """
        self.model = model
        self.max_iterations = max_iterations
        self.timeout = timeout  # Ensure this line is included

    def _call_llm(
        self, prompt: str, temperature: float = 0.5, max_retries: int = 3
    ) -> str:
        """
        Call the OpenAI API with the given prompt and return the response.

        Args:
            prompt (str): The prompt to send to the API.
            temperature (float): Sampling temperature for the API response.
            max_retries (int): Number of retries in case of failure.

        Returns:
            str: The content of the API response.
        """
        for _ in range(max_retries):  # Changed 'attempt' to '_'
            try:
                with console.status(
                    "[bold green]Calling OpenAI API...", spinner="dots"
                ):
                    response = completion(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                return response["choices"][0]["message"]["content"].strip()
            # pylint: disable=broad-except
            except Exception as e:
                console.print(f"[red]Error: {e}")
                return ""
        console.print("[red]Failed to get a response from OpenAI API after max retries")
        return ""

    def inner_dialogue_agent(self, query: str, previous_response: str) -> str:
        """
        Generate a new prompt based on the original query and previous response.

        Args:
            query (str): The original user query.
            previous_response (str): The previous response from the LLM.

        Returns:
            str: The generated prompt for the next iteration.
        """
        prompt = (
            f"Given the original query: '{query}' and the previous response: '{previous_response}', "
            "generate an instructive and context-specific prompt to refine and improve the answer. "
            "Ensure that the new prompt encourages deeper reasoning or addresses any gaps in the previous response."
        )
        return self._call_llm(prompt)

    def llm_agent(self, query: str, prompt: str) -> str:
        """
        Call the LLM agent with the given query and prompt.

        Args:
            query (str): The user query.
            prompt (str): The prompt to refine the response.

        Returns:
            str: The response from the LLM agent.
        """
        full_prompt = f"Query: {query}\nPrompt: {prompt}\nResponse:"
        return self._call_llm(full_prompt)

    def stopping_criterion(self, response: str) -> bool:
        """
        Determine if the stopping criterion has been met based on the response.

        Args:
            response (str): The response from the LLM.

        Returns:
            bool: True if the stopping criterion is met, False otherwise.
        """
        lower_response = response.lower()
        return any(
            keyword in lower_response
            for keyword in [
                "answer:",
                "final answer:",
                "conclusion:",
                "summary:",
                "the answer is:",
            ]
        )

    def aiot(self, query: str) -> str:
        """
        Execute the AIoT process for the given query.

        Args:
            query (str): The user query to process.

        Returns:
            str: The final response after iterations.
        """
        console.print("\n[bold cyan]Starting AIoT...[/bold cyan]")
        current_response = self.llm_agent(query, "Initial Prompt")

        for iteration in range(1, self.max_iterations + 1):
            console.print(f"\n[bold]Iteration {iteration}:[/bold]")
            console.print(Panel(current_response, title="LLMA Response", expand=False))

            if self.stopping_criterion(current_response):
                console.print("[green]Stopping criterion met.[/green]")
                break

            new_prompt = self.inner_dialogue_agent(query, current_response)
            console.print(Panel(new_prompt, title="IDA Generated Prompt", expand=False))
            current_response = self.llm_agent(query, new_prompt)
            time.sleep(self.timeout)

        console.print("[bold cyan]AIoT completed.[/bold cyan]\n")
        return current_response

    def giot(self, query: str, fixed_iterations: int) -> str:
        """
        Execute the GIoT process for the given query with a fixed number of iterations.

        Args:
            query (str): The user query to process.
            fixed_iterations (int): The number of iterations to perform.

        Returns:
            str: The final response after iterations.
        """
        console.print("\n[bold magenta]Starting GIoT...[/bold magenta]")
        current_response = self.llm_agent(query, "Initial Prompt")

        for iteration in range(1, fixed_iterations + 1):
            console.print(f"\n[bold]Iteration {iteration}:[/bold]")
            console.print(Panel(current_response, title="LLMA Response", expand=False))

            new_prompt = self.inner_dialogue_agent(query, current_response)
            console.print(Panel(new_prompt, title="IDA Generated Prompt", expand=False))
            current_response = self.llm_agent(query, new_prompt)
            time.sleep(self.timeout)

        console.print("[bold magenta]GIoT completed.[/bold magenta]\n")
        return current_response


def get_user_query() -> str:
    sample_query = (
        "A textile dye containing an extensively conjugated pi-electrons emits light with energy of 2.3393 eV. "
        "What color of light is absorbed by the organic compound? Pick an answer from the following options:\n"
        "A. Red\nB. Yellow\nC. Blue\nD. Violet"
    )

    console.print(
        Panel.fit("Enter your query (or press Enter to use the sample query):")
    )
    user_input = Prompt.ask("Query", default=sample_query, show_default=True)

    if not user_input.strip():
        console.print("[yellow]No input provided. Using sample query.[/yellow]")
        return sample_query

    return user_input


def timeout_handler(signum, frame):
    """
    Handle the timeout signal.

    Args:
        signum (int): The signal number.
        frame (FrameInfo): The frame information.
    """
    raise TimeoutError("Process took too long")


def interrupt_handler(signum, frame):
    """
    Handle the interrupt signal.

    Args:
        signum (int): The signal number.
        frame (FrameInfo): The frame information.
    """
    raise KeyboardInterrupt("User interrupted the process")


def run_iot(iot: IterationOfThought, query: str, method: str) -> str:
    """
    Run the specified IoT method (AIoT or GIoT) and return the result.

    Args:
        iot (IterationOfThought): The IoT instance to use.
        query (str): The user query to process.
        method (str): The method to run ("AIoT" or "GIoT").

    Returns:
        str: The result of the IoT method.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Running {method}...", total=None)
        result = (
            iot.aiot(query) if method == "AIoT" else iot.giot(query, fixed_iterations=3)
        )
        progress.update(task, completed=True)
    return result


def display_results(aiot_result: Optional[str], giot_result: Optional[str]):
    """
    Display the results of the IoT methods.

    Args:
        aiot_result (Optional[str]): The result of the AIoT method.
        giot_result (Optional[str]): The result of the GIoT method.
    """
    table = Table(title="Results Comparison")
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Response", style="magenta")

    if aiot_result:
        table.add_row("AIoT", aiot_result)
    if giot_result:
        table.add_row("GIoT", giot_result)

    console.print(table)


@click.command()
@click.option(
    "--method",
    type=click.Choice(["AIoT", "GIoT", "both"]),
    default="both",
    help="Choose the method to run",
)
@click.option("--verbose", is_flag=True, help="Enable verbose output")
def main(method: str, verbose: bool) -> None:
    """Main entry point for the IoT application.

    Args:
        method (str): The method to run (AIoT, GIoT, or both).
        verbose (bool): Flag to enable verbose output.
    """
    if verbose:
        logger.add("debug.log", level="DEBUG")
    else:
        logger.remove()
        logger.add(lambda _: None)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.signal(signal.SIGINT, interrupt_handler)

    iot = IterationOfThought(model=MODEL, max_iterations=5, timeout=2)
    query = get_user_query()
    console.print(
        Panel(Markdown(f"**Query:** {query}"), title="Input Query", expand=False)
    )

    try:
        signal.alarm(GLOBAL_TIMEOUT)

        aiot_result = None
        giot_result = None

        if method in ["AIoT", "both"]:
            aiot_result = run_iot(iot, query, "AIoT")

        if method in ["GIoT", "both"]:
            giot_result = run_iot(iot, query, "GIoT")

        display_results(aiot_result, giot_result)

        signal.alarm(0)
    except TimeoutError:
        console.print("⚠️ [bold red]Process timed out. Please try again.[/bold red]")
    except KeyboardInterrupt:
        console.print("✋ [bold yellow]Process interrupted by user.[/bold yellow]")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        console.print(f"❌ [bold red]An unexpected error occurred: {e}[/bold red]")


if __name__ == "__main__":
    main()
