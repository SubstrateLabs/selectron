#!/usr/bin/env python
import os
import subprocess
import sys

from rich.console import Console
from rich.prompt import Confirm

ABOUT_FILE = "src/selectron/__about__.py"
PYPI_TOKEN_ENV_VAR = "PYPI_TOKEN"

console = Console()


def run_command(command: list[str], cwd: str | None = None) -> bool:
    """runs a command using subprocess and prints output/errors using rich."""
    command_str = " ".join(command)
    console.print(f"> running: [bold cyan]{command_str}[/]")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            bufsize=1,  # line buffered
        )

        # stream stdout
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                console.print(f"[grey50]stdout |[/] {line.strip().lower()}")  # lowercase output
            process.stdout.close()

        # stream stderr
        stderr_output = []
        if process.stderr:
            for line in iter(process.stderr.readline, ""):
                stderr_output.append(line.strip())
                console.print(f"[bold red]stderr |[/] {line.strip().lower()}")  # lowercase output
            process.stderr.close()

        process.wait()
        return_code = process.returncode

        if return_code == 0:
            console.print(f"cmd success: [bold green]{command_str}[/]")
            return True
        else:
            console.print(f"cmd failed w/ exit code {return_code}: [bold red]{command_str}[/]")
            # print captured stderr again just in case streaming missed something or for emphasis
            if stderr_output:
                console.print("[bold red]--- captured stderr ---[/]")
                for line in stderr_output:
                    console.print(f"[red]{line.lower()}[/]")  # lowercase output
                console.print("[bold red]--- end stderr ---[/]")
            return False

    except FileNotFoundError:
        console.print(
            f"[bold red]error: command not found - {command[0]}. is hatch installed and in path??[/]"
        )
        return False
    except Exception:
        console.print("[bold red]unexpected error occurred:[/]")
        console.print_exception(show_locals=False)
        return False


def main():
    # ensure script is run from the intended directory (cli/)
    if not os.path.exists("pyproject.toml") or not os.path.exists(ABOUT_FILE):
        console.print(
            f"[bold red]error: run this script from the 'cli' dir pls (needs 'pyproject.toml' and '{ABOUT_FILE}').[/]"
        )
        sys.exit(1)

    console.rule("[bold yellow]step 1: update version[/]")
    console.print(f"remember to update the version number in: [bold cyan]{ABOUT_FILE}[/]")
    if not Confirm.ask("did you update version?", default=False):
        console.print("[yellow]publish cancelled.[/]")
        sys.exit(0)

    console.rule("[bold yellow]step 2: check pypi token[/]")
    pypi_token = os.getenv(PYPI_TOKEN_ENV_VAR)
    if not pypi_token:
        console.print(f"[bold red]error: env var {PYPI_TOKEN_ENV_VAR} not set.[/]")
        sys.exit(1)
    console.print(f"[green]pypi token found ({PYPI_TOKEN_ENV_VAR})[/]")

    console.rule("[bold yellow]step 3: confirmation[/]")
    if not Confirm.ask("publish?", default=False):
        console.print("[yellow]cancelled.[/]")
        sys.exit(0)

    console.rule("[bold yellow] step 4: build[/]")
    build_command = ["hatch", "build", "-t", "wheel"]
    if not run_command(build_command):
        console.print("[bold red]build step failed :/ aborting publish.[/]")
        sys.exit(1)

    console.rule("[bold yellow]step 4.5: verify build artifacts[/]")
    dist_files = os.listdir("dist")
    sdist_files = [f for f in dist_files if f.endswith(".tar.gz")]
    wheel_files = [f for f in dist_files if f.endswith(".whl")]

    if sdist_files:
        console.print(
            f"[bold red]error: source distribution found in dist/: {sdist_files}. aborting.[/]"
        )
        console.print("ensure build command only targets wheels (e.g., 'hatch build -t wheel').")
        sys.exit(1)

    if not wheel_files:
        console.print("[bold red]error: no wheel files found in dist/ after build. aborting.[/]")
        sys.exit(1)

    console.print(f"[green]verified: only wheel artifacts found in dist/: {wheel_files}[/]")

    console.rule("[bold yellow]step 5: publish[/]")
    publish_command = ["hatch", "publish", "-u", "__token__", "-a", pypi_token]
    if not run_command(publish_command):
        console.print("[bold red]publish step failed :/[/]")
        sys.exit(1)

    console.rule("[bold green]publish complete![/]")


if __name__ == "__main__":
    main()
