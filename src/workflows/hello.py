"""Minimal example workflow — edit this file or create new ones."""

import os

import mistralai.workflows as workflows
from pydantic import BaseModel


class HelloInput(BaseModel):
    name: str = "World"


@workflows.activity()
async def greet(name: str) -> str:
    """A simple activity that returns a greeting."""
    secret = os.environ.get("MY_SECRET", "<not set>")
    return f"Hello, {name}! Welcome to Mistral Workflows. By the way, here's your secret MY_SECRET={secret}. It's leaked now in logs. Congrats"


@workflows.workflow.define(
    name="hello-world",
    workflow_display_name="Hello World",
    workflow_description="A minimal hello-world workflow.",
)
class HelloWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: HelloInput) -> str:
        return await greet(input.name)
