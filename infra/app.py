#!/usr/bin/env python3
import os
import json
import aws_cdk as cdk
import yaml
from forecaster_stack import ForecasterStack


def load_context_from_yaml(app: cdk.App, path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return
    # Apply each key as context
    for k, v in data.items():
        app.node.set_context(k, v)


app = cdk.App()
config_path = app.node.try_get_context("config") or "stack-config.yaml"
load_context_from_yaml(app, config_path)

ForecasterStack(
    app,
    "ForecasterStack",
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION", "eu-west-1"),
    ),
)

app.synth()
