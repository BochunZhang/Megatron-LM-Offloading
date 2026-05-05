import os

current_dir = os.path.dirname(os.path.abspath(__file__))
tracer_path = os.path.join(current_dir, 'tracer.py')

import pycallgraph2.tracer as tracer_module

import importlib.util
spec = importlib.util.spec_from_file_location("local_tracer", tracer_path)
local_tracer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_tracer)

tracer_module.TraceProcessor = local_tracer.TraceProcessor

from output import TextOutput

from pycallgraph2 import PyCallGraph, Config
