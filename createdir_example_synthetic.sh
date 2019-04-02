#!/bin/sh

rm -r tool_intro/example_synthetic
cp -r retrieval_tool_examples/ret_code1.7 tool_intro/example_synthetic
(cd tool_intro/example_synthetic; ln -fs ../../cost_fn.png .)
