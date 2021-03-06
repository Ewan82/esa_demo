#!/bin/sh

rm -r tool_intro/example_synthetic
cp -r retrieval_tool_examples/ret_code1.7 tool_intro/example_synthetic
( cd tool_intro/example_synthetic;
  ln -fs ../../cost_fn.png .; make simulate.x runretrieval.x )

rm -rf mba_intro/example_mba
cp -r retrieval_tool_examples/ret_code1.7 mba_intro/example_mba
cp    mba_intro/target_schedule_S??_2017.asc mba_intro/example_mba/input
(cd   mba_intro/example_mba;
 ln -fs ../Make-mba.inc Make-debug.inc;
 ln -fs ../../explist_post.py .)
(cd   mba_intro/example_mba; make setup; make simulate.x runua.x)
