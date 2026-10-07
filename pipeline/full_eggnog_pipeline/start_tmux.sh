#!/bin/bash
tmux new-session -d -s eggnog_full "bash pipeline/full_eggnog_pipeline/run_pipeline.sh"
echo "Started pipeline in tmux session 'eggnog_full'."
echo "To view progress, run: tmux attach -t eggnog_full"
echo "To detach again, press Ctrl+B then D."
echo "Logs are being written to pipeline/full_eggnog_pipeline/pipeline.log"
