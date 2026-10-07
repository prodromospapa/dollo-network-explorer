#!/bin/bash
tail -n 0 -f pipeline/full_eggnog_pipeline/pipeline.log | awk '
/Pipeline finished at/ { print "PIPELINE_COMPLETE"; exit 0 }
/Traceback/ { print "PIPELINE_ERROR"; exit 1 }
/Error:/ { print "PIPELINE_ERROR"; exit 1 }
'
