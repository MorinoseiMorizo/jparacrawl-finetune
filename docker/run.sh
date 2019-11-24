#!/bin/sh

docker run -it --gpus 1 -v ~/jparacrawl-experiments:/host_disk jparacrawl-fairseq bash
