#!/bin/bash
GAMES="bigfish bossfight coinrun starpilot caveflyer dodgeball fruitbot chaser"
count=0
for game in $GAMES; do
    echo 'CUDA_VISIBLE_DEVICES='$count' python ppo_clip_train.py --env_name' $game '--vision_mode normal --experiment_name clip_v0 --num_processes 256 --seg'
    echo 'tmux new-session -d -s' clip$count
    
    tmux new-session -d -s clip$count 'CUDA_VISIBLE_DEVICES='"$count"' python ppo_clip_train.py --env_name '"$game"' --vision_mode normal --experiment_name clip_v0 --num_processes 256 --seg'
    count=$((count+1))
done
