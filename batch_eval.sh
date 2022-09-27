GAMES="bigfish bossfight caveflyer chaser climber coinrun dodgeball fruitbot heist jumper leaper maze miner ninja plunder starpilot"
for GM in $GAMES
do
  #CUDA_VISIBLE_DEVICES=0 python eval_model.py --eval_env_name $GM --env_name $GM"p" --experiment_name $GM"_p75" --use_decoder --bottleneck 256 --BASE_PATH /media/data_cifs/projects/prj_procgen/lax/procgen/procgen/train-procgen/train_procgen/results
  
  #CUDA_VISIBLE_DEVICES=6 python eval_model.py --env_name $GM --experiment_name fg_p1_r1 --BASE_PATH /media/data_cifs/projects/prj_procgen/lax/prj-taxRL/train-procgen/train_procgen/results --vision_mode fg_mask --stochasticity 1. --output fg_p1_test --test_generalization

CUDA_VISIBLE_DEVICES=3 python eval_modelV2.py --env_name $GM --experiment_name pixels_p1_r1 --BASE_PATH /media/data_cifs/projects/prj_procgen/lax/prj-taxRL/train-procgen/train_procgen/results/ --vision_mode normal

CUDA_VISIBLE_DEVICES=3 python eval_modelV2.py --env_name $GM --experiment_name semantic_p1_r1 --BASE_PATH /media/data_cifs/projects/prj_procgen/lax/prj-taxRL/train-procgen/train_procgen/results/ --vision_mode semantic_mask

CUDA_VISIBLE_DEVICES=3 python eval_modelV2.py --env_name $GM --experiment_name pixels_p1_r1 --BASE_PATH /media/data_cifs/projects/prj_procgen/lax/prj-taxRL/train-procgen/train_procgen/results/ --vision_mode normal --test_generalization

CUDA_VISIBLE_DEVICES=3 python eval_modelV2.py --env_name $GM --experiment_name semantic_p1_r1 --BASE_PATH /media/data_cifs/projects/prj_procgen/lax/prj-taxRL/train-procgen/train_procgen/results/ --vision_mode semantic_mask --test_generalization

done
