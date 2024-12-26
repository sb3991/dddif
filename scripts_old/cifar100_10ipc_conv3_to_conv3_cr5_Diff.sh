python ./main_dif.py \
--subset "cifar100" \
--arch-name "conv3" \
--factor 1 \
--num-crop 5 \
--mipc 200 \
--ipc 10 \
--stud-name "conv3" \
--re-epochs 4000 \
--classifier_scale 1 \
--prompt_strength  7.5 \
--exp_id "Guided" \
--num_inference_steps 50 \
# --exp_id "DPMSolverMultistepScheduler"
# --exp_id "DPMSolve_CADS"
# ./main_dif.py 