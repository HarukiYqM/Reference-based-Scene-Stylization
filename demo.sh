REF_STY="../refs/flower/flower_blue.png"
SCENE_ID=-1
python train_ref.py --eval -s ../refs/flower_llff -m ../output/flower_blue --convert_SHs_python --sh_degree 3 --start_checkpoint ../refs/flower_final.pth --iterations 3000 --densify_until_iter 1500 --ref_img ${REF_STY} --scene_id ${SCENE_ID} --densify_grad_threshold 5e-5
python render.py -m ../output/flower_blue



