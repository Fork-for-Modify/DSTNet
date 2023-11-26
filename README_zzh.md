## Multi-GPU train:
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4328 basicsr/train.py -opt options/train/Deblur/train_Deblur_GOPRO_inr.yml --launcher pytorch


### Testing
```
python basicsr/test.py -opt options/test/Deblur/test_Deblur_GOPRO_inr.yml
cd results
python merge_full.py
python calculate_psnr.py
```
- Before running merge_full.py, you should change the parameters in this file of Line 5,6,7,8.
- The deblured result will be in `'./results/dataset_name/'`.
- Before running calculate_psnr.py, you should change the parameters in this file of Line 5,6.
- We calculate PSNRs/SSIMs by running calculate_psnr.py