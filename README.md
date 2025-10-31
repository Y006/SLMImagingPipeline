在根目录运行 `python -m src.pipeline.single_shot`

掩膜生成命令：

```shell
python gen_mask.py --type coded --method MLS --n_bits 8
python gen_mask.py --type coded --method MURA --n_bits 11
python gen_mask.py --type phase --noise_period 8 8 --n_iter 15
python gen_mask.py --type random --fill_ratio 0.6
python gen_mask.py --type fza --radius 0.56e-3
```

