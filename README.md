# UI_Search_Engine
## Requirement

```
python3 -m pip install -r requirements.txt
```

## Training

```
python3 train.py \
--data_root "train_data" \
--cache_root "results/run/cache" \
--model_dir "results/run" \
--epochs 10
```

## Testing

```
python3 test.py \
--data_root "." \
--cache_root "results/run/cache" \
--model_dir "results/run" \
--checkpoint "results/run/state_dict_final.pth" \
--use_flann 0 \
--test_dir "test_data" \
--result_dir "results/test_result" \
```

