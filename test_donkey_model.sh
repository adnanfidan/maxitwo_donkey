model_name=$1

python test_donkey_model.py --env-name donkey \
    --donkey-exe-path /Users/adnanfidan/Documents/Thesis/simulators/donkeysim-maxibon-macos.app \
    --seed 0 --num-episodes 50 \
    --agent-type supervised \
    --model-path models/new_model_v1.ckpt \
    --max-angle 270 \
    --no-save-archive








