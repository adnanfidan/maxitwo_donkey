model_name=$1

python test_donkey_model.py --env-name donkey \
    --donkey-exe-path /Users/adnanfidan/Documents/Thesis/simulators/donkeysim-maxibon-macos.app \
    --seed 1 --num-episodes 50 \
    --agent-type supervised \
    --model-path models/last_test.ckpt \
    --max-angle 90 \
    --no-save-archive








