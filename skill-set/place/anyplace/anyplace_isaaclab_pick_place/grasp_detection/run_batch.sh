source ~/.bashrc

exps=($(ls ~/gpp/general_pick_place/isaac_lab_pick_place/data/hanging | grep scaled))

for exp in "${exps[@]}"
do
  python main.py --exp ~/gpp/general_pick_place/isaac_lab_pick_place/data/hanging/$exp
done

