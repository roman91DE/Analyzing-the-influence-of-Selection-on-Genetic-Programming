#!/opt/homebrew/bin/fish

conda activate "gp_research"         
cd ~/github/geneticProgramming/seminar/src || exit

set TOTAL_NUM_RUNS 3

for IND in (seq 0 $TOTAL_NUM_RUNS)
  python ./main.py y1 squared
end
