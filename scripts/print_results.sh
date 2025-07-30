# 222
seed=$1
python scripts/print_result.py --root_dir_path warehouse/cluster | grep $seed | grep 'zero' | tee /dev/tty | wc -l
echo "----------------------------------------"
python scripts/print_result.py --root_dir_path warehouse/cluster | grep $seed | grep 'random' | tee /dev/tty | wc -l
echo "----------------------------------------"
python scripts/print_result.py --root_dir_path warehouse/with_pe | grep $seed | grep 'num_steps' | tee /dev/tty | wc -l
echo "----------------------------------------"
python scripts/print_result.py --root_dir_path "warehouse/patience=200" | grep $seed | grep 'num_steps' | tee /dev/tty | wc -l