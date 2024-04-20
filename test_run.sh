#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J snn_dev_run  
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem=40GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=03:30:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgdyplomanci5-gpu-a100
## Specyfikacja partycji
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus=1
## Plik ze standardowym wyjściem
#SBATCH --output="/net/tscratch/people/plgjakubcaputa/output_files/out/out_bad__rgb_snn_rep.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="/net/tscratch/people/plgjakubcaputa/output_files/err/err_bad_rgb_snn_rep.err"

# Function to update a parameter in the YAML file
update_param() {
    param="$1"
    new_value="$2"
    sed -i "s/^\($param\s*:\s*\).*/\1$new_value/" config.yml
}

ml CUDA/11.8

source $SCRATCH/venvs/spiking_env/bin/activate

cd $SCRATCH/snn_dvs/
srun python src/train_lightning.py

update_param "lr" "0.00001"
srun python src/train_lightning.py

update_param "lr" "0.0001"
update_param "weight_decay" "0.05"

srun python src/train_lightning.py

update_param "lr" "0.0001"
update_param "weight_decay" "0.2"

srun python src/train_lightning.py

update_param "lr" "0.001"
update_param "weight_decay" "0.05"

srun python src/train_lightning.py

update_param "lr" "0.001"
update_param "weight_decay" "0.2"
srun python src/train_lightning.py

update_param "lr" "0.001"
update_param "weight_decay" "0.05"
srun python src/train_lightning.py

update_param "lr" "0.0001"
update_param "weight_decay" "0.1"
srun python src/train_lightning.py

update_param "lr" "0.0001"
update_param "weight_decay" "0.1"
srun python src/train_lightning.py
