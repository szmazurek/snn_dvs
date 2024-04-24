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
#SBATCH --time=04:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgdyplomanci5-gpu-a100
## Specyfikacja partycji
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus=1
## Plik ze standardowym wyjściem
#SBATCH --output="/net/tscratch/people/plgjakubcaputa/output_files/out/output.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="/net/tscratch/people/plgjakubcaputa/output_files/err/error.err"

# Function to update a parameter in the YAML file
update_param() {
    param="$1"
    new_value="$2"
    sed -i "/^\s*$param\s*:/s/:.*$/: $new_value/" config.yml
}

replace_slash() {
    sed -i 's/SLASH/\//g' "$1"
}

ml CUDA/11.8

source $SCRATCH/venvs/spiking_env/bin/activate

cd $SCRATCH/snn_dvs/

srun python src/train_lightning.py

update_param "n_samples" "3,"
update_param "repeats" "3,"
srun python src/train_lightning.py


update_param "n_samples" "2,"
update_param "repeats" "2,"
srun python src/train_lightning.py

update_param "  root_path" ".SLASHdatasetsSLASHdataset_jaadSLASHbad_weatherSLASH,"
replace_slash "config.yml"
srun python src/train_lightning.py

update_param "n_samples" "3,"
update_param "repeats" "3,"
srun python src/train_lightning.py

update_param "n_samples" "10,"
update_param "repeats" "10,"
srun python src/train_lightning.py



