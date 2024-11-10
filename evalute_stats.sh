#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J tests
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem=40GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:02:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgdyplomanci5-gpu-a100
## Specyfikacja partycji
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus=1
## Plik ze standardowym wyjściem
#SBATCH --output="/net/tscratch/people/plgjakubcaputa/output_files/out/2024_10_28_%j.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="/net/tscratch/people/plgjakubcaputa/output_files/out/2024_10_28_%j.out"


ml CUDA/11.8

source $SCRATCH/venvs/spiking_env/bin/activate

cd $SCRATCH/snn_dvs/

srun python src/energy_usage.py


