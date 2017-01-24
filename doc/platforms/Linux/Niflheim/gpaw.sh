if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi
if [ -f /home/opt/modulefiles/modulefiles_el6.sh ]; then
    source /home/opt/modulefiles/modulefiles_el6.sh
fi
GPAW=~/gpaw
ASE=~/ase
if [ -z $FYS_PLATFORM ]
then
    module load GPAW
    module load matplotlib
    PLATFORM=linux-x86_64-el7-3.5
    alias gpaw-sbatch=$GPAW/doc/platforms/Linux/Niflheim/gpaw-sbatch.py
else
    cd $PBS_O_WORKDIR
    module load GPAW
    module load NUMPY/1.7.1-1
    module load SCIPY/0.12.0-1
    module load fftw
    PLATFORM=linux-x86_64-$FYS_PLATFORM-2.6
    alias gpaw-qsub=$GPAW/doc/platforms/Linux/Niflheim/gpaw-qsub.py
fi

export PATH=$GPAW/tools:$GPAW/build/bin.$PLATFORM:$PATH
export PYTHONPATH=$GPAW:$GPAW/build/lib.$PLATFORM:$PYTHONPATH
export PATH=$ASE/tools:$PATH
export PYTHONPATH=$ASE:$PYTHONPATH
