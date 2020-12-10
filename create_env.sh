conda config --add channels conda-forge

conda create -y --name recsys-nnmf --file requirements.txt

source activate recsys-nnmf

pip install similarity
pip install telepot
pip install -e git+https://github.com/changyaochen/rbo.git@master#egg=rbo