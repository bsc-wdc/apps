# $1 = Numero de elementos (10M, 100M, 1B)
# $2 = Numero de centros (100)
# $3 = Dimensiones (1000, 100, 10)
# $4 = Fragmentos (512)

python src/generator_files.py $1 $2 $3 $4
