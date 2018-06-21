# $1 = Number of points (10M, 100M, 1B)
# $2 = Number of centers (100)
# $3 = Dimensions (1000, 100, 10)
# $4 = Fragments (512)

python src/generator_multiplefiles.py $1 $2 $3 $4
