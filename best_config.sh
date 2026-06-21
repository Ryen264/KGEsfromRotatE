# Best Configuration for ComplEx
# 
# bash run.sh train ComplEx FB15k 0 0 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002 -rp 3
bash run.sh train ComplEx FB15k 0 0 1024 256 1000 500.0 1.0 0.001 159 16 -de -dr -r 0.000002 -rp 3
# bash run.sh train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001 -rp 3
bash run.sh train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 188 16 -de -dr -r 0.00001 -rp 3
# bash run.sh train ComplEx wn18 0 0 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001 -rp 3
bash run.sh train ComplEx wn18 0 0 512 1024 500 200.0 1.0 0.001 144 8 -de -dr -r 0.00001 -rp 3
# bash run.sh train ComplEx wn18rr 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005 -rp 3
bash run.sh train ComplEx wn18rr 0 0 512 1024 500 200.0 1.0 0.002 235 8 -de -dr -r 0.000005 -rp 3
# bash run.sh train ComplEx countries_S1 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 -rp 3 --countries
bash run.sh train ComplEx countries_S1 0 0 512 64 1000 1.0 1.0 0.000002 6667 8 -de -dr -r 0.0005 -rp 3 --countries
# bash run.sh train ComplEx countries_S2 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 -rp 3 --countries
bash run.sh train ComplEx countries_S2 0 0 512 64 1000 1.0 1.0 0.000002 6667 8 -de -dr -r 0.0005 -rp 3 --countries
# bash run.sh train ComplEx countries_S3 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 -rp 3 --countries
bash run.sh train ComplEx countries_S3 0 0 512 64 1000 1.0 1.0 0.000002 10000 8 -de -dr -r 0.0005 -rp 3 --countries
#
