all:
	g++ -Wall -std=c++11 -fopenmp main.cpp utility.cpp lr.cpp -o main 

clean:
	rm main

test_seismic_parallel:
	./main -t seismic_train.csv -p seismic_test.csv -nr 16000 -nc 50 -nl 32 -tr 4000 -tl 50 -i 20 -a 0.01 -l 0.005 -seq 0 -nthread 1

test_seismic_parallel16:
	./main -t seismic_train.csv -p seismic_test.csv -nr 16000 -nc 50 -nl 32 -tr 4000 -tl 50 -i 5 -a 0.01 -l 0.005 -seq 0 -nthread 32

test_seismic:
	./main -t seismic_train.csv -p seismic_test.csv -nr 16000 -nc 50 -nl 32 -tr 4000 -tl 50 -i 20 -a 0.01 -l 0.005 -seq 1 -nthread 1

test_wine:
	./main -t wine_train.csv -p wine_test.csv -nr 3526 -nc 11 -nl 7 -tr 882 -tl 11 -i 100 -a 0.001 -l 0.005