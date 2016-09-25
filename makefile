cc = g++
IPATH= ../eigen3/
#IPATH= /home/huan/Project/eigen3/

ffnn_main : ffnn_main.o ffnn.o
	$(cc) -I $(IPATH) ffnn_main.o ffnn.o -o ffnn_main

ffnn_main.o : ffnn_main.cpp ffnn.h
	$(cc) -I $(IPATH) -c ffnn_main.cpp
ffnn.o : ffnn.cpp ffnn.h
	$(cc) -I $(IPATH) -c ffnn.cpp

.PHONY : clean
clean :
	rm ffnn_main *.o
