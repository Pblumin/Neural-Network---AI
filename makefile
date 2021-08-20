mainNN.exe: mainNN.o NN.o
	g++ -o mainNN.exe mainNN.o NN.o 

mainNN.o: mainNN.cpp NN.h
	g++ -c mainNN.cpp

NN.o: NN.cpp NN.h
	g++ -c NN.cpp

debug:
	g++ -g -o NNDebug.exe mainNN.cpp NN.cpp

clean:
	rm -f *.exe *.o *.stackdump *~

backup:
	test -d backups || mkdir backups
	cp *.cpp backups
	cp *.h backups
	cp Makefile backups
