.SUFFIXES: .c .cpp

CC=g++-4.0
CFLAGS=-I . -I /usr/include/ -g -O2
LIBS = -L/usr/lib  


%.o:	%.cpp
	$(CC) -c -o $@ $(CFLAGS) $<

%.o:	%.c
	gcc -c -o $@ $(CFLAGS) $<

%.o:	%.f
	g77 -c -o $@ $<

target = fastcrf
all: $(target)

fastcrf: Main.o LinearCRF.o LinearChain.o Evaluator.o Parameter.o Data.o LBFGS.o Utility.o
	$(CC) -o $@ Main.o LinearCRF.o LinearChain.o Evaluator.o Parameter.o Data.o LBFGS.o Utility.o $(CFLAGS) $(LIBS)
	
clean:
	rm $(target) *.o 

