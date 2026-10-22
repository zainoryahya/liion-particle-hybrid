# Build the exact modal-recursion library used by samsung/ss_thermal.py
#   Linux:   make            -> libmodrec.so
#   macOS:   make            -> libmodrec.so (loaded the same way)
#   Windows: gcc -O2 -shared -o libmodrec.so modrec.c   (MinGW; the .so name is kept on purpose)
CC ?= gcc
libmodrec.so: modrec.c
	$(CC) -O2 -fPIC -shared -o $@ $< -lm
clean:
	rm -f libmodrec.so
