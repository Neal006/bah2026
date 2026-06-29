CC     = gcc
CFLAGS = -O3 -march=native -std=c99 -Wall

all: cog_centroid.dll

cog_centroid.dll: cog_centroid.c
	$(CC) $(CFLAGS) -shared -o $@ $<

clean:
	rm -f cog_centroid.dll
