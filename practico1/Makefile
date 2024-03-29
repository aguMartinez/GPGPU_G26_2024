#Makefile del Práctico 1 | GPGPU 2024 | FING | Udelar
all: main

# Objetivos que no son archivos.
.PHONY: all clean rebuild


# directorios
HDIR    = include
CDIR  = src
ODIR    = obj

MODULOS = utils cacheUtils ejercicio1 ejercicio2 
PUNTOH = utils cacheUtils ejercicio1 ejercicio2

# lista de archivos, con directorio y extensión
HS   = $(PUNTOH:%=$(HDIR)/%.h)
CPPS = $(MODULOS:%=$(CDIR)/%.c)
OS   = $(MODULOS:%=$(ODIR)/%.o)

PRINCIPAL=main
EJECUTABLE=main

LIB=prct1GPGPU.a

# compilador
CC = g++
# opciones de compilación
CCFLAGS = -O0 -Wall -g -I$(HDIR)


$(ODIR)/$(PRINCIPAL).o:$(PRINCIPAL).c
	$(CC) $(CCFLAGS) -c $< -o $@

# cada .o depende de su .c
# $@ se expande para tranformarse en el objetivo
# $< se expande para tranformarse en la primera dependencia
$(ODIR)/%.o: $(CDIR)/%.c $(HDIR)/%.h
	$(CC) $(CCFLAGS) -c $< -o $@

$(LIB):$(ODIR)/$(PRINCIPAL).o $(OS)
	ar -qc $@ $^

biblioteca:$(LIB)

# $^ se expande para tranformarse en todas las dependencias
$(EJECUTABLE): $(ODIR)/$(PRINCIPAL).o $(OS)
	$(CC) $(CCFLAGS) $^ -o $@
	
clean:
	rm -f $(EJECUTABLE) $(ODIR)/$(PRINCIPAL).o $(OS) $(LIB)
	
	
rebuild:
	make clean
	make