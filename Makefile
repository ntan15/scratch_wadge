CC = mpicc
CXX = mpicxx
F77 = mpif77
FC = mpif90
CFLAGS = --std=gnu11 -g -O2

CFLAGS +=-fno-common -fomit-frame-pointer

CFLAGS += -Wconversion -Wno-sign-conversion \
          -Wcast-align -Wchar-subscripts -Wall -W \
          -Wpointer-arith -Wwrite-strings -Wformat-security -pedantic \
          -Wextra -Wno-unused-parameter

# asd flags
DEPS_HEADERS += asd.h
DEPS_SOURCE  += asd.c
CFLAGS +=-DASD_DEBUG
CFLAGS +=-DASD_USE_LUA

# list of libraries to build
TPLS ?= lua occa

UNAME_S := $(shell uname -s)

# Lua flags
CPPFLAGS += -Ilua/src
LDFLAGS += -Llua/src
LDLIBS += -llua -lm

# occa flags
CPPFLAGS += -Iocca/include
LDFLAGS += -Locca/lib
LDLIBS += -locca
ifeq ($(UNAME_S),Linux)
 LDFLAGS += -Wl,-rpath=$(CURDIR)/occa/lib,--enable-new-dtags
endif

all: euler2d

lua:
	tar xzf vendor/lua-*.tar.gz && mv lua-* lua
	cd lua && $(MAKE) CC=$(CC) posix

occa:
	tar xzf vendor/occa-*.tar.gz && mv occa-* occa
	cd occa && $(MAKE) OCCA_DEVELOPER=1 DEBUG=1 CC=$(CC) CXX=$(CXX) FC=$(FC)

# Dependencies
euler2d: euler2d.c $(DEPS_SOURCE) $(DEPS_HEADERS)  | $(TPLS)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH) \
        $< $(DEPS_SOURCE) $(LOADLIBES) $(LDLIBS) -o $@

# Rules
.PHONY: clean realclean
clean:
	rm -rf euler2d *.o

realclean:
	rm -rf $(TPLS)
	git clean -X -d -f
