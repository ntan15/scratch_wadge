
CC = mpicc
CXX = mpicxx
F77 = mpif77
FC = mpif90

CFLAGS = --std=c11 -g
CXXFLAGS = -std=c++11 -g

# below are flags that work with clang and gcc
CFLAGS += -Wconversion -Wno-sign-conversion \
          -Wcast-align -Wchar-subscripts -Wall -W \
          -Wpointer-arith -Wwrite-strings -Wformat-security -pedantic \
          -Wextra -Wno-unused-parameter

CXXFLAGS += -Wconversion -Wno-sign-conversion \
            -Wcast-align -Wchar-subscripts -Wall -W \
            -Wpointer-arith -Wwrite-strings -Wformat-security -pedantic \
            -Wextra -Wno-unused-parameter

DEBUG = 0
ifeq (${DEBUG},1)
  CFLAGS += -O1
  CFLAGS += -fno-optimize-sibling-calls -fno-omit-frame-pointer
  CFLAGS += -fsanitize=address,undefined

  CXXFLAGS += -O1
  CXXFLAGS += -fno-optimize-sibling-calls -fno-omit-frame-pointer
  CXXFLAGS += -fsanitize=address,undefined

  CPPFLAGS +=-DASD_DEBUG
  OCCA_MAKE_FLAGS = OCCA_DEVELOPER=1 DEBUG=1
else
  CFLAGS += -O2
  CFLAGS += -fno-common -fno-omit-frame-pointer

  CXXFLAGS += -O2
  CXXFLAGS += -fno-common -fno-omit-frame-pointer
endif

# asd flags
DEPS_HEADERS += asd.h types.h
DEPS_SOURCE  += asd.c
CPPFLAGS +=-DASD_USE_LUA

# list of libraries to build
TPLS ?= eigen lua occa

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

.PHONY: all clean debug realclean

all: eulertri eulertet

debug:
	$(MAKE) DEBUG=1

eigen:
	tar xzf vendor/eigen-*.tar.gz && mv eigen-* eigen

lua:
	tar xzf vendor/lua-*.tar.gz && mv lua-* lua
	cd lua && $(MAKE) CC=$(CC) posix

occa:
	tar xzf vendor/occa-*.tar.gz && mv occa-* occa
	cd occa && $(MAKE) $(OCCA_MAKE_FLAGS) CC=$(CC) CXX=$(CXX) FC=$(FC)

clean:
	rm -rf eulertri eulertet *.o

realclean:
	rm -rf $(TPLS)
	git clean -X -d -f

asd.o: asd.c asd.h | $(TPLS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

eulertri.o: CPPFLAGS+=-DELEM_TYPE=0
eulertri.o: euler.c operators_c.h $(DEPS_HEADERS)  | $(TPLS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

eulertet.o: CPPFLAGS+=-DELEM_TYPE=1
eulertet.o: euler.c operators_c.h $(DEPS_HEADERS)  | $(TPLS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

operators_c.o: CPPFLAGS+=-Ieigen
operators_c.o: operators_c.cpp operators_c.h operators.h $(DEPS_HEADERS) | $(TPLS)
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

# for building operators using eigen
operators.o: CPPFLAGS+=-Ieigen
operators.o: operators.cpp operators.h | $(TPLS)
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

eulertri: asd.o eulertri.o operators.o operators_c.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

eulertet: asd.o eulertet.o operators.o operators_c.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@
