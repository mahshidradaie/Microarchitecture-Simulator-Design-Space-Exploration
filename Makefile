CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Isrc

.PHONY: all bench test experiments clean
all: build/archsim

build/archsim: src/main.cpp src/isa.hpp src/cpu.hpp src/bpred.hpp src/cache.hpp src/core.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) src/main.cpp -o $@

build/test_units: tests/test_units.cpp src/*.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) tests/test_units.cpp -o $@

bench:
	python3 scripts/benchmarks.py build/bench

test: build/archsim build/test_units bench
	./build/test_units
	python3 scripts/run_tests.py

experiments: build/archsim bench
	python3 scripts/run_experiments.py

clean:
	rm -rf build
