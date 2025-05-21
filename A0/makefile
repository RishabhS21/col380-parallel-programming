CXX = g++
CXXFLAGS = -O3 -std=c++17 -pg
TARGET = main

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET) gmon.out

