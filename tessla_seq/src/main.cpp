//
// Created by daniel on 20.05.21.
//

#include "main.h"
#include "Writer.h"

IntStream createTestStreamInt(size_t s, size_t c){
    IntStream testStream;
    for (size_t i = 2; i < s; i++) {
        IntEvent x{i,(int) (s-i*c)};
        testStream.push_back(x);
    }
    return testStream;
}

UnitStream createTestStreamUnit(size_t s) {
    UnitStream testStream;
    for (size_t i=0; i<s; i++) {
        UnitEvent ue{i};
        testStream.push_back(ue);
    }
    return testStream;
}

int main(int argc, char **argv){
    // Test streams
    IntStream test1 = createTestStreamInt(10,1);
    IntStream test2 = createTestStreamInt(15,3);
    UnitStream test_unit1 = createTestStreamUnit(7);
    IntStream test_unit2 = createTestStreamInt(7,8);
    IntStream result1 = time(test1);
    UnitStream result2 = delay(test1,test_unit1);
    printIntStream(result1);
    printUnitStream(result2);
    Writer writeOut = Writer("output.txt");
    writeOut.addIntStream("z1",result1);
    writeOut.addUnitStream("z2",result2);
    writeOut.writeOutputFile();
    return 0;
}



