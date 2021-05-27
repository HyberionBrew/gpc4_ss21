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
UnitStream createUnitTestStreamDelay(){

    UnitEvent ue{2};
    testStream.push_back(ue);
    UnitEvent ue{7};
    testStream.push_back(ue);
    UnitEvent ue{8};
    testStream.push_back(ue);
}

IntStream createIntTestStreamDelay(){
    IntEvent x{1,1};
    testStream.push_back(x);
    IntEvent x{2,2};
    testStream.push_back(x);
    IntEvent x{3,4};
    testStream.push_back(x);
    IntEvent x{4,1};
    testStream.push_back(x);
    IntEvent x{6,2};
    testStream.push_back(x);
    IntEvent x{7,3};
    testStream.push_back(x);
}

void delayTest()IntStream test2 = createIntTestStreamDelay();
    UnitStream test1 = createUnitTestStreamDelay();
    IntStream test2 = createIntTestStreamDelay();

}
/*
void merge_test(){
    for(int i = 0; i < 50; i++){
            s1.push_back(i);
        }
    }
}*/

int main(int argc, char **argv){
    // Test streams
    IntStream test1 = createTestStreamInt(10,1);
    IntStream test2 = createTestStreamInt(15,3);
    UnitStream test_unit1 = createTestStreamUnit(7);
    IntStream test_unit2 = createTestStreamInt(7,8);
    IntStream result1 = time(test1);
    UnitStream result2 = delay(test1,test_unit1);
    IntStream result3 = last(test1,test_unit1);
    printIntStream(result1);
    printUnitStream(result2);
    Writer writeOut = Writer("output.txt");
    printIntStream(result1);
    printUnitStream(test_unit1);

    //printIntStream(result3);
    //writeOut.addIntStream("z1",result1);
    //writeOut.addIntStream("z2",result1);
    //writeOut.addUnitStream("z3",test_unit1);
    writeOut.writeOutputFile();
    return 0;
}



