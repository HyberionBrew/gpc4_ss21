//
// Created by daniel on 20.05.21.
//

#include "main.h"
#include "Writer.h"

IntStream createTestStreamInt(size_t s, size_t c){
    IntStream testStream;
    for (size_t i = 2; i < s; i++) {
        IntEvent x{i,(int) (s-i*c)};
        testStream.stream.push_back(x);
    }
    return testStream;
}

UnitStream createTestStreamUnit(size_t s) {
    UnitStream testStream;
    for (size_t i=0; i<s; i++) {
        UnitEvent ue{i};
        testStream.stream.push_back(ue);
    }
    return testStream;
}

UnitStream createUnitTestStreamLast(){
    UnitStream testStream;
    UnitEvent ue{02};
    testStream.stream.push_back(ue);
    UnitEvent ue1{2};
    testStream.stream.push_back(ue1);
    UnitEvent ue2{4};
    testStream.stream.push_back(ue2);
    UnitEvent ue3{5};
    testStream.stream.push_back(ue3);
    UnitEvent ue4{9};
    testStream.stream.push_back(ue4);
    return testStream;
}

IntStream createIntTestStreamLast(){
    IntStream testStream;
    IntEvent x{3,1};
    testStream.stream.push_back(x);
    IntEvent y{6,3};
    testStream.stream.push_back(y);
    IntEvent z{8,6};
    testStream.stream.push_back(z);
    return testStream;
}



UnitStream createUnitTestStreamDelay(){
    UnitStream testStream;
    UnitEvent ue{2};
    testStream.stream.push_back(ue);
    UnitEvent ue1{7};
    testStream.stream.push_back(ue1);
    UnitEvent ue2{8};
    testStream.stream.push_back(ue2);
    return testStream;
}

IntStream createIntTestStreamDelay(){
    IntStream testStream;
    IntEvent x{1,1};
    testStream.stream.push_back(x);
    IntEvent y{2,2};
    testStream.stream.push_back(y);
    IntEvent z{3,4};
    testStream.stream.push_back(z);
    IntEvent x1{4,1};
    testStream.stream.push_back(x1);
    IntEvent x2{6,2};
    testStream.stream.push_back(x2);
    IntEvent x3{7,3};
    testStream.stream.push_back(x3);
    return testStream;
}

void delayTest(){
    UnitStream test1 = createUnitTestStreamDelay();
    IntStream test2 = createIntTestStreamDelay();
    UnitStream result2 = delay(test2,test1);
    printUnitStream(result2);
}

void lastTest(){
    UnitStream test1 = createUnitTestStreamLast();
    IntStream test2 = createIntTestStreamLast();
    IntStream result2 = last(test2,test1);
    printIntStream(result2);
}

void merge_test(){
    IntStream s1, s2;
    for(size_t i = 0; i < 50; i++){
        if (i % 4 == 0){
            IntEvent ev(i, 2*i);
            s1.stream.push_back(ev);
        }
        if (i % 5 == 0){
            IntEvent ev(i, 3*i);
            s2.stream.push_back(ev);
        }
    }
    printIntStream(s1);
    printIntStream(s2);
    IntStream merged_stream = merge(s1, s2);
    printIntStream(merged_stream);
    Writer wr("merge_output.txt");
    wr.addStream("s1", s1);
    wr.addStream("s2", s2);
    wr.addStream("merged", merged_stream);
    wr.writeOutputFile();
}

int main(int argc, char **argv){
    // Test streams
    merge_test();
    /*IntStream test1 = createTestStreamInt(10,1);
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
    //delayTest();
    lastTest();
    //printIntStream(result3);
    //writeOut.addIntStream("z1",result1);
    //writeOut.addIntStream("z2",result1);
    //writeOut.addUnitStream("z3",test_unit1);
    writeOut.writeOutputFile();*/
    return 0;
}



