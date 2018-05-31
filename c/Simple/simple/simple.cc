/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#include<iostream>
#include<fstream>
#include<string>
#include"simple.h"

using namespace std;

#define FILE_NAME "counter"

void usage() {
    cerr << "[ERROR] Bad numnber of parameters" << endl;
    cout << "    Usage: simple <counterValue>" << endl;
}

int main(int argc, char *argv[]) {
    // Check and get parameters
    if (argc != 2) {
        usage();
        return -1;
    }
    string initialValue = argv[1];
    file fileName = strdup(FILE_NAME);
    
    // Init compss
    compss_on();
    
    //compss_test();

    // Write file
    ofstream fos;
    compss_ofstream(fileName, fos);
    if (fos.is_open()) {
        fos << initialValue << endl;
        fos.close();
    } else {
        cerr << "[ERROR] Unable to open file" << endl;
        return -1;
    }
    cout << "Initial counter value is " << initialValue << endl;
       
    // Execute increment
    increment(fileName);
    
    // Read new value
    string finalValue;
    ifstream fis;
    compss_ifstream(fileName, fis);
    if (fis.is_open()) {
        if (getline(fis, finalValue)) {
            cout << "Final counter value is " << finalValue << endl;
            fis.close();
        } else {
            cerr << "[ERROR] Unable to read final value" << endl;
            fis.close();
            return -1;
        }
    } else {
        cerr << "[ERROR] Unable to open file" << endl;
        return -1;
    }

    FILE* test = compss_fopen(fileName, "r");

    int c;
    //if (test) {
    //    while ((c = getc(test)) != EOF)
    //        putchar(c);
    //    fclose(test);
    //}
    c = getc(test);
    putchar(c);

    fclose(test);

    // Close COMPSs and end
    compss_off();
    return 0;
}
