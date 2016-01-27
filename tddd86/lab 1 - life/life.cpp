// This is the CPP file you will edit and turn in.
// Also remove these comments here and add your own.
// TODO: remove this comment header

#include <iostream>
#include "grid.h"
#include "lifeutil.h"
#include<fstream>
#include <string>
#include <cstdlib>
using namespace std;

void printIntroduction(){
    cout<<endl<<"Welcome to the TDDD86 Game of Life,"<<endl<<
            "a simulation of the lifecycle of a bacteria colony."<<endl<<
            "Cells (X) live and die by the following rules:"<<endl<<
            " - A cell with 1 or fewer neighbours dies."<<endl<<
            " - Locations with 2 neighbours remain stable."<<endl<<
            " - Locations with 3 neighbours will create life."<<endl<<
            " - A cell with 4 or more neighbours dies."<<endl<<endl;
}
void getFileName(string& fileName){
    cout<<endl<<"Grid input file name?";
    cin>>fileName;
}
void getFileData(string& fileName,int& rows,int& columns,Grid<char>& world){
    ifstream input (fileName);
    string line;
    getline (input,line);
    rows =atoi(line.c_str());
    getline (input,line);
    columns =atoi(line.c_str());
    world.resize(rows,columns);
    for (int i=0;i<rows;i=i+1){
        getline (input,line);
        for (int j=0;j<columns;j=j+1){
            world[i][j]=line[j];
        }
    }
    input.close();
}
void printWorld(Grid<char>& world){
    int rows=world.numRows();
    int columns=world.numCols();
    for (int i=0;i<rows;i=i+1){
        cout<<endl;
        for (int j=0;j<columns;j=j+1){
            cout<<world[i][j];
        }
    }
    cout<<endl;
}
int liveOrDead(Grid<char>& world,int row,int column){
    int rows=world.numRows();
    int columns=world.numCols();
    if (row>rows-1 or row<0){
        return 0;
    }
    else if (column>columns-1 or column<0){
            return 0;
        }
    else {
        return world[row][column]=='X';

    }
}

void calculateNewGen(Grid<char>& world){
    int rows=world.numRows();
    int columns=world.numCols();
    int temp=0;
    vector< vector<int> > pos(8,vector<int>(2));
    pos[0][0]=pos[0][1]=pos[1][0]=pos[2][0]=pos[3][1]=pos[6][1]=1;
    pos[1][1]=pos[3][0]=pos[4][0]=pos[4][1]=pos[5][0]=pos[7][1]=-1;
    pos[2][1]=pos[5][1]=pos[6][0]=pos[7][0]=0;
    Grid<char> tempWorld(rows,columns);
    for (int i=0;i<rows;i=i+1){
        for (int j=0;j<columns;j=j+1){
            temp=0;
            for (int k=0;k<8;k=k+1){
                temp=temp+liveOrDead(world,i+pos[k][0],j+pos[k][1]);
            }

            if (temp==2 and world[i][j]=='X'){
                tempWorld[i][j]='X';
            }
            else if (temp==3){
                tempWorld[i][j]='X';
            }
            else{
                tempWorld[i][j]='-';
            }
        }
    }
    world=tempWorld;
}

int main() {

string fileName="";
int rows=0;
int columns=0;
Grid<char> world(1,1);
string choice="a";

printIntroduction();
getFileName(fileName);
getFileData(fileName,rows,columns,world);
clearConsole();
printWorld(world);
while (choice!="q"){
    cout<<"a)nimate, t)ick, q)uit?"<<"";
    cin>>choice;
    if (choice=="t"){
calculateNewGen(world);
clearConsole();
printWorld(world);
    }
    if (choice=="a"){
        for (int i=0;i<11;i=i+1){
            calculateNewGen(world);
            clearConsole();
            printWorld(world);
            pause(100);
        }
    }
}
    std::cout << "Have a nice Life!" << std::endl;
    return 0;
}
