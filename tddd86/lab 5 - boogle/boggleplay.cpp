/*marso329 joher316 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "Boggle.h"
#include "bogglemain.h"
#include "strlib.h"
const string alphabet  = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/*
 * Plays one game of Boggle using the given boggle game state object.
 */

//checks the user input when he/she enters a own board, it goes through each letter and checks if it a substring of alphabet. also checks the size of the input
vector<string> checkInput(string input){
    vector<string> temp;
    if (input.size()!=16){
        return temp;
    }
    else{
        for (int i=0;i<16;i++){
            string substring;
            substring=input.substr(i,1);
            if (alphabet.find(substring) != string::npos) {
                temp.push_back(substring);
            }
            else{
                return temp;
            }

        }
    }
    return temp;
}

//simply prints out the board
void printBoard(Boggle& boggle){
    for (int i=0;i<4;i++){
        for (int j=0;j<4;j++){
            cout<<boggle.getLetter(i,j);
        }
        cout<<endl;
    }
    cout<<endl;
}

//prints out the users current score
void presentUserScore(Boggle& boggle){
    cout<<"Your score:"<<boggle.getUserScore()<<endl;

}

//prints out the users current words in a nice manner
void presentUserWords(Boggle& boggle){
    vector<string> temp;
    temp=boggle.getUserWords();
    cout<<"your words:"<<"("<<temp.size()<<"): {";
    for (unsigned int i=0;i<temp.size();i++){
        cout<<temp[i]<<" ";
        if (i%4==3){
            cout<<endl;
        }
    }
    cout<<"}"<<endl;
}

//if the user finds a new word this is used to present it.
void presentNewWord(string word){
    cout<<"you found a new word: "<<word<<endl;
}

//prints out all the computers word, depending on the userscore the output differs
void presentComputersWords(Boggle& boggle){
    vector<string> temp;
    temp=boggle.getComputerWords();
    cout<<"My words(all "<<temp.size()<<" of them):";
    cout<<"{";
    int score=0;
    for (unsigned int i=0;i<temp.size();i++){
        score=score+temp[i].size()-3;
        cout<<'"'+temp[i]+'"'+",";
        if (i%8==0){
            cout<<endl;
        }
    }
    cout<<endl;
    if (score>boggle.getUserScore()){
        cout<<"my score is "<<score<<" which is more than yours"<<endl;

    }
    else if (score==boggle.getUserScore()){
        cout<<"my score is "<<score<<" which is the same as yours, i guess you do have a gig of ram"<<endl;
    }
    else{
        cout<<"my score is "<<score<<" which is the less than yours, that is impossible"<<endl;
    }
}

//mainfunctions of the game
void playOneGame(Boggle& boggle) {
    string input="";
    //runs until a board is generated
    while (1){
        cout<<endl;
        cout<<"Do you want to generate a random board?: ";
        cin>>input;
        //the user wants to enter an own board
        if (input=="n"){
            //runs until the user has entered a correct board,
            while(1){
                cout<<"Type the 16 letters to appear on the board:";
                cin>>input;
                vector<string> temp;
                temp=checkInput(input);
                if (temp.size()==16){
                    for (int i=0;i<16;i++){
                        boggle.addLetter(temp[i]);
                    }
                    break;
                }
                else{
                    cout<<"That is not a valid 16-letter board String. Try again."<<endl;
                }

            }
            boggle.calculateWords();
            break;
        }
        //if the user wants to autogenerate a board this i used
        if(input=="y"){
            boggle.setupBoardAuto();
            break;
        }
    }
    bool usersTurn=true;
    clearConsole();
    //runs until the user enters "n"
    while (usersTurn){
        cout<<"it's your turn!"<<endl;
        printBoard(boggle);
        presentUserWords(boggle);
        presentUserScore(boggle);
        input="";
        cout<<"enter a word or  enter n to end your turn:";
        cin>>input;
        //end while loop
        if (input=="n"){
            usersTurn=false;
        }
        else{
            //checks if the word is correct
            if (boggle.addUserWord(input)){
                //if it is then the score is added to the userscore
                boggle.addUserScore(input.size()-3);
                //says that you have found a new word
                presentNewWord(input);
            }
            else{
                //if it not a correct word then the error is printed out for the user
                cout<<boggle.getUserError()<<endl;
            }
        }
    }
    //prints out with how much the computer has won
    cout<<"It is my turn! now you are going down"<<endl;
    presentComputersWords(boggle);
    boggle.resetBoard();

}

/*
 * Erases all currently visible text from the output console.
 */
void clearConsole() {
#if defined(_WIN32) || defined(_WIN64)
    std::system("CLS");
#else
    // assume POSIX
    std::system("clear");
#endif
}
