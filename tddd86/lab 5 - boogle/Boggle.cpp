/*marso329 joher316*/

#include <sstream>
#include "Boggle.h"
#include "random.h"
#include "shuffle.h"
#include "strlib.h"
#include <vector>
#include <list>
#include <algorithm>
#include <random>

using namespace std;
static const int NUM_CUBES = 16;   // the number of cubes in the game
static const int CUBE_SIDES = 6;   // the number of sides on each cube
static string CUBES[NUM_CUBES] = {        // the letters on all 6 sides of every cube
                                          "AAEEGN", "ABBJOO", "ACHOPS", "AFFKPS",
                                          "AOOTTW", "CIMOTU", "DEILRX", "DELRVY",
                                          "DISTTY", "EEGHNW", "EEINSU", "EHRTVW",
                                          "EIOSST", "ELRTTY", "HIMNQU", "HLNNRZ"
                                 };
//used to go through all nearby letters
static int ROWCHANGE[8]={
    1,1,1,-1,-1,-1,0,0
};
//used to go through all nearby letters
static int COLUMNCHANGE[8]={
    1,-1,0,1,-1,0,1,-1
};

Boggle::Boggle(){
    //creates the dictionary
    lex=Lexicon(DICTIONARY_FILE);
    //initialise the board
    board.letters=std::vector< std::vector< string >> ( BOARD_SIZE, std::vector<string> ( BOARD_SIZE, "" ) );
    board.rows=BOARD_SIZE;
    board.columns=BOARD_SIZE;

}

//uses a rng to pick letters from the cubes and then shuffels the vectors in the board struct and then shuffels the strings in each vector, ends with backtracking the whole board
void Boggle::setupBoardAuto(){
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(0, CUBE_SIDES-1);
    for (int i=0;i<NUM_CUBES;i++){
        addLetter(CUBES[i].substr(distr(eng),1));}
    shuffle(board.letters);
    for (int i=0;i<BOARD_SIZE;i++){
        shuffle(board.letters[i]);
    }
    allCorrectWords=backtracking();

}

//used when the board is entered manually, just takes all the words from backtracking and puts them in allCorrectWords
void Boggle::calculateWords(){
    allCorrectWords=backtracking();
}

//clears the boardsstruct, removes the users words and score,removes all words created with backtracking
void Boggle::resetBoard(){
    board.letters=std::vector< std::vector< string >> ( BOARD_SIZE, std::vector<string> ( BOARD_SIZE, "" ) );
    board.size=0;
    userScore=0;
    usersWord.clear();
    allCorrectWords.clear();
}


//returns the last usererror
string Boggle::getUserError(){
    return userWordError;
}


//returns true if a string is in allCorrectWords and longer than 4
bool Boggle::isCorrectWord(string word){
    return (((std::find(allCorrectWords.begin(), allCorrectWords.end(), (word)) != allCorrectWords.end()))&&(word.size()>=4));
}

//checks if a word is already guessed, if it is long enough and its is a correct word. If it is then it is added to the users word and true is returned, is it not then a error is set and false is returned
bool Boggle::addUserWord(string word){

    if(!((std::find(usersWord.begin(), usersWord.end(), (word)) != usersWord.end()))){
        if(word.size()>=4){
            if (isCorrectWord(word)){
                usersWord.push_back(word);
                return true;
            }
            else{
                userWordError="That word is not in the dictionary.";
                return false;
            }
        }
        else{
            userWordError="That word is not long enough.";
            return false;
        }
    }
    else{
        userWordError="You have already guessed that word.";
        return false;
    }


}

//returns true if a string is already in usersWord,
bool Boggle::userHasWord(string word){
    return ((std::find(usersWord.begin(), usersWord.end(), (word)) != usersWord.end()));
}

//add a letter to the board
void Boggle::addLetter(string letter){
    int row=board.size/board.rows;
    int column=board.size%board.columns;
    board.letters[row][column]=letter;
    board.size++;

}

//returns all the words that the computer has that the users has not guessed
vector<string> Boggle::getComputerWords(){
    vector<string> temp;
    for (unsigned int i=0;i<allCorrectWords.size();i++){
        if (!userHasWord(allCorrectWords[i])){
            temp.push_back(allCorrectWords[i]);
        }
    }
    return temp;


}

//adds score the to the userscore
void Boggle::addUserScore(int score){
    userScore=userScore+score;
}

//returns the userscure
int Boggle::getUserScore(){
    return userScore;
}

//returns what letter is a position row,column
string Boggle::getLetter(int row,int column){
    if ((row>-1)&&(row<board.rows)&&(column>-1)&&(column<board.columns)){
        return board.letters[row][column];
    }
    else{
        return "";
    }
}

//retuns the users word
vector<string> Boggle::getUserWords(){
    return usersWord;
}

//a recursive function that searches through the board, in this case 16 of these functions are run simultaneously, might be a little overkill but cool.
void backtrackingSub(string currentWord,Board board,vector<vector<bool>> visited,vector<string>* toReturn,Lexicon &lex,int currentRow,int currentColumn){
    //help struct to create a local function
    struct Help{
        string getLetter(int row,int column,Board board){
            if ((row>-1)&&(row<board.rows)&&(column>-1)&&(column<board.columns)){
                return board.letters[row][column];
            }
            else{
                return "";
            }
        }
    };
    Help helper;
    //go through all nearby letters
    for (int i=0;i<8;i++){
        //get the checkPosition
        int rowCheck=currentRow+ROWCHANGE[i];
        int columnCheck=currentColumn+COLUMNCHANGE[i];
        //get the letter from that position
        string newLetter=helper.getLetter(rowCheck,columnCheck,board);
        //if there are a letter there and we have not been there before we can check it out
        if ((newLetter!="")&&!visited[rowCheck][columnCheck]){
            //create a new word
            string tempWord=currentWord+newLetter;
            //if this word i a correct word
            if ((tempWord.size()>=4)&&(lex.contains(tempWord))){
                //we add it to the toReturn vector
                (*toReturn).push_back(tempWord);
            }
            //if lex contains a word with this prefix we evaluate is some more
            if(lex.containsPrefix(tempWord)){
                //create a new visisted vector
                vector<vector<bool>> visitedNew;
                visitedNew=visited;
                //set this place to visited
                visitedNew[rowCheck][columnCheck]=true;
                //we check if there are any more words that are longer
                backtrackingSub(tempWord,board,visitedNew,toReturn,lex,rowCheck,columnCheck);
            }
        }
    }


}



//create all threads and collect all words that they have found
vector<string> Boggle::backtracking(){

    //create a lsit that will store pointers to all threads
    list<thread*> threads;

    //a vector that will store all vectors that will be used in the threads
    vector<vector<string>> words;
    vector<vector<string>*> wordsNew;
    //create all the threads
    for (int i=0;i<board.rows;i++){
        for (int j=0;j<board.columns;j++){

            //current letter
            string currentWord=getLetter(i,j);

            //if there are any words that starts with this letter we start a thread for that letter
            if (lex.containsPrefix(currentWord)){

                //a vector that tells the thread which letter that have been visited
                vector<vector<bool>> visited;
                visited=std::vector< std::vector< bool >> ( 4, std::vector<bool> ( 4, false ) );
                visited[i][j]=true;

                //create a vector to be used in the thread to store found words
                vector<string> toReturn;
                vector<string> *toReturnNew =new vector<string>;
                //stores this vector in words
                wordsNew.push_back(toReturnNew);
                //keep track of current row;
                int currentRow=i;

                //keep track of current column
                int currentColumn=j;
                //create a thread
                thread* temp= new  thread(backtrackingSub,currentWord,ref(board),visited,toReturnNew,ref(lex),currentRow,currentColumn);

                //store this thread in a list of threads
                threads.push_back(temp);
            }
        }
    }
    for (std::list<thread*>::iterator it = threads.begin(); it != threads.end(); it++){
        (*it)->join();
    }
    vector<string> allWords;
    for (unsigned int i=0;i<wordsNew.size();i++){
        for (unsigned int j=0;j<(*wordsNew[i]).size();j++){
            if (!(std::find(allWords.begin(), allWords.end(), (*wordsNew[i])[j]) != allWords.end()))
            {
                allWords.push_back((*wordsNew[i])[j]);
            }
        }
    }
    return allWords;

}


