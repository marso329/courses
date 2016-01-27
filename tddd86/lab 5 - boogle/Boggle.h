#ifndef _boggle_h
#define _boggle_h


#include <iostream>
#include <string>
#include "shuffle.h"
#include "lexicon.h"
#include <thread>

using namespace std;

//used to keep track of the board
struct Board{
    vector<vector<string>> letters;
    int size=0;
    int rows=0;
    int columns=0;

};
class Boggle {
public:

    //constructor
    Boggle();

    //destructor
    ~Boggle() = default;

    //adds a letter to the board
    void addLetter(string letter);

    //returns the letter at position row,column
    string getLetter(int row,int column);

   //returns a vector with strings that the users has guessed correct on
    vector<string> getUserWords();

    //returns the users score
    int getUserScore();

    //returns all the words the computer has calculated using backtrack algorithm()
    vector<string> getComputerWords();

    //returns a vector of strings that contains all words in the current board
    vector<string> backtracking();

    //adds the score to the current userscore
    void addUserScore(int score);

    //checks if a string is a word on the board
    bool isCorrectWord(string word);

    //used when the user has entered letter manually to backtrack that board
    void calculateWords();

    //checks if user already has this word
    bool userHasWord(string word);

    //returns the users last error
    string getUserError();

    //resets the board for a new game
    void resetBoard();

    //adds a word to the users collection of words
    bool addUserWord(string word);

    //setups the board automatic
    void setupBoardAuto();


private:

    //the current board
    Board board;

    //the lexicon
    Lexicon lex;

    //all curroct words in the board
    vector<string> allCorrectWords;

    //contains all the users correct words
    vector<string> usersWord;

    //keeps track of the userscore
    int userScore=0;

    //keeps track of what was the users last error
    string userWordError;

    //the size of the board
    const int BOARD_SIZE = 4;

    //the minimum length of words
    const int MIN_WORD_LENGTH = 4;

    //the words file
    const string DICTIONARY_FILE = "EnglishWords.dat";

};

#endif
