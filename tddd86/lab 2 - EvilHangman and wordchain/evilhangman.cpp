/*an implement of evil hangman by marso329 and joher316
 */
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <cctype>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
//aplhabet
const string alphabet  = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

//check if a string can be converted to a int
bool stringIsInt(string temp){
    bool hasOnlyDigits = true;
    for (size_t n = 0; n < temp.length(); n++)
    {
        if (!isdigit( temp[ n ] ))
        {
            hasOnlyDigits = false;
            break;
        }
    }
    if (0==atoi(temp.c_str())){

        hasOnlyDigits = false;
    }
    return hasOnlyDigits;
}

//reads in the dictionary to a multimap, the key is the length of the word
void getDictionary(multimap<int,string>& dict){
    ifstream myfile("dictionary.txt");
    //ifstream myfile("di.txt");
    string word;
    while(getline(myfile,word)){
        dict.insert(pair<int,string>(word.length(),word));
    }
}

//gets a in from the user using stringIsInt to check if input can be converted
void getIntInput(int& s){
    string input;
    cin>>input;
    while (!stringIsInt(input)) {
        cout << "ERROR, enter a number greater than 0:";
        cin.clear();
        cin >> input;
    }
    s= atoi(input.c_str());

}

//compares if a word is a part of a wordfamiliy for example if good is a part of the family ---l which is it not
bool compareWords(string word1,string word2,string firstGroup,string currentChar){
    if (word1==firstGroup &&word2.find(currentChar)<word2.size()){
        return false;
    }
    if (word1.length()!=word2.length()){
        return false;
    }
    for (size_t i=0;i<word1.length();i=i+1){
        if (word1[i]!='-'){
            if (word1[i]!=word2[i]){
                return false;
            }
        }
        if(word1[i]=='-' && word2[i]==currentChar[0]){
            return false;
        }
    }
    return true;
}

//presents all the information for the user between each guess
void presentCurrentStatus(int& guesses,string& currentWord,int wordsLeft,string& showWords,string& guessedLetters){
    cout<<"you have "<<guesses<<" guesses left."<<endl;
    cout<<"you previous guesses are: ";
    for (size_t i=0;i<guessedLetters.size();i=i+1){
        cout<<guessedLetters[i]<<" ";
    }
    cout<<endl;
    cout<<"the current words is: "<<currentWord<<endl;
    if (showWords=="y"){
        cout<<"the current number of words left is: "<<wordsLeft<<endl;
    }

}

//gets a letter from the user, if the user has already guesses on that words it asks the user again
void getCharFromUser(string& currentChar,string& guessedLetters,string alphabet){
    cout<<"enter a character: ";
    cin>>currentChar;
    while (currentChar.size()>1 ||alphabet.find(currentChar)>(alphabet.size()-1)|| guessedLetters.find(currentChar)<guessedLetters.size() ){
        cout<<"ERROR enter a valid character:";
        cin>>currentChar;
        cout<<endl;
    }
    guessedLetters.append(currentChar);
}

/*creates a vector of wordfamilies recursive. For example if the current word is ---- and the user guesses on a the vector will contain
----,a---,-a--,aa-- and so on
*/
void createGroups(string word,string letter,vector<string>& groups,int pos){
    if(pos==0){
        groups.push_back(word);
    }
    string temp;
    if (!((pos+1)> word.size())){
        if (word[pos]=='-'){
            int tempInt=groups.size();
            for (int i=0;i<tempInt;i=i+1){
                temp=groups[i];
                temp.replace(pos,1,letter);
                groups.push_back(temp);

            }
        }
        createGroups(word,letter,groups,pos+1);
    }
}

//creates a new vector with correct words after a guess,also does th checks if the user has won
void calculateWordFamilies(vector<string>& wordsOfRightSize,string& currentChar,string& currentWord,int&  guesses,bool& won){
    vector<string> groups;
    int temp=0;
    //step 1:create the new groups
    createGroups(currentWord,currentChar,groups,0);

    //creates  map that keeps track of the size of the families
    map<string,int> values;
    for(vector<string>::const_iterator i = groups.begin(); i != groups.end(); ++i) {
        values[*i]=0;
    }

    //creates a multimap that will contain the families
    multimap<string,string> families;

    //goes through every words and adds it the groups that it fits, one words can be in more than one group
    for(vector<string>::const_iterator i = wordsOfRightSize.begin(); i != wordsOfRightSize.end(); ++i) {
        for(vector<string>::const_iterator j = groups.begin(); j != groups.end(); ++j) {
            if (compareWords(*j,*i,groups[0],currentChar)){
                families.insert(pair<string,string>(*j,*i));
                values[*j]=values[*j]+1;

            }

        }
    }

    //picks out the largest family
    string maxFamily;
    int max=0;
    for(vector<string>::const_iterator i = groups.begin(); i != groups.end(); ++i) {
        if (values[*i]>max){
            maxFamily=*i;
            max=values[*i];
        }
    }
    //adds all the words in the largest family the the vector wordsOfRightSize
    std::pair <std::multimap<string,string>::iterator, std::multimap<string,string>::iterator> ret;
    ret = families.equal_range(maxFamily);
    wordsOfRightSize.clear();
    for (std::multimap<string,string>::iterator it=ret.first; it!=ret.second; ++it){
        wordsOfRightSize.push_back(it->second);
    }

    //if the currentword has not changed the user will loose a guess
    if(currentWord==maxFamily && maxFamily.size()>0 ){
        guesses=guesses-1;
    }

    //sets the currentWord to the largest group
    if (maxFamily.size()>0){
        currentWord=maxFamily;
    }

    //if the curretnword does not contain any "-" the user has found the word
    if(currentWord.find("-")>currentWord.size()){
        won=true;
    }
}

//tells the user that he/she has lost
void youLost(vector<string>& wordsOfRightSize){
    cout<<"you are out of guesses, the right word was: "<<wordsOfRightSize[0]<<endl;
}

void youWon(string currentWord){
    cout<<"congratulations, you bet me. The word was:"<<currentWord<<endl;
}

//mainloop
int main() {

    cout << "Welcome to Hangman." << endl;

    //variables
    multimap<int,string> dict;
    string guessedLetters;
    int wordLength;
    int guesses;
    string currentWord;
    string showWords;
    bool won=false;
    vector<string> wordsOfRightSize;
    string currentChar;
    string repeat="y";

    //1.copies information from file to dict
    getDictionary(dict);
    while (repeat=="y"){
        //2.ask user for wordlenght
        cout << "Enter wordlength:";
        getIntInput(wordLength);
        while (dict.count(wordLength)==0){
            cout<<"ERROR no word of that length, enter new wordlength:";
            getIntInput(wordLength);
        }

        //3.asks user for number of guesses
        cout << "Enter number of guesses:";
        getIntInput(guesses);

        //4. asks user if he or she wants to see the remaining number of possible words
        cout << "Do you want to see remaining possible words after each guess y)es or n)o:";
        cin>>showWords;

        //5.a copies all words of correct length to a vector
        std::pair <std::multimap<int,string>::iterator, std::multimap<int,string>::iterator> ret;
        ret = dict.equal_range(wordLength);
        for (std::multimap<int,string>::iterator it=ret.first; it!=ret.second; ++it){
            wordsOfRightSize.push_back(it->second);
        }
        for (int i=0;i<wordLength;i=i+1){
            currentWord.push_back('-');
        }
        while(!won &&guesses>0){
            //5.b
            presentCurrentStatus(guesses,currentWord,wordsOfRightSize.size(),showWords,guessedLetters);

            //5.c
            getCharFromUser(currentChar,guessedLetters,alphabet);

            //5.d-e
            calculateWordFamilies(wordsOfRightSize,currentChar,currentWord,guesses,won);

            //5.f
            if (guesses<1){
                youLost(wordsOfRightSize);
            }

            //5.g
            if (won){
                youWon(currentWord);
            }
        }
        cout<<"you want to play again? y)es or n)o:";
        cin>>repeat;
        cout<<endl;
        won=false;
        guessedLetters.clear();
        currentWord.clear();
    }
    return 0;
}
