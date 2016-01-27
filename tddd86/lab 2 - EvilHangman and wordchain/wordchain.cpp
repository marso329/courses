/*an implement of wordchain by marso329 and joher316
 */
#include <fstream>
#include <iostream>
#include <string>
#include <queue>
#include <stack>
#include <map>
using namespace std;

const string alphabet  = "abcdefghijklmnopqrstuvwxyz";


stack<string> myChain(string& first_word,string& second_word){

    queue<stack<string>> myQueue;

    ifstream myfile("dictionary.txt");

    string word;

    map<string,string> wordList;
    map<string,string> usedWords;
    usedWords[first_word]=first_word;

    while(getline(myfile,word)){
        wordList[word]=word;
    }

    stack<string> firstStack;

    firstStack.push(first_word);

    myQueue.push(firstStack);

    while(!myQueue.empty()){
        //cout<<word;
        stack<string> temp;
       temp = myQueue.front();
       myQueue.pop();
       if (temp.top() == second_word){
           return temp;
       }
       else{
           for(int i=0; i<first_word.length();i=i+1){
               for(int j=0; j < alphabet.length();j=j+1){
                   word=temp.top();
                   word[i]=alphabet[j];
                   //cout<<word;
                   if(wordList.count(word)==1 && usedWords.count(word)==0){
                       //cout<<word;
                       stack<string> copyStack;
                       copyStack = temp;
                       copyStack.push(word);
                       myQueue.push(copyStack);
                       usedWords[word]=word;

                   }
               }
           }
       }
    }
    //return stack<string>;
}
int main() {
    string first_word;
    string second_word;

    cout << "Welcome to TDDD86 Word Chain." << endl;
    cout << "If you give me two English words, I will transform the" << endl;
    cout << "first into the second by changing one letter at a time." << endl;
    cout << endl;


    cout << "Please type two words: ";
    cin>>first_word;
    cin>>second_word;
    cout << "Chain from " << second_word;
    cout << " back to " << first_word << endl;
    stack<string> printstack;
    printstack=myChain(first_word,second_word);
    while(!printstack.empty()){
        cout<<printstack.top()<<endl;
              printstack.pop();
    }

    // TODO: Finish the program!

    return 0;
}
