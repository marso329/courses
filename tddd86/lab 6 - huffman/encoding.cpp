
#include "encoding.h"
#include <queue>          // std::priority_queue
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>


//counts the number of times each elements occur in the stream
map<int, int> buildFrequencyTable(istream& input) {
    map<int, int> freqTable;
    int temp;
    //while not end of file
    while(input.peek()!=EOF){
        //fetch a byte
        temp=input.get();
        freqTable[temp]=freqTable[temp]+1;
    }
    //adds a end of file element
    freqTable[EOF]=1;
    return freqTable;
}

//builds the coding tree
HuffmanNode* buildEncodingTree(const map<int, int> &freqTable) {

    //own comparator struct
    struct LessThanByOccurance: public std::binary_function<HuffmanNode*, HuffmanNode*, bool>
    {
        bool operator()(const HuffmanNode* lhs, const HuffmanNode* rhs) const
        {
            return lhs->count > rhs->count;
        }
    };

    //create a priority queue with a vector as base
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, LessThanByOccurance> pq;
    for (map<int,int>::const_iterator it=freqTable.begin(); it!=freqTable.end(); ++it){
        //creates a huffmannode for each character and adds it to the queue
        HuffmanNode* temp=new HuffmanNode;
        temp->character=(it->first);
        temp->count=(it->second);
        pq.emplace(temp);
    }
    //while the queue is not empty
    while (!pq.empty()){
        //if the queue has more than two objects
        if (pq.size()>=2){
            //create a new node
            HuffmanNode* temp=new HuffmanNode;
            //takes the first element in queue and lets it be the zero in the new node
            temp->zero=pq.top();
            pq.pop();
            //takes the first element in queue and lets it be the one in the new node
            temp->one=pq.top();
            pq.pop();
            //the new node count is the sum
            temp->count=temp->one->count+temp->zero->count;
            //puts it back in the queue
            pq.emplace(temp);

        }
        else{
            //if there is only one node in the queue then we return it as the root to the tree
            HuffmanNode* temp=pq.top();
            return temp;
        }

    }

    return nullptr;
}


void buildRecursive(HuffmanNode* node,string position,map<int,string>* toReturn){
    //if the node is a leaf then we add the nodes character to the map with the path here as key
    if (node->isLeaf()){
        (*toReturn)[node->character]=position;
    }
    else{
        if(node->one){
            //we search the right tree
            buildRecursive(node->one,position+"1",toReturn);
        }
        if(node->zero){
            //we search the left tree
            buildRecursive(node->zero,position+"0",toReturn);
        }
    }

}

//build the encoding map by traversing the tree
map<int, string> buildEncodingMap(HuffmanNode* encodingTree) {
    //creates a map
    map<int, string> encodingMap;
    //creates a pointer to the map so the recursion adds element to this map
    map<int,string>* encodingMapPointer=&encodingMap;
    //traverse the tree
    buildRecursive(encodingTree,"",encodingMapPointer);
    //return the map
    return encodingMap;
}

//encodes th input using encoding map and sends it to output
void encodeData(istream& input, const map<int, string> &encodingMap, obitstream& output) {
    map<int,int>::const_iterator it;
    int temp;
    string encoding;
    //as long as the next element is not a end of file
    while(input.peek()!=EOF){
        temp=input.get();
        //gets the encoding of the char
        encoding=(encodingMap.find(temp)->second);
        //writes each bit
        while (encoding.size()>0){
            if (encoding[0]=='0'){
                output.writeBit(0);
            }
            else{
                output.writeBit(1);
            }
            encoding.erase(0,1);
        }
    }
    //writes and of file
    encoding=(encodingMap.find(EOF)->second);
    while (encoding.size()>0){
        if (encoding[0]=='0'){
            output.writeBit(0);
        }
        else{
            output.writeBit(1);
        }
        encoding.erase(0,1);
    }
}

//decodes the input using the encodingtree and puts it on the output
void decodeData(ibitstream& input, HuffmanNode* encodingTree, ostream& output) {
    bool end=false;
    HuffmanNode* currentNode=encodingTree;
    int currentBit;
    int outputChar;
    while (!end){
        //we are on a leaf
        if (currentNode->isLeaf()){
            //get the outputchar
            outputChar=currentNode->character;
            //if the output char is eof we quit
            if (outputChar==EOF){
                end=true;
                break;
            }
            //we put that char on the output
            output.put(outputChar);
            //set the current node to the root
            currentNode=encodingTree;

        }
        else{
            //if where are not on a leaf we read a bit and go wherever that bit tells us to go
            currentBit=input.readBit();
            if (currentBit==0){
                currentNode=currentNode->zero;
            }
            else{
                currentNode=currentNode->one;
            }

        }
    }
}

//converts a vector of bytes to a int
int bytesToInt(vector<unsigned char> s){
    int toReturn=0;
    int shifter=0;
    for (unsigned int i=0;i<s.size();i++){
        toReturn=toReturn+(s[i]<<shifter);
        shifter=shifter+8;
    }
    return toReturn;
}


//converts a 32bit int into four or less chars in a vector
vector<unsigned char> intToBytes(int s){
    vector<unsigned char> toReturn;
    vector<unsigned char> input;
    input.push_back( (s & 0x000000ff));
    input.push_back( (s & 0x0000ff00) >> 8);
    input.push_back( (s & 0x00ff0000) >> 16);
    input.push_back( (s & 0xff000000) >> 24);
    int temp;
    if (input[3]!=0){
        temp=3;
    }
    else if (input[2]!=0){
        temp=2;
    }
    else{
        temp=1;
    }
    for (int i=0;i<=temp;i++){
        toReturn.push_back(input[i]);
    }
    return toReturn;

}



//writes the header to the file
void writeHeaderNew(const map<int,int> &freqTable,obitstream& output){
    //for each element int the freqtable
    for (map<int,int>::const_iterator it=freqTable.begin(); it!=freqTable.end(); ++it){
        //gets the key in the map
        int key=it->first;
        //special case, there can only be one -1 in each file and we signal this char in this way
        if (key==-1){
            output.put(1);
            output.put(':');
            output.put(1);
            output.put(0);


        }
        //another special case, this however can occur multiple times in a file so we much send the number of occurances
        else if (key==255){
            //signal that this is special
            output.put(1);
            output.put(':');
            output.put(2);
            output.put(0);
            output.put('|');
            //send the number of occurances
            if ((it->second)>254){
                //if we need more than one byte
                vector<unsigned char> intBytes;
                intBytes=intToBytes(it->second);
                for (int i=0;i<intBytes.size();i++){
                    output.put(intBytes[i]);
                }
            }
            else{
                //if the number can fit in one byte
                output.put((it->second));
            }

        }

        else{
            //the normal case for each character between 0 and 254
            //send the key
            output.put(key);
            output.put(':');

            int temp=(it->second);
            //if it fit in one byte
            if (temp<255){
                output.put(it->second);
            }
            //else we split it up into more bytes
            else{
                vector<unsigned char> intBytes;
                intBytes=intToBytes(temp);
                for (unsigned int i =0;i<intBytes.size();i++){
                    //the lsb comes first
                    output.put(intBytes[i]);
                }
            }}
        //used to separate data
        output.put('|');

    }
    //used to signal that the header is over
    output.put('\0');
    output.put(':');
    output.put('\0');
    output.put('\0');
    output.put('|');
}

//used to read the file header
map<int,int> readHeaderNew(istream& input){
    map<int, int> freqTable;
    int value;
    unsigned char key;
    bool end=false;
    //while there is still some header
    while (!end){
        //get the char
        key=input.get();
        //get separator
        input.get();
        //get the number of occurances
        value=(unsigned char)input.get();
        //if the next char is not ; then there are more bytes for the value
        if (input.peek()!='|'){

            //we fetch the and convert them to an int
            vector<unsigned char> toConvert;
            toConvert.push_back(value);
            while (input.peek()!='|'){
                toConvert.push_back(input.get());
            }
            value=bytesToInt(toConvert);
            //here we add it to the map
            if ((value!=0)&&(value!=1)&&(value!=2)){
                freqTable[key]=value;
            }
            //the number 1 in two bytesm seems like a waste but is used to signal a special case
            else if(value==1){
                freqTable[-1]=1;
            }
            //the number 2 in two bytes also seems like a waste but is used to signal that char 255
            else if(value==2){
                //fetch seperator
                input.get();
                //get the number of occurances for 255
                value=input.get();
                if (input.peek()!='|'){
                    vector<unsigned char> toConvert;
                    toConvert.push_back(value);
                    while (input.peek()!='|'){
                        toConvert.push_back(input.get());
                    }
                    value=bytesToInt(toConvert);

                }
                //adds it to the map
                freqTable[255]=value;
            }
            else{
                //if we send 0 in two bytes that signals that the header is over
                end=true;
            }
        }
        //the normal case, just add the new char to the map with the value
        else{
            freqTable[key]=value;
        }
        //get separator
        input.get();
    }
    return freqTable;
}


//uses the above functions to compress the files
void compress(istream& input, obitstream& output) {
    map<int,int> freqTable=buildFrequencyTable(input);
    HuffmanNode* tree=buildEncodingTree(freqTable);
    map<int,string> encodingmap=buildEncodingMap(tree);
    writeHeaderNew(freqTable,output);
    input.clear();
    input.seekg(0, ios::beg);
    encodeData(input,encodingmap,output);


}

//uses functions above to decompress the file
void decompress(ibitstream& input, ostream& output) {
    map<int, int> freqTable=readHeaderNew(input);
    HuffmanNode* tree=buildEncodingTree(freqTable);
    decodeData(input,tree,output);
}

void freeTree(HuffmanNode* node) {
    struct help{
        void freeRecursive(HuffmanNode* node){
            if (node->zero->isLeaf()){
                delete node->zero;
            }
            else{
                freeTree(node->zero);
            }
            if (node->one->isLeaf()){
                delete node->one;
            }
            else{
                freeTree(node->one);
            }

        }

    };
    help helper;

    helper.freeRecursive(node);
    delete node;

}
